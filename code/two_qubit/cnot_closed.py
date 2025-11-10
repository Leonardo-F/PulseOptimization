import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time 
# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
# 启用64位精度
jax.config.update("jax_enable_x64", True)


# 可采用多进程并行计算，对原始评分器进行了优化
from two_transmon_grader import DispersiveCNOTPulseGrader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")


# Create smooth cosine pulse with much smoother transitions
def smooth_cosine_pulse(t):
    """Generate smooth cosine pulse with minimal high frequency components"""
    result = np.zeros_like(t)
    
    # Use only fundamental frequency to minimize derivatives
    result += 1.0 * np.cos(t)
    
    # Very gentle envelope to ensure smooth start and end
    # Use raised cosine window for maximum smoothness
    window = 0.6 * (1 - np.cos(t))  # Hann window - very smooth
    
    result = result * window
    
    return result

# Imaginary part: use simplified orthogonal pulse
def smooth_cosine_pulse_im(t):
    """Generate smooth orthogonal pulse"""
    result = np.zeros_like(t)
    
    # Fundamental frequency, 90 degree phase difference
    result += 1.0 * np.cos(t + np.pi/2)
    
    # Same gentle window
    window = 0.5 * (1 - np.cos(t))  # Hann window
    
    result = result * window
    
    return result

class TwoQubitGRAPE:
    """
    两比特系统GRAPE优化（封闭系统）
    基于色散极限的transmon耦合模型
    """
    def __init__(self, nq_levels, omega1, omega2, omega_d, alpha1, alpha2, 
                 J, lambda_coupling, dt, n_steps):
        """
        参数:
        nq_levels: 每个transmon的能级数
        omega1, omega2: 两个transmon的频率 (rad/s)
        omega_d: 驱动频率 (rad/s)
        alpha1, alpha2: 非谐性 (rad/s)
        J: 比特间耦合强度 (rad/s)
        lambda_coupling: 相对耦合强度
        dt: 时间步长 (s)
        n_steps: 时间步数
        """
        self.nq = nq_levels
        self.n_steps = n_steps
        self.dt = dt
        
        # 系统参数
        # self.omega1 = omega1
        # self.omega2 = omega2
        # self.omega_d = omega_d
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.J = J
        self.lambda_coupling = lambda_coupling
        
        # 失谐
        self.delta1 = omega1 - omega_d
        self.delta2 = omega2 - omega_d
        
        # 构建算符
        self._build_operators()
        
        # 构建静态哈密顿量
        self._build_static_hamiltonian()
        
    def _create_annihilation(self, n):
        """创建湮灭算符"""
        a = jnp.zeros((n, n), dtype=jnp.complex128)
        for i in range(n-1):
            a = a.at[i, i+1].set(jnp.sqrt(i+1))
        return a
    
    def _build_operators(self):
        """构建双比特系统的算符"""
        # 单比特算符
        a1 = self._create_annihilation(self.nq)
        a2 = self._create_annihilation(self.nq)
        I = jnp.eye(self.nq, dtype=jnp.complex128)
        
        # 张量积构建双比特算符
        self.b1 = jnp.kron(a1, I)
        self.b2 = jnp.kron(I, a2)
        self.b1_dag = self.b1.T.conj()
        self.b2_dag = self.b2.T.conj()
        
        # 数算符
        self.n1 = self.b1_dag @ self.b1
        self.n2 = self.b2_dag @ self.b2
        
    def _build_static_hamiltonian(self):
        """构建静态哈密顿量H_0"""
        # 非谐性项: -(alpha/2) * (b†b)^2
        H_anh1 = -(self.alpha1/2) * (self.n1 @ self.n1)
        H_anh2 = -(self.alpha2/2) * (self.n2 @ self.n2)
        
        # 失谐项: (omega - omega_d + alpha/2) * b†b
        H_drift1 = (self.delta1 + self.alpha1/2) * self.n1
        H_drift2 = (self.delta2 + self.alpha2/2) * self.n2
        
        # 比特间耦合: J * (b1†b2 + b1b2†)
        H_coupling = self.J * (self.b1_dag @ self.b2 + self.b1 @ self.b2_dag)
        
        self.H_static = H_drift1 + H_drift2 + H_anh1 + H_anh2 + H_coupling
        
        # 驱动算符
        # H_d_re = 1/2 * [(b1† + b1) + λ(b2† + b2)]
        # H_d_im = i/2 * [(b1† - b1) + λ(b2† - b2)]
        self.H_drive_re = 0.5 * (
            (self.b1 + self.b1_dag) + 
            self.lambda_coupling * (self.b2 + self.b2_dag)
        )
        self.H_drive_im = 0.5j * (
            (self.b1_dag - self.b1) + 
            self.lambda_coupling * (self.b2_dag - self.b2)
        )
    
    def get_hamiltonian(self, omega_re, omega_im):
        """
        获取时间依赖的哈密顿量
        omega_re, omega_im: 实部和虚部驱动幅度 (rad/s)
        """
        H = self.H_static + omega_re * self.H_drive_re + omega_im * self.H_drive_im
        return H
    
    @partial(jit, static_argnums=(0,))
    def propagate_step(self, state, omega_re, omega_im):
        """单步演化"""
        H = self.get_hamiltonian(omega_re, omega_im)
        U = jax.scipy.linalg.expm(-1j * H * self.dt)
        return U @ state
    
    @partial(jit, static_argnums=(0,))
    def forward_propagation(self, pulses, initial_state):
        """前向传播整个脉冲序列"""
        def scan_fn(state, inputs):
            omega_re, omega_im = inputs
            new_state = self.propagate_step(state, omega_re, omega_im)
            return new_state, new_state
        
        final_state, states = jax.lax.scan(
            scan_fn, initial_state, (pulses[:, 0], pulses[:, 1])
        )
        return final_state, states
    
    def _build_target_cnot(self):
        """构建目标CNOT门"""
        # 在计算子空间 {|00⟩, |01⟩, |10⟩, |11⟩} 上的CNOT
        # 基态顺序：|00⟩=0, |01⟩=1, |10⟩=nq, |11⟩=nq+1
        dim = self.nq * self.nq
        U_cnot = jnp.eye(dim, dtype=jnp.complex128)
        
        # CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        idx_00, idx_01 = 0, 1
        idx_10, idx_11 = self.nq, self.nq + 1
        
        # 交换 |10⟩ 和 |11⟩
        U_cnot = U_cnot.at[idx_10, idx_10].set(0)
        U_cnot = U_cnot.at[idx_10, idx_11].set(1)
        U_cnot = U_cnot.at[idx_11, idx_11].set(0)
        U_cnot = U_cnot.at[idx_11, idx_10].set(1)
        
        return U_cnot
    
    @partial(jit, static_argnums=(0,))
    def gate_fidelity(self, pulses, initial_states):
        """
        计算门保真度（平均态保真度）
        initial_states: shape (n_states, dim)
        """
        U_target = self._build_target_cnot()
        
        def compute_fidelity_single(psi0):
            psi_final, _ = self.forward_propagation(pulses, psi0)
            psi_target = U_target @ psi0
            overlap = jnp.vdot(psi_target, psi_final)
            return jnp.abs(overlap)**2
        
        # 对所有初态计算保真度并平均
        fidelities = jax.vmap(compute_fidelity_single)(initial_states)
        return jnp.mean(fidelities)
    
    @partial(jit, static_argnums=(0,))
    def derivative_penalty(self, pulses):
        """计算导数惩罚项，与官方评分器保持一致"""
        N = self.n_steps
        pen = 0.0
        h_d = 2 * jnp.pi * 2.7e6  # 2.7 MHz in rad/s
        A_penalty = 0.1
        
        for j in range(N - 1):
            for k in range(2):  # Only 2 channels now
                diff = pulses[j + 1, k] - pulses[j, k]
                r2 = (diff / h_d) ** 2
                pen += jnp.exp(jnp.minimum(r2, 50.0)) - 1.0
        return A_penalty * pen / N
    
    @partial(jit, static_argnums=(0,))
    def amplitude_penalty(self, pulses):
        """计算幅度惩罚项，与官方评分器保持一致"""
        N = self.n_steps
        pen = 0.0
        h_a = 2 * jnp.pi * 200e6  # 200 MHz in rad/s
        A_penalty = 0.1
        
        for j in range(N):
            for k in range(2):  # Only 2 channels now
                r2 = (pulses[j, k] / h_a) ** 2
                pen += jnp.exp(jnp.minimum(r2, 50.0)) - 1.0
        return A_penalty * pen / N
    
    @partial(jit, static_argnums=(0,))
    def compute_leakage(self, pulses, initial_states):
        """
        计算从计算子空间的泄漏
        
        计算子空间：{|00⟩, |01⟩, |10⟩, |11⟩}
        """
        # 计算每个初态的演化终态
        def evolve_single(psi0):
            final_state, _ = self.forward_propagation(pulses, psi0)
            return final_state
        
        final_states = jax.vmap(evolve_single)(initial_states)
        
        # 计算子空间的投影算符
        dim = self.nq ** 2
        
        # 计算子空间基态的索引
        # |00⟩ -> 0, |01⟩ -> 1, |10⟩ -> nq, |11⟩ -> nq+1
        comp_indices = jnp.array([0, 1, self.nq, self.nq + 1])
        
        # 计算每个终态的泄漏
        def compute_leakage_single(final_state):
            # 计算在计算子空间的布居数
            pop_comp = 0.0
            for idx in comp_indices:
                pop_comp += jnp.abs(final_state[idx])**2
            # 泄漏 = 1 - 计算子空间布居数
            return 1.0 - pop_comp
        
        # 对所有初态计算泄漏并平均
        leakages = jax.vmap(compute_leakage_single)(final_states)
        return jnp.mean(leakages)

    @partial(jit, static_argnums=(0,))
    def cost_function(self, params_flat, initial_states):
        """
        完整的成本函数，包含所有评分标准
        
        根据官方评分标准：
        - 门保真度 (80%权重)
        - 泄漏抑制 (15%权重)
        - 脉冲质量约束 (5%权重)，包括幅度惩罚和导数惩罚
        """
        pulses = params_flat.reshape((self.n_steps, 2))
        
        # 1. 门保真度 (80%权重)
        gate_fidelity = self.gate_fidelity(pulses, initial_states)
        gate_error = 1.0 - gate_fidelity
        
        # 2. 泄漏抑制 (15%权重)
        leakage = self.compute_leakage(pulses, initial_states)
        leakage_score = jnp.maximum(0.0, 1.0 - leakage * 5.0)
        
        # 3. 脉冲质量约束 (5%权重)
        amp_pen = self.amplitude_penalty(pulses)
        der_pen = self.derivative_penalty(pulses)
        total_penalty = amp_pen + der_pen
        penalty_score = jnp.maximum(0.0, 1.0 - total_penalty)
        
        # 综合得分 (越高越好)
        overall_score = (
            0.80 * gate_fidelity +
            0.15 * leakage_score +
            0.05 * penalty_score
        )
        
        # 转换为最小化问题 (成本越低越好)
        cost = 1.0 - overall_score
        
        # 放大以改善数值条件
        return cost * 1e6
    
    def optimize(self, initial_pulses, initial_states, maxiter=200, disp=True):
        """
        使用L-BFGS-B优化脉冲
        
        参数:
        initial_pulses: 初始脉冲猜测 (n_steps, 2)
        initial_states: 用于优化的初态集合 (n_states, dim)
        """
        initial_states = jnp.array(initial_states)
        grad_fn = jit(grad(self.cost_function, argnums=0))
        
        self.iteration = 0
        
        def cost_and_grad(params_flat):
            params_jax = jnp.array(params_flat)
            cost = self.cost_function(params_jax, initial_states)
            gradient = grad_fn(params_jax, initial_states)
            
            if self.iteration % 10 == 0 and disp:
                pulses = params_jax.reshape((self.n_steps, 2))
                
                # 计算所有评分标准
                gate_fidelity = self.gate_fidelity(pulses, initial_states)
                gate_error = 1.0 - gate_fidelity
                
                leakage = self.compute_leakage(pulses, initial_states)
                leakage_score = float(jnp.maximum(0.0, 1.0 - leakage * 5.0))
                
                amp_pen = self.amplitude_penalty(pulses)
                der_pen = self.derivative_penalty(pulses)
                total_penalty = amp_pen + der_pen
                penalty_score = float(jnp.maximum(0.0, 1.0 - total_penalty))
                
                overall_score = (
                    0.80 * gate_fidelity +
                    0.15 * leakage_score +
                    0.05 * penalty_score
                )
                
                grad_norm = float(jnp.linalg.norm(gradient))
                
                print(f"Iter {self.iteration}:")
                print(f"  Gate Fidelity: {gate_fidelity:.6f} (Error: {gate_error:.6f})")
                print(f"  Leakage: {leakage:.6f} (Score: {leakage_score:.6f})")
                print(f"  Penalties: Amp={amp_pen:.6f}, Der={der_pen:.6f}, Total={total_penalty:.6f} (Score: {penalty_score:.6f})")
                print(f"  Overall Score: {overall_score:.6f}, |∇| = {grad_norm:.2e}")
            
            self.iteration += 1
            return float(cost), np.array(gradient, dtype=np.float64)
        
        x0 = initial_pulses.flatten()
        result = minimize(
            cost_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        result.pulses = result.x.reshape((self.n_steps, 2))
        
        # 计算所有评分标准
        gate_fidelity = self.gate_fidelity(result.pulses, initial_states)
        gate_error = 1.0 - gate_fidelity
        
        leakage = self.compute_leakage(result.pulses, initial_states)
        leakage_score = float(jnp.maximum(0.0, 1.0 - leakage * 5.0))
        
        amp_pen = self.amplitude_penalty(result.pulses)
        der_pen = self.derivative_penalty(result.pulses)
        total_penalty = amp_pen + der_pen
        penalty_score = float(jnp.maximum(0.0, 1.0 - total_penalty))
        
        overall_score = (
            0.80 * gate_fidelity +
            0.15 * leakage_score +
            0.05 * penalty_score
        )
        
        # 添加所有评分标准到结果对象
        result.gate_fidelity = float(gate_fidelity)
        result.gate_error = float(gate_error)
        result.leakage = float(leakage)
        result.leakage_score = float(leakage_score)
        result.amplitude_penalty = float(amp_pen)
        result.derivative_penalty = float(der_pen)
        result.total_penalty = float(total_penalty)
        result.penalty_score = float(penalty_score)
        result.overall_score = float(overall_score)
        
        return result

def generate_initial_states(nq_levels):
    """
    生成36个初态（6个单比特基准态的张量积）
    """
    # 单比特基准态
    ket0 = jnp.zeros(nq_levels, dtype=jnp.complex128).at[0].set(1)
    ket1 = jnp.zeros(nq_levels, dtype=jnp.complex128).at[1].set(1)
    ket_plus = (ket0 + ket1) / jnp.sqrt(2)
    ket_minus = (ket0 - ket1) / jnp.sqrt(2)
    ket_plus_i = (ket0 + 1j*ket1) / jnp.sqrt(2)
    ket_minus_i = (ket0 - 1j*ket1) / jnp.sqrt(2)
    
    single_qubit_states = [ket0, ket1, ket_plus, ket_minus, ket_plus_i, ket_minus_i]
    
    # 张量积
    two_qubit_states = []
    for s1 in single_qubit_states:
        for s2 in single_qubit_states:
            state = jnp.kron(s1, s2)
            two_qubit_states.append(state)
    
    return jnp.array(two_qubit_states)

# ============================================================================
# 主程序：封闭系统CNOT门优化
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("两比特CNOT门脉冲优化 - 封闭系统")
    print("="*70)
    
    # 系统参数（题目给定）
    nq_levels = 3
    two_pi = 2 * jnp.pi
    omega1 = two_pi * 4.380e9   # 4.380 GHz
    omega2 = two_pi * 4.614e9   # 4.614 GHz
    omega_d = two_pi * 4.498e9  # 4.498 GHz
    alpha1 = two_pi * 0.210e9   # 210 MHz
    alpha2 = two_pi * 0.215e9   # 215 MHz
    J = two_pi * (-0.003e9)     # -3 MHz
    lambda_coupling = 1.03
    
    # 时间参数
    dt = 0.5e-9  # 0.5 ns
    n_steps = 300  # 150 ns gate time
    
    print(f"\n系统配置:")
    print(f"  比特1频率: {omega1/(two_pi)*1e-9:.3f} GHz")
    print(f"  比特2频率: {omega2/(two_pi)*1e-9:.3f} GHz")
    print(f"  驱动频率: {omega_d/(two_pi)*1e-9:.3f} GHz")
    print(f"  比特间耦合: {J/(two_pi)*1e-6:.1f} MHz")
    print(f"  门时间: {n_steps * dt * 1e9:.0f} ns")
    print(f"  时间步数: {n_steps}")
    
    # 初始化优化器
    grape = TwoQubitGRAPE(
        nq_levels, omega1, omega2, omega_d, alpha1, alpha2,
        J, lambda_coupling, dt, n_steps
    )
    
    # 生成36个初态
    initial_states = generate_initial_states(nq_levels)
    print(f"\n初态数量: {len(initial_states)}")
    
    # Initial pulse guess
    # Strategy: Use smoother pulse shapes to reduce high frequency components
    initial_pulses = np.zeros((n_steps, 2))

    # Time axis (normalized to [0, 2π])
    t = np.linspace(0, 2*np.pi, n_steps)

    # Real part: use smooth cosine function
    envelope_re = smooth_cosine_pulse(t)
    # Normalize and set amplitude
    envelope_re = envelope_re / np.max(np.abs(envelope_re))
    initial_pulses[:, 0] = envelope_re * two_pi * 50e6  # 50 MHz amplitude


    envelope_im = smooth_cosine_pulse_im(t)
    # Normalize and set amplitude
    envelope_im = envelope_im / np.max(np.abs(envelope_im))
    initial_pulses[:, 1] = envelope_im * two_pi * 50e6  # 50 MHz amplitude

    
    # 绘制初始脉冲的波形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time_ns = np.arange(n_steps + 1) * dt * 1e9
    
    ax1.step(time_ns, 
             np.append(initial_pulses[:, 0], initial_pulses[-1, 0]) / (two_pi * 1e6),
             where='post', linewidth=2, color='blue', label='Ω_re')
    ax1.set_ylabel('Ω_re / 2π (MHz)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.step(time_ns,
             np.append(initial_pulses[:, 1], initial_pulses[-1, 1]) / (two_pi * 1e6),
             where='post', linewidth=2, color='red', label='Ω_im')
    ax2.set_ylabel('Ω_im / 2π (MHz)', fontsize=12)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle('初始脉冲波形（自然余弦函数叠加）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initial_pulses_visualization.png', dpi=150)
    print("初始脉冲可视化已保存到: initial_pulses_visualization.png")
    
    print("\n开始GRAPE优化...")
    print("(这可能需要几分钟时间)\n")
    
    result = grape.optimize(
        initial_pulses,
        initial_states,
        maxiter=5000,
        disp=True
    )
    
    print("\n" + "="*70)
    print("优化结果（封闭系统）")
    print("="*70)
    
    # 保存脉冲
    np.save('pulses_closed.npy', result.pulses)
    print(f"\n脉冲已保存到: pulses_closed.npy")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time_ns = np.arange(n_steps + 1) * dt * 1e9
    
    ax1.step(time_ns, 
             np.append(result.pulses[:, 0], result.pulses[-1, 0]) / (two_pi * 1e6),
             where='post', linewidth=2, color='blue', label='Ω_re')
    ax1.set_ylabel('Ω_re / 2π (MHz)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.step(time_ns,
             np.append(result.pulses[:, 1], result.pulses[-1, 1]) / (two_pi * 1e6),
             where='post', linewidth=2, color='red', label='Ω_im')
    ax2.set_ylabel('Ω_im / 2π (MHz)', fontsize=12)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle(f'优化的CNOT脉冲（封闭系统，F={result.gate_fidelity:.6f}）', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pulses_closed_visualization.png', dpi=150)
    print("脉冲可视化已保存到: pulses_closed_visualization.png")
    
    print("\n" + "="*70)
    print("第一阶段完成！")
    print("下一步：将此脉冲作为初值，在开放系统中进行鲁棒性优化")
    print("="*70)


    # 看一下在开放系统上的分数
    # 初始化官方评分器
    grader = DispersiveCNOTPulseGrader(
        nq_levels=3,
        n_steps=300,
        dt=5e-10,          # 0.5 ns
        T1_q1=50e-6,
        T1_q2=50e-6,
        Tphi_q1=30e-6,
        Tphi_q2=30e-6,
        nbar_q1=0.0,
        nbar_q2=0.0,
        sigma_detune_q1_Hz=0.5e6,  # 0.5 MHz
        sigma_detune_q2_Hz=0.5e6,  # 0.5 MHz
        n_shots=10,        # 默认评分shots
        h_a_Hz=200e6,
        h_d_Hz=2.7e6,
        A_penalty=0.1,
        computing_method='parallel' # 评分采用并行计算
    )

    grader.grade_submission(result.pulses, n_shots=10, seed=42, verbose=True)