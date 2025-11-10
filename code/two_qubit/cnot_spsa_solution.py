# cnot_pulse_direct_optimization.py
import os
import time
import numpy as np
from typing import List, Tuple
import json

import matplotlib.pyplot as plt
# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 可采用多进程并行计算，对原始评分器进行了优化
from two_transmon_grader import DispersiveCNOTPulseGrader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")



# 生成初始化脉冲
def generate_initial_pulse(n_steps: int, dt: float, method: str = "gaussian", 
                          target_angle: float = np.pi, seed=42) -> np.ndarray:
    """    
    参数:
        n_steps: 脉冲步数
        dt: 时间步长
        method: 脉冲生成方法 ("gaussian", "random")
        target_angle: 目标旋转角度，gaussian 需要
        seed: 随机种子，默认为42，用于随机脉冲的生成

    返回:
        pulses: shape (n_steps, 2), 单位 rad/s
    """

    if method == "gaussian":
        def gaussian_envelope(n_steps: int, sigma_frac: float = 0.2) -> np.ndarray:
            """
            生成归一化高斯包络（最大值约为1，不做L2/L1归一化），中心在(n_steps-1)/2。
            sigma_frac: 相对于总步数的标准差比例，0.2~0.25较常用
            """
            t = np.arange(n_steps)
            center = 0.5 * (n_steps - 1)
            sigma = sigma_frac * n_steps
            env = np.exp(-0.5 * ((t - center) / sigma) ** 2)
            return env


        def build_area_matched_gaussian(n_steps: int, dt: float, target_angle: float = np.pi) -> np.ndarray:
            """
            基于面积匹配生成I路高斯初值，使得 sum(I)*dt ≈ target_angle。
            Q路置零。
            返回 pulses: shape (n_steps, 2), 单位 rad/s
            """
            env = gaussian_envelope(n_steps, sigma_frac=0.22)
            # 在RWA下 H = (Ω/2) σ_x, 所以Ω T = 目标旋角；此处Ω就是pulses[:,0]本身
            # 我们用面积匹配： sum(Ω)*dt = target_angle
            area = np.sum(env) * dt
            if area < 1e-18:
                raise ValueError("Envelope area too small.")
            amp = target_angle / area  # rad/s
            I = amp * env
            Q = np.zeros_like(I)
            return np.column_stack([I, Q])

        # 生成高斯脉冲
        pulses = build_area_matched_gaussian(n_steps, dt, target_angle)

    elif method == "random":
        # 随机脉冲生成        
        rng = np.random.RandomState(seed)
        pulses = rng.uniform(-1.0, 1.0, size=(n_steps, 2))
        # Amax = 2 * np.pi * 179e6  # 默认 179 MHz，幅度阈值
        # pulses = rng.uniform(-1.0, 1.0, size=(n_steps, 2)) * Amax

    elif method == "rectangular":
        def build_rectangular_pulse(n_steps: int, dt: float, target_angle: float = np.pi, duty_cycle: float = 0.7, use_q: bool = True) -> np.ndarray:
            """
            生成矩形脉冲，基于面积匹配使得 sum(I)*dt ≈ target_angle。
            """
            # 创建矩形包络
            env = np.zeros(n_steps)
            # 计算脉冲起始和结束位置，使脉冲居中
            pulse_length = int(n_steps * duty_cycle)
            start_idx = (n_steps - pulse_length) // 2
            end_idx = start_idx + pulse_length
            # 设置矩形脉冲区域的值为1
            env[start_idx:end_idx] = 1.0
            
            # 面积匹配：sum(I)*dt = target_angle
            area = np.sum(env) * dt
            if area < 1e-18:
                raise ValueError("Pulse area too small. Adjust duty_cycle.")
            amp = target_angle / area  # rad/s
            
            I = amp * env
            
            if use_q:
                # 使用非零的Q路脉冲，这里使用较小的振幅，相位差为π/2
                # 这有助于更好地控制量子系统的相位
                Q = 0.3 * amp * env  # Q路振幅为I路的30%
            else:
                Q = np.zeros_like(I)
            
            return np.column_stack([I, Q])


        # 生成矩形脉冲
        pulses = build_rectangular_pulse(n_steps, dt, target_angle)
    else:
        raise ValueError(f"未知的初始脉冲构建方法: {method}")
    return pulses



def smooth_pulses(pulses: np.ndarray, smooth_len: int = 5) -> np.ndarray:
    """
    对脉冲进行轻度Hann平滑。
    pulses: shape (n_steps,), rad/s
    返回: shape (n_steps,), rad/s
    """
    # 轻度平滑，降低高频，减少P_d
    smooth_len = max(1, int(smooth_len))
    if smooth_len > 1:
        # 使用Hann窗卷积
        win = np.hanning(smooth_len)
        win = win / win.sum() if win.sum() != 0 else win
        pulses = np.convolve(pulses, win, mode="same")
    return pulses


class CNOTPulseOptimizer:
    """
    开放系统CNOT门脉冲优化（SPSA + 直接脉冲优化）
    - 变量：I/Q各n_steps个脉冲值（共2*n_steps维）
    - 目标：最大化评分器overall_score（平均多个seed，包含n_shots的ensemble）
    - 优化：直接在300个脉冲步长上进行优化，而非通过结点参数化
    """

    def __init__(
        self,
        grader: DispersiveCNOTPulseGrader(computing_method = 'parallel'), # 评分器采用并行计算
        n_steps: int = 300,
        dt: float = 5e-10,
        Amax_MHz: float = 200.0,
        smooth_len: int = 5,
        rng_seed: int = 1234,
    ):
        self.grader = grader
        self.n_steps = n_steps
        self.dt = dt
        self.smooth_len = smooth_len
        self.rng = np.random.RandomState(rng_seed)

        # 振幅上界（rad/s）
        self.Amax = 2 * np.pi * Amax_MHz * 1e6

        # 变量维度：2*n_steps（I/Q脉冲）
        self.dim = 2 * n_steps

    def vec_to_pulses(self, x: np.ndarray) -> np.ndarray:
        """
        将优化变量x映射成 pulses
        - x[:n_steps]: I脉冲值（以tanh映射到[-Amax,Amax]）
        - x[n_steps:2*n_steps]: Q脉冲值
        """
        assert x.shape[0] == self.dim
        # 直接使用x的前n_steps和接下来n_steps个元素作为脉冲值
        sI = x[:self.n_steps]
        sQ = x[self.n_steps:2*self.n_steps]

        # 直接将sI和sQ映射到脉冲幅度范围
        I_pulses = self.Amax * np.tanh(sI)
        Q_pulses = self.Amax * np.tanh(sQ)

        # # 对脉冲进行轻度平滑处理
        # I = smooth_pulses(I_pulses, smooth_len=self.smooth_len)
        # Q = smooth_pulses(Q_pulses, smooth_len=self.smooth_len)

        # pulses = np.column_stack([I, Q]).astype(np.float64)

        # 不进行平滑处理
        pulses = np.column_stack([I_pulses, Q_pulses]).astype(np.float64)
        return pulses

    def pulses_to_init_vec(self, pulses_init: np.ndarray) -> np.ndarray:
        """
        将初始脉冲直接映射为优化变量x向量（通过arctanh反变换），用于SPSA初值。
        - pulses_init: 初始脉冲，shape (n_steps, 2)
        """
        # 直接使用脉冲值作为初始优化变量（通过arctanh反变换）
        def safe_atanh(y):
            y = np.clip(y, -0.999, 0.999)
            return 0.5 * np.log((1 + y) / (1 - y))

        xI = safe_atanh(pulses_init[:, 0] / self.Amax)
        xQ = safe_atanh(pulses_init[:, 1] / self.Amax)

        return np.concatenate([xI, xQ]).astype(np.float64)

    def evaluate_score(self, pulses: np.ndarray,
                       seeds: List[int], n_shots: int) -> float:
        """
        对多个seed取平均overall_score，作为鲁棒目标。
        """
        scores = []
        for sd in seeds:
            res = self.grader.grade_submission(
                pulses, n_shots=n_shots, seed=sd, verbose=False
            )
            scores.append(res["overall_score"])
        return float(np.mean(scores))

    def spsa_optimize(
        self,
        x0: np.ndarray,
        max_iter: int = 200,
        a: float = 0.5,
        c: float = 0.08,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        n_shots: int = 10,
        seeds: List[int] = (42, 123),
        x_clip: float = 3.0,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, dict]:
        """
        核心SPSA循环：最大化目标（overall_score）
        - x变量是未约束的实数，但会被clip防止tanh饱和
        - 每步评估2次（x+/-c Δ）
        """
        x = x0.copy()
        print(f"x shape: {x.shape}")
        best_x = x.copy()
        best_score = -1e9
        hist = [] # 存储每一次迭代的信息
        
        for k in range(max_iter):
            # 记录开始时间
            iter_start_time = time.time()
            ak = a / pow(A + k + 1, alpha)
            ck = c / pow(k + 1, gamma)

            # Rademacher分布扰动 {-1,+1}^dim
            delta = self.rng.choice([-1.0, 1.0], size=x.shape)

            x_plus = np.clip(x + ck * delta, -x_clip, x_clip)
            x_minus = np.clip(x - ck * delta, -x_clip, x_clip)

            # 两次评估
            pulses_p = self.vec_to_pulses(x_plus)
            f_plus = self.evaluate_score(pulses_p, seeds=seeds, n_shots=n_shots)

            pulses_m = self.vec_to_pulses(x_minus)
            f_minus = self.evaluate_score(pulses_m, seeds=seeds, n_shots=n_shots)

            # SPSA梯度估计（maximize）
            ghat = (f_plus - f_minus) / (2.0 * ck) * delta

            # 上升更新
            x = x + ak * ghat
            x = np.clip(x, -x_clip, x_clip)

            # 记录与best
            pulses_x = self.vec_to_pulses(x)
            f_x = self.evaluate_score(pulses_x, seeds=seeds, n_shots=n_shots)
            if f_x > best_score:
                best_score = f_x
                best_x = x.copy()
                pulses_best = self.vec_to_pulses(best_x)
                # 保存脉冲
                np.save("cnot_pulses_optimized.npy", pulses_best)

            # 记录迭代时间
            iter_time = time.time() - iter_start_time

            hist.append({
                "iter": k,
                "score": f_x,
                "best": best_score,
                "ak": ak,
                "ck": ck,
                "iter_time": iter_time,
            })
            if verbose:
                # 每次迭代都打印结果和消耗的时间
                print(f"[SPSA] iter={k+1:4d} score={f_x:.6f} best={best_score:.6f} ak={ak:.3e} ck={ck:.3e} iter_time={iter_time:.2f}s")

        return best_x, best_score, hist

    def run(self,
            iters: int = 200,
            shots: int = 10,
            seeds: List[int] = (42, 123),
            init_method: str = "gaussian",
            pulses_init: np.ndarray = None,
            file_name: str = None,
            verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """
        单阶段优化流程
        init_method: 构建初始脉冲，默认使用高斯
        """
        print("计算初始分数...")
        init_time = time.time()


        if pulses_init is None:
            # 外界未传入脉冲，则使用内置的脉冲生成
            file_name = init_method
            if init_method == "gaussian":
                # 高斯形状在量子控制中通常是较好的初始猜测。分数初始就很高
                pulses_init = generate_initial_pulse(self.n_steps, self.dt, method="gaussian", target_angle=np.pi)
            elif init_method == "random":
                # 构建初始脉冲（随机）
                pulses_init = generate_initial_pulse(self.n_steps, self.dt, method="random", seed=1234)
            elif init_method == "rectangular":
                # 构建初始脉冲（矩形）
                pulses_init = generate_initial_pulse(self.n_steps, self.dt, method="rectangular")

            else:
                raise ValueError(f"未知的初始脉冲构建方法: {init_method}")
            
            # 设置优化参数
            a = 0.20
            c = 0.12
        else:
            if file_name is None:
                file_name = "afferent"

            a = 0.01
            c = 0.01        
        

        x0 = self.pulses_to_init_vec(pulses_init)
        init_score = self.evaluate_score(pulses_init, seeds=[42], n_shots=shots)
        init_time = time.time() - init_time
        print(f"初始分数: {init_score:.6f}, 初始消耗时间: {init_time:.2f}s")

        print("开始优化迭代...")
        x_best, final_score, iter_hist = self.spsa_optimize(
            x0=x0, max_iter=iters,
            a=a, c=c, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=shots, seeds=list(seeds), x_clip=3.0, verbose=verbose)
        print(f"优化结束: best_score={final_score:.6f}")

        # 输出当前最优的脉冲
        pulses_best = self.vec_to_pulses(x_best)

        # 将初始分数 init_score 存到 iter_hist 中
        iter_hist.append({
            "初始分数": init_score,
        })
        
        # 保存脉冲
        np.save(f"results/cnot_pulses_{file_name}.npy", pulses_best)
        print(f"已保存脉冲到 results/cnot_pulses_{file_name}.npy")

        # 存储历史记录
        with open(f"results/cnot_history_{file_name}.json", 'w') as f:
            json.dump(iter_hist, f)
        print(f"已保存历史记录到 results/cnot_history_{file_name}.json")


        return pulses_best, iter_hist


# 定义评分函数，方便多进程调用，及结果对比
def evaluate_pulse(args, computing_method='serial'):
    pulse_data, verbose = args
    # 为每个进程创建独立的评分器实例，使用官方评分器
    local_grader = DispersiveCNOTPulseGrader(
    nq_levels=3,
    n_steps=300,
    dt=5e-10,
    n_shots=10,
    computing_method=computing_method
    )
    results = local_grader.grade_submission(pulse_data, n_shots=10, seed=None, verbose=verbose)
    return results['overall_score'], results['gate_error'], results["gate_fidelity"],results['leakage_score'], results['penalty_score']


# 脉冲可视化函数
def plot_pulses(pulses, n_steps, dt=5e-10, title="优化后的脉冲"):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time_ns = np.arange(n_steps + 1) * dt * 1e9
    two_pi = 2 * np.pi

    ax1.step(time_ns, 
             np.append(pulses[:, 0], pulses[-1, 0]) / (two_pi * 1e6),
             where='post', linewidth=2, color='blue', label='Ω_re')
    ax1.set_ylabel('Ω_re / 2π (MHz)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.step(time_ns,
             np.append(pulses[:, 1], pulses[-1, 1]) / (two_pi * 1e6),
             where='post', linewidth=2, color='red', label='Ω_im')
    ax2.set_ylabel('Ω_im / 2π (MHz)', fontsize=12)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle(f'{title}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()


def extract_scores_from_iter_hist(iter_hist):
    """
    从迭代历史记录中提取score值组成列表

    """
    scores = []
    # 将初始值作为第一个元素
    scores.append(iter_hist[-1]["初始分数"])
    for record in iter_hist:
        if "score" in record:
            scores.append(record["score"])
    
    return scores
def plot_iter_hist(iter_hist, title="优化过程中的评分变化"):


    scores_list = extract_scores_from_iter_hist(iter_hist)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(scores_list, linewidth=2, color='blue')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f'{title}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()



if __name__ == "__main__":
    # 初始化官方评分器（双比特CNOT）
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

    optimizer = CNOTPulseOptimizer(
        grader=grader,
        n_steps=300,
        dt=5e-10,
        Amax_MHz=200.0,    # 幅度上限 150 MHz
        smooth_len=3,      # 轻度平滑窗口
        rng_seed=42
    )

    pulses_best, iter_hist = optimizer.run(
        iters=100,
        shots=10,
        seeds=[42],
        init_method="rectangular",
        verbose=True
    )

    grader.grade_submission(pulses_best, n_shots=10, seed=None, verbose=True)
