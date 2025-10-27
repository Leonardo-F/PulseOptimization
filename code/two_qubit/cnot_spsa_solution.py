import os
import time
import numpy as np
from typing import List, Tuple
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))
from two_transmon_grader import DispersiveCNOTPulseGrader


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

# 使用多段高斯叠加
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def multi_gaussian_initial_guess(n_steps, dt):
    t = np.linspace(0, n_steps * dt, n_steps)
    pulse_I = (
        gaussian(t, mu=37.5e-9, sigma=10e-9) +
        gaussian(t, mu=112.5e-9, sigma=15e-9)
    )
    pulse_Q = 0.5 * gaussian(t, mu=75e-9, sigma=20e-9)  # 弱Q驱动
    return np.column_stack([pulse_I, pulse_Q]) * 50e6  # 缩放到合理振幅(rad/s)

def cnot_optimized_initial_guess(n_steps, dt):
    """
    为CNOT门设计的优化初始脉冲，考虑色散耦合区域的物理特性
    - 使用多个高斯峰组合，覆盖脉冲全过程
    - 调整I/Q通道的相位关系，增强CNOT门的纠缠特性
    - 考虑泄漏抑制和振幅平滑过渡
    """
    t = np.linspace(0, n_steps * dt, n_steps)
    total_time = n_steps * dt
    
    # I通道：使用3个高斯峰，分别对应CNOT门的三个关键阶段
    pulse_I = (
        1.2 * gaussian(t, mu=0.25*total_time, sigma=0.08*total_time) +
        1.0 * gaussian(t, mu=0.50*total_time, sigma=0.10*total_time) +
        0.8 * gaussian(t, mu=0.75*total_time, sigma=0.08*total_time)
    )
    
    # Q通道：使用相位延迟的高斯峰，增强相干性
    pulse_Q = (
        0.3 * gaussian(t, mu=0.20*total_time, sigma=0.07*total_time) +
        0.7 * gaussian(t, mu=0.50*total_time, sigma=0.09*total_time) +
        0.4 * gaussian(t, mu=0.80*total_time, sigma=0.07*total_time)
    )
    
    # 添加一个缓慢上升和下降的包络，减少边缘处的导数惩罚
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_steps) / (n_steps - 1)))
    pulse_I *= envelope
    pulse_Q *= envelope
    
    # 缩放到合理振幅，考虑系统参数
    scaling_factor = 70e6  # 稍高于之前的值，以提高初始保真度
    return np.column_stack([pulse_I, pulse_Q]) * scaling_factor  # rad/s


def knots_to_pulses(knots: np.ndarray, n_steps: int, smooth_len: int = 5) -> np.ndarray:
    """
    将K个结点线性插值到n_steps步，并进行轻度Hann平滑。
    knots: shape (K,), 值为rad/s
    返回: shape (n_steps,), rad/s
    """
    K = len(knots)
    x_knots = np.linspace(0, n_steps - 1, K)
    x = np.arange(n_steps)
    arr = np.interp(x, x_knots, knots)

    # 轻度平滑，降低高频，减少P_d
    smooth_len = max(1, int(smooth_len))
    if smooth_len > 1:
        # 使用Hann窗卷积
        win = np.hanning(smooth_len)
        win = win / win.sum() if win.sum() != 0 else win
        arr = np.convolve(arr, win, mode="same")
    return arr


class RobustOpenSystemSPSA:
    """
    开放系统CNOT门鲁棒脉冲优化（SPSA + 带限参数化）
    - 变量：I/Q各K个结点（共2K维）
    - 目标：最大化评分器overall_score（平均多个seed，包含n_shots的ensemble）
    """

    def __init__(
        self,
        grader: DispersiveCNOTPulseGrader,
        n_steps: int = 300,
        dt: float = 5e-10,
        K: int = 20,
        Amax_MHz: float = 200.0,
        smooth_len: int = 5,
        rng_seed: int = 1234,
    ):
        self.grader = grader
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
        self.smooth_len = smooth_len
        self.rng = np.random.RandomState(rng_seed)

        # 振幅上界（rad/s）
        self.Amax = 2 * np.pi * Amax_MHz * 1e6

        # 变量维度：2K（I/Q结点）
        self.dim = 2 * K

    def vec_to_pulses(self, x: np.ndarray) -> np.ndarray:
        """
        将优化变量x映射成 pulses
        - x[:K]: I结点（以tanh映射到[-Amax,Amax]）
        - x[K:2K]: Q结点
        """
        assert x.shape[0] == self.dim
        sI = x[:self.K]
        sQ = x[self.K:2*self.K]

        I_knots = self.Amax * np.tanh(sI)
        Q_knots = self.Amax * np.tanh(sQ)

        I = knots_to_pulses(I_knots, self.n_steps, smooth_len=self.smooth_len)
        Q = knots_to_pulses(Q_knots, self.n_steps, smooth_len=self.smooth_len)

        pulses = np.column_stack([I, Q]).astype(np.float64)
        return pulses

    def pulses_to_init_vec(self, pulses_init: np.ndarray) -> np.ndarray:
        """
        将一个初始脉冲（300步）压缩为K结点的x向量（通过插值逆映射+arctanh），用于SPSA初值。
        """
        # 先提取I/Q在K个结点处的值（在原300步上的线性采样）
        x_knots = np.linspace(0, self.n_steps - 1, self.K)
        I_knots = np.interp(x_knots, np.arange(self.n_steps), pulses_init[:, 0])
        Q_knots = np.interp(x_knots, np.arange(self.n_steps), pulses_init[:, 1])

        # 反映射： knots = Amax * tanh(s) => s = atanh(knots/Amax)
        def safe_atanh(y):
            y = np.clip(y, -0.999, 0.999)
            return 0.5 * np.log((1 + y) / (1 - y))

        xI = safe_atanh(I_knots / self.Amax)
        xQ = safe_atanh(Q_knots / self.Amax)

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
        a: float = 0.15,
        c: float = 0.10,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        n_shots: int = 10,
        seeds: List[int] = (42, 123),
        print_every: int = 10,
        x_clip: float = 3.0,
    ) -> Tuple[np.ndarray, float, dict]:
        """
        核心SPSA循环：最大化目标（overall_score）
        - x变量是未约束的实数，但会被clip防止tanh饱和
        - 每步评估2次（x+/-c Δ）
        """
        x = x0.copy()
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
                np.save(f"cnot_spsa_pulses.npy", pulses_best)

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

            # 每次迭代都打印结果和消耗的时间
            print(f"[SPSA] iter={k+1:4d} score={f_x:.6f} best={best_score:.6f} ak={ak:.3e} ck={ck:.3e} iter_time={iter_time:.2f}s")

        return best_x, best_score, hist

    def run(self,
            phase1_iters: int = 150,
            phase2_iters: int = 80,
            phase1_shots: int = 5,
            phase1_seeds: List[int] = (11, 22),
            phase2_shots: int = 10,
            phase2_seeds: List[int] = (101, 202, 303),
            save_prefix: str = "cnot_open_system") -> Tuple[np.ndarray, dict]:
        """
        两阶段鲁棒优化流程：
        Phase1: 快速粗搜（少shots、少seeds）
        Phase2: 默认shots与多seed做精修
        """
        # 构建初始脉冲，使用专为CNOT门优化的初始猜测
        # pulses_init = build_area_matched_gaussian(self.n_steps, self.dt, target_angle=np.pi)
        # pulses_init = multi_gaussian_initial_guess(self.n_steps, self.dt)
        pulses_init = cnot_optimized_initial_guess(self.n_steps, self.dt)
        x0 = self.pulses_to_init_vec(pulses_init)

        print("Phase 1: 粗搜开始")
        x_best, s_best, hist1 = self.spsa_optimize(
            x0=x0, max_iter=phase1_iters,
            a=0.20, c=0.12, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=phase1_shots, seeds=list(phase1_seeds),
            print_every=10
        )
        print(f"Phase 1结束: best_score={s_best:.6f}")

        # Phase 2: 精修（n_shots=10、多seed）
        print("Phase 2: 精修开始")
        x_best2, s_best2, hist2 = self.spsa_optimize(
            x0=x_best, max_iter=phase2_iters,
            a=0.12, c=0.08, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=phase2_shots, seeds=list(phase2_seeds),
            print_every=10
        )
        pulses_best = self.vec_to_pulses(x_best2)
        final_score = self.evaluate_score(pulses_best, seeds=list(phase2_seeds), n_shots=phase2_shots)
        print(f"Phase 2结束: best_score={final_score:.6f}")

        # 保存脉冲
        np.save(f"{save_prefix}_pulses.npy", pulses_best)
        print(f"已保存脉冲到 {save_prefix}_pulses.npy")

        # 存储历史记录
        with open(f"{save_prefix}_history_phase1.json", 'w') as f:
            json.dump(hist1, f)
        with open(f"{save_prefix}_history_phase2.json", 'w') as f:
            json.dump(hist2, f)
        print(f"已保存历史记录到 {save_prefix}_history_phase1.json 和 {save_prefix}_history_phase2.json")

        # 最终正式评分（比赛默认：n_shots=10、seed可固定一个或取平均）
        final_results = self.grader.grade_submission(pulses_best, n_shots=10, seed=42, verbose=True)
        self.grader.save_results(final_results, f"{save_prefix}_results.json")

        return pulses_best, final_results

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
        A_penalty=0.1
    )

    optimizer = RobustOpenSystemSPSA(
        grader=grader,
        n_steps=300,
        dt=5e-10,
        K=150,              # 20个结点 -> 300步插值
        Amax_MHz=200.0,    # 幅度上限 2π×200 MHz
        smooth_len=5,      # 轻度平滑窗口
        rng_seed=1234
    )

    pulses_best, results = optimizer.run(
        phase1_iters=50, 
        phase2_iters=80,
        phase1_shots=5,
        phase1_seeds=(11, 22),
        phase2_shots=10,
        phase2_seeds=(101, 202, 303),
        save_prefix="cnot_spsa"
    )