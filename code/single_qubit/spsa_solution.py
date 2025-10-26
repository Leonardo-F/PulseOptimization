import os
import time
import numpy as np
from typing import List, Tuple
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))
from single_transmon_grader import TransmonPulseGrader


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


def build_area_matched_gaussian(n_steps: int, dt: float, target_angle: float = np.pi/2) -> np.ndarray:
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
    开放系统√X门鲁棒脉冲优化（SPSA + 带限参数化）
    - 变量：I/Q各K个结点 + 相位φ（共2K+1维）
    - 目标：最大化评分器overall_score（平均多个seed，包含n_shots的ensemble）
    """

    def __init__(
        self,
        grader: TransmonPulseGrader,
        n_steps: int = 30,
        dt: float = 5e-10,
        K: int = 10,
        Amax_MHz: float = 150.0,
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

        # 变量维度：2K（I/Q结点） + 1（φ）
        self.dim = 2 * K + 1

    def vec_to_pulses_phi(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        将优化变量x映射成 (pulses, phi)
        - x[:K]: I结点（以tanh映射到[-Amax,Amax]）
        - x[K:2K]: Q结点
        - x[-1]: φ变量，tanh映射到[-π, π]
        """
        assert x.shape[0] == self.dim
        sI = x[:self.K]
        sQ = x[self.K:2*self.K]
        sphi = x[-1]

        I_knots = self.Amax * np.tanh(sI)
        Q_knots = self.Amax * np.tanh(sQ)
        phi = np.pi * np.tanh(sphi)

        I = knots_to_pulses(I_knots, self.n_steps, smooth_len=self.smooth_len)
        Q = knots_to_pulses(Q_knots, self.n_steps, smooth_len=self.smooth_len)

        pulses = np.column_stack([I, Q]).astype(np.float64)
        return pulses, float(phi)

    def pulses_to_init_vec(self, pulses_init: np.ndarray, phi_init: float = 0.0) -> np.ndarray:
        """
        将一个初始脉冲（30步）压缩为K结点的x向量（通过插值逆映射+arctanh），用于SPSA初值。
        """
        # 先提取I/Q在K个结点处的值（在原30步上的线性采样）
        x_knots = np.linspace(0, self.n_steps - 1, self.K)
        I_knots = np.interp(x_knots, np.arange(self.n_steps), pulses_init[:, 0])
        Q_knots = np.interp(x_knots, np.arange(self.n_steps), pulses_init[:, 1])

        # 反映射： knots = Amax * tanh(s) => s = atanh(knots/Amax)
        def safe_atanh(y):
            y = np.clip(y, -0.999, 0.999)
            return 0.5 * np.log((1 + y) / (1 - y))

        xI = safe_atanh(I_knots / self.Amax)
        xQ = safe_atanh(Q_knots / self.Amax)
        xphi = safe_atanh(phi_init / np.pi)

        return np.concatenate([xI, xQ, [xphi]]).astype(np.float64)

    def evaluate_score(self, pulses: np.ndarray, phi: float,
                       seeds: List[int], n_shots: int) -> float:
        """
        对多个seed取平均overall_score，作为鲁棒目标。
        """
        scores = []
        for sd in seeds:
            res = self.grader.grade_submission(
                pulses, phi, n_shots=n_shots, seed=sd, verbose=False
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
        n_shots: int = 15,
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
            pulses_p, phi_p = self.vec_to_pulses_phi(x_plus)
            f_plus = self.evaluate_score(pulses_p, phi_p, seeds=seeds, n_shots=n_shots)

            pulses_m, phi_m = self.vec_to_pulses_phi(x_minus)
            f_minus = self.evaluate_score(pulses_m, phi_m, seeds=seeds, n_shots=n_shots)

            # SPSA梯度估计（maximize）
            ghat = (f_plus - f_minus) / (2.0 * ck) * delta

            # 上升更新
            x = x + ak * ghat
            x = np.clip(x, -x_clip, x_clip)

            # 记录与best
            pulses_x, phi_x = self.vec_to_pulses_phi(x)
            f_x = self.evaluate_score(pulses_x, phi_x, seeds=seeds, n_shots=n_shots)
            if f_x > best_score:
                best_score = f_x
                best_x = x.copy()

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
            phase1_shots: int = 7,
            phase1_seeds: List[int] = (11, 22),
            phase2_shots: int = 15,
            phase2_seeds: List[int] = (101, 202, 303),
            save_prefix: str = "sqrtx_open_system") -> Tuple[np.ndarray, float, dict]:
        """
        两阶段鲁棒优化流程：
        Phase1: 快速粗搜（少shots、少seeds）
        Phase2: 默认shots与多seed做精修
        """
        # 构建初始脉冲（高斯面积匹配），高斯形状在量子控制中通常是较好的初始猜测。分数刚开始积极很高
        pulses_init = build_area_matched_gaussian(self.n_steps, self.dt, target_angle=np.pi/2)
        x0 = self.pulses_to_init_vec(pulses_init, phi_init=0.0)

        print("Phase 1: 粗搜开始")
        x_best, s_best, hist1 = self.spsa_optimize(
            x0=x0, max_iter=phase1_iters,
            a=0.20, c=0.12, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=phase1_shots, seeds=list(phase1_seeds),
            print_every=10
        )
        # pulses_p1, phi_p1 = self.vec_to_pulses_phi(x_best)
        print(f"Phase 1结束: best_score={s_best:.6f}")

        # Phase 2: 精修（n_shots=15、多seed）
        print("Phase 2: 精修开始")
        x_best2, s_best2, hist2 = self.spsa_optimize(
            x0=x_best, max_iter=phase2_iters,
            a=0.12, c=0.08, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=phase2_shots, seeds=list(phase2_seeds),
            print_every=10
        )
        pulses_best, phi_best = self.vec_to_pulses_phi(x_best2)
        final_score = self.evaluate_score(pulses_best, phi_best, seeds=list(phase2_seeds), n_shots=phase2_shots)
        print(f"Phase 2结束: best_score={final_score:.6f}")

        # 保存脉冲
        np.save("pulses_spsa.npy", pulses_best)
        print("已保存脉冲到 pulses_spsa.npy")

        # 存储历史记录
        with open(f"{save_prefix}_history_phase1.json", 'w') as f:
            json.dump(hist1, f)
        with open(f"{save_prefix}_history_phase2.json", 'w') as f:
            json.dump(hist2, f)
        print(f"已保存历史记录到 {save_prefix}_history_phase1.json 和 {save_prefix}_history_phase2.json")


        # 最终正式评分（比赛默认：n_shots=15、seed可固定一个或取平均）
        final_results = self.grader.grade_submission(pulses_best, phi_best, n_shots=15, seed=42, verbose=True)
        self.grader.save_results(final_results, f"{save_prefix}_results.json")

        return pulses_best, phi_best

if __name__ == "__main__":
    # 初始化官方评分器（单比特√X）
    grader = TransmonPulseGrader(
        n_levels=4,
        n_steps=30,
        alpha=-2 * np.pi * 0.2e9,
        omega_q=2 * np.pi * 5.0e9,
        omega_d=2 * np.pi * 5.0e9,
        dt=5e-10,          # 0.5 ns
        T1=50e-6,
        T_phi=30e-6,
        n_bar=0.05,
        sigma_freq=0.5e6,  # 0.5 MHz
        n_shots=15,        # 默认评分shots
        h_a=179e6,
        h_d=22.4e6,
        A_penalty=0.1
    )

    optimizer = RobustOpenSystemSPSA(
        grader=grader,
        n_steps=30,
        dt=5e-10,
        K=10,               # 10个结点 -> 30步插值
        # Amax_MHz=150.0,     # 幅度上限 2π×150 MHz，2π 是切换成角频率
        Amax_MHz=179.0,     # 幅度上限 2π×150 MHz
        smooth_len=5,       # 轻度平滑窗口
        rng_seed=1234
    )

    pulses_best, phi_best = optimizer.run(
        phase1_iters=50,         # 可按算力调节（越大一般越好）
        phase2_iters=80,
        phase1_shots=7,
        phase1_seeds=(11, 22),
        phase2_shots=15,
        phase2_seeds=(101, 202, 303),
        save_prefix="spsa"
    )


