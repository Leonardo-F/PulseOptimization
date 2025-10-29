# cnot_pulse_optimization_simple.py
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


class CNOTPulseOptimizer:
    """
    开放系统CNOT门脉冲优化（SPSA + 带限参数化）
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

            # 每次迭代都打印结果和消耗的时间
            print(f"[SPSA] iter={k+1:4d} score={f_x:.6f} best={best_score:.6f} ak={ak:.3e} ck={ck:.3e} iter_time={iter_time:.2f}s")

        return best_x, best_score, hist

    def run(self,
            iters: int = 200,
            shots: int = 10,
            seeds: List[int] = (42, 123),
            init_method: str = "gaussian") -> Tuple[np.ndarray, dict]:
        """
        单阶段优化流程
        init_method: 构建初始脉冲，默认使用高斯
        """
        print("计算初始分数...")
        init_time = time.time()
        
        if init_method == "gaussian":
            # 构建初始脉冲（高斯面积匹配）
            pulses_init = build_area_matched_gaussian(self.n_steps, self.dt, target_angle=np.pi)
            a = 0.15
            c = 0.10
        else:
            # 构建初始脉冲（随机）
            pulses_init = self.rng.uniform(-1.0, 1.0, size=(self.n_steps, 2)) * self.Amax
            a = 0.20
            c = 0.12

        x0 = self.pulses_to_init_vec(pulses_init)
        init_score = self.evaluate_score(pulses_init, seeds=[42], n_shots=shots)
        init_time = time.time() - init_time
        print(f"初始分数: {init_score:.6f}, 初始消耗时间: {init_time:.2f}s")

        print("开始优化迭代...")
        x_best, final_score, iter_hist = self.spsa_optimize(
            x0=x0, max_iter=iters,
            a=a, c=c, alpha=0.602, gamma=0.101, A=10.0,
            n_shots=shots, seeds=list(seeds))
        print(f"优化结束: best_score={final_score:.6f}")

        # 输出当前最优的脉冲
        pulses_best = self.vec_to_pulses(x_best)

        # 将初始分数 init_score 存到 iter_hist 中
        iter_hist.append({
            "初始分数": init_score,
        })
        
        # 保存脉冲
        np.save(f"cnot_pulses_{init_method}.npy", pulses_best)
        print(f"已保存脉冲到 cnot_pulses_{init_method}.npy")

        # 存储历史记录
        with open(f"cnot_history_{init_method}.json", 'w') as f:
            json.dump(iter_hist, f)
        print(f"已保存历史记录到 cnot_history_{init_method}.json")

        # 最终正式评分
        final_results = self.grader.grade_submission(pulses_best, n_shots=10, seed=42, verbose=True)
        self.grader.save_results(final_results, f"cnot_results_{init_method}.json")

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

    optimizer = CNOTPulseOptimizer(
        grader=grader,
        n_steps=300,
        dt=5e-10,
        K=50,              # 20个结点 -> 300步插值
        Amax_MHz=200.0,    # 幅度上限 2π×200 MHz
        smooth_len=5,      # 轻度平滑窗口
        rng_seed=1234
    )

    pulses_best, results = optimizer.run(
        iters=200,
        shots=10,
        seeds=[42, 123],
        init_method="gaussian"
    )