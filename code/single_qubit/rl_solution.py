import numpy as np
from typing import List, Tuple


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))
from single_transmon_grader import TransmonPulseGrader

from single_qubit_solution import knots_to_pulses, RobustOpenSystemSPSA
# 复用前文的 knots_to_pulses、RobustOpenSystemSPSA.vec_to_pulses_phi 等函数/思路
# 为简洁，这里内联必要方法（映射与初值生成）

def gaussian_envelope(n_steps: int, sigma_frac: float = 0.22):
    t = np.arange(n_steps)
    center = 0.5*(n_steps-1)
    sigma = sigma_frac * n_steps
    return np.exp(-0.5*((t-center)/sigma)**2)

def build_init_pulses(n_steps: int, dt: float) -> np.ndarray:
    env = gaussian_envelope(n_steps)
    area = env.sum()*dt
    amp = (np.pi/2)/area
    I = amp*env
    Q = np.zeros_like(I)
    return np.column_stack([I,Q]).astype(np.float64)

def knots_to_pulses(knots: np.ndarray, n_steps: int, smooth_len: int=5) -> np.ndarray:
    K = len(knots)
    x_knots = np.linspace(0, n_steps-1, K)
    x = np.arange(n_steps)
    arr = np.interp(x, x_knots, knots)
    if smooth_len>1:
        win = np.hanning(smooth_len)
        win = win/win.sum() if win.sum()!=0 else win
        arr = np.convolve(arr, win, mode="same")
    return arr

class REINFORCEOpenSystem:
    def __init__(self, grader: TransmonPulseGrader,
                 n_steps=30, dt=5e-10, K=10, Amax_MHz=150.0, smooth_len=5, seed=2025):
        self.grader = grader
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
        self.smooth_len = smooth_len
        self.Amax = 2*np.pi*Amax_MHz*1e6
        self.dim = 2*K+1
        self.rng = np.random.RandomState(seed)

    def map_vec(self, z: np.ndarray) -> Tuple[np.ndarray, float]:
        # z为策略输出的实数向量（均值+噪声），映射到(pulses, phi)
        sI = z[:self.K]; sQ = z[self.K:2*self.K]; sphi = z[-1]
        I_knots = self.Amax*np.tanh(sI)
        Q_knots = self.Amax*np.tanh(sQ)
        phi = np.pi*np.tanh(sphi)
        I = knots_to_pulses(I_knots, self.n_steps, self.smooth_len)
        Q = knots_to_pulses(Q_knots, self.n_steps, self.smooth_len)
        return np.column_stack([I,Q]).astype(np.float64), float(phi)

    def evaluate(self, pulses, phi, seeds: List[int], n_shots: int) -> float:
        vals = []
        for sd in seeds:
            res = self.grader.grade_submission(pulses, phi, n_shots=n_shots, seed=sd, verbose=False)
            vals.append(res["overall_score"])
        return float(np.mean(vals))

    def init_mean(self) -> np.ndarray:
        pulses0 = build_init_pulses(self.n_steps, self.dt)
        # 压缩为K结点并做atanh反映射
        x_knots = np.linspace(0, self.n_steps-1, self.K)
        Ik = np.interp(x_knots, np.arange(self.n_steps), pulses0[:,0])
        Qk = np.interp(x_knots, np.arange(self.n_steps), pulses0[:,1])
        def safe_atanh(y):
            y = np.clip(y, -0.999, 0.999)
            return 0.5*np.log((1+y)/(1-y))
        muI = safe_atanh(Ik/self.Amax)
        muQ = safe_atanh(Qk/self.Amax)
        mu_phi = 0.0
        return np.concatenate([muI, muQ, [mu_phi]]).astype(np.float64)

    def train(self, max_iter=300, batch_size=6, lr_mu=0.08, lr_logstd=0.02,
              n_shots=10, seeds=(11,22), x_clip=3.0):
        mu = self.init_mean()
        log_std = np.full(self.dim, -1.5, dtype=np.float64)  # 初始较小探索
        best_score = -1e9
        best_z = mu.copy()

        baseline = 0.0
        b_momentum = 0.9

        for it in range(max_iter):
            std = np.exp(log_std)
            zs = []
            scores = []

            # 采样batch
            for _ in range(batch_size):
                eps = self.rng.randn(self.dim)
                z = np.clip(mu + std*eps, -x_clip, x_clip)
                pulses, phi = self.map_vec(z)
                R = self.evaluate(pulses, phi, seeds=list(seeds), n_shots=n_shots)
                zs.append(z); scores.append(R)
                if R>best_score:
                    best_score = R
                    best_z = z.copy()

            zs = np.array(zs)
            scores = np.array(scores)
            # baseline更新（指数平均）
            baseline = b_momentum*baseline + (1-b_momentum)*float(scores.mean())
            adv = scores - baseline

            # REINFORCE梯度（对mu和log_std）
            # ∇_mu logN = (z - mu)/std^2, ∇_logstd logN = ((z-mu)^2/std^2 - 1)
            g_mu = np.zeros_like(mu)
            g_logstd = np.zeros_like(log_std)
            for i in range(batch_size):
                diff = zs[i] - mu
                inv_var = 1.0/(std**2 + 1e-12)
                g_mu += adv[i] * (diff * inv_var)
                g_logstd += adv[i] * ((diff**2)*inv_var - 1.0)

            g_mu /= batch_size
            g_logstd /= batch_size

            # 梯度上升
            mu = np.clip(mu + lr_mu * g_mu, -x_clip, x_clip)
            log_std = np.clip(log_std + lr_logstd * g_logstd, -4.0, 1.5)  # 限制探索尺度

            if (it+1) % 10 == 0:
                print(f"[RL] iter={it+1:4d} score_mean={scores.mean():.6f} best={best_score:.6f} std_mean={std.mean():.3f}")

        # 返回最优结果
        pulses_best, phi_best = self.map_vec(best_z)
        return pulses_best, phi_best, best_score


if __name__ == "__main__":
    grader = TransmonPulseGrader(
        n_levels=4, n_steps=30,
        alpha=-2*np.pi*0.2e9,
        omega_q=2*np.pi*5.0e9,
        omega_d=2*np.pi*5.0e9,
        dt=5e-10, T1=50e-6, T_phi=30e-6, n_bar=0.05,
        sigma_freq=0.5e6, n_shots=15,
        h_a=179e6, h_d=22.4e6, A_penalty=0.1
    )
    agent = REINFORCEOpenSystem(grader, K=10, Amax_MHz=150.0, smooth_len=5)
    pulses, phi, best = agent.train(max_iter=400, batch_size=6, lr_mu=0.08, lr_logstd=0.02,
                                    n_shots=7, seeds=(11,22))
    np.save("pulses_rl.npy", pulses)
    res = grader.grade_submission(pulses, phi, n_shots=15, seed=42, verbose=True)
    grader.save_results(res, "sqrtx_open_system_rl_results.json")