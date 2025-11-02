import numpy as np
from typing import List, Tuple
import sys
import os
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))
from two_transmon_grader import DispersiveCNOTPulseGrader


def gaussian_envelope(n_steps: int, sigma_frac: float = 0.22):
    """生成高斯包络"""
    t = np.arange(n_steps)
    center = 0.5 * (n_steps - 1)
    sigma = sigma_frac * n_steps
    return np.exp(-0.5 * ((t - center) / sigma)** 2)


def build_init_pulses(n_steps: int, dt: float) -> np.ndarray:
    """构建初始脉冲"""
    env = gaussian_envelope(n_steps)
    # 为CNOT门设置合适的初始幅度
    amp = 2 * np.pi * 50e6  # 初始设置为50 MHz
    re = amp * env
    im = np.zeros_like(re)
    return np.column_stack([re, im]).astype(np.float64)


def knots_to_pulses(knots: np.ndarray, n_steps: int, smooth_len: int = 5) -> np.ndarray:
    """将控制点插值为完整脉冲序列"""
    K = len(knots)
    x_knots = np.linspace(0, n_steps - 1, K)
    x = np.arange(n_steps)
    arr = np.interp(x, x_knots, knots)
    if smooth_len > 1:
        win = np.hanning(smooth_len)
        win = win / win.sum() if win.sum() != 0 else win
        arr = np.convolve(arr, win, mode="same")
    return arr


class CNOTREINFORCEOptimizer:
    """用于CNOT门优化的REINFORCE算法实现"""
    
    def __init__(self, grader: DispersiveCNOTPulseGrader, 
                 n_steps=300, dt=5e-10, K=20, Amax_MHz=150.0, smooth_len=5, seed=2025):
        """
        初始化强化学习优化器
        
        参数:
        - grader: 评分器实例
        - n_steps: 时间步数
        - dt: 时间步长
        - K: 每通道的控制点数
        - Amax_MHz: 最大幅度（MHz）
        - smooth_len: 平滑窗口长度
        - seed: 随机种子
        """
        self.grader = grader
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
        self.smooth_len = smooth_len
        self.Amax = 2 * np.pi * Amax_MHz * 1e6  # 转换为rad/s
        self.dim = 2 * K  # 两通道，每通道K个控制点
        self.rng = np.random.RandomState(seed)
    
    def map_vec(self, z: np.ndarray) -> np.ndarray:
        """将策略输出向量映射为脉冲序列"""
        # z分为实部和虚部两部分
        s_re = z[:self.K]
        s_im = z[self.K:2*self.K]
        
        # 使用tanh激活限制在[-Amax, Amax]范围内
        re_knots = self.Amax * np.tanh(s_re)
        im_knots = self.Amax * np.tanh(s_im)
        
        # 插值为完整脉冲序列
        re = knots_to_pulses(re_knots, self.n_steps, self.smooth_len)
        im = knots_to_pulses(im_knots, self.n_steps, self.smooth_len)
        
        return np.column_stack([re, im]).astype(np.float64)
    
    def evaluate(self, pulses: np.ndarray, seeds: List[int], n_shots: int) -> float:
        """评估脉冲性能"""
        vals = []
        for sd in seeds:
            res = self.grader.grade_submission(pulses, n_shots=n_shots, seed=sd, verbose=False)
            vals.append(res["overall_score"])
        return float(np.mean(vals))
    
    def init_mean(self) -> np.ndarray:
        """初始化策略参数的均值"""
        # 生成初始脉冲
        pulses0 = build_init_pulses(self.n_steps, self.dt)
        
        # 压缩为K个控制点并做atanh反映射
        x_knots = np.linspace(0, self.n_steps - 1, self.K)
        rek = np.interp(x_knots, np.arange(self.n_steps), pulses0[:, 0])
        imk = np.interp(x_knots, np.arange(self.n_steps), pulses0[:, 1])
        
        # 安全的atanh函数
        def safe_atanh(y):
            y = np.clip(y, -0.999, 0.999)
            return 0.5 * np.log((1 + y) / (1 - y))
        
        # 反映射到原始空间
        mu_re = safe_atanh(rek / self.Amax)
        mu_im = safe_atanh(imk / self.Amax)
        
        return np.concatenate([mu_re, mu_im]).astype(np.float64)
    
    def train(self, max_iter=300, batch_size=8, lr_mu=0.05, lr_logstd=0.01, 
              n_shots=7, seeds=(11, 22, 33), x_clip=3.0):
        """
        训练强化学习模型
        
        参数:
        - max_iter: 最大迭代次数
        - batch_size: 批次大小
        - lr_mu: 均值参数学习率
        - lr_logstd: 标准差参数学习率
        - n_shots: 每次评估的shots数
        - seeds: 随机种子列表
        - x_clip: 参数裁剪范围
        
        返回:
        - 最优脉冲序列
        - 最佳得分
        """
        # 初始化策略参数
        mu = self.init_mean()
        log_std = np.full(self.dim, -1.5, dtype=np.float64)  # 初始较小的探索方差
        best_score = -1e9
        best_z = mu.copy()
        
        # 评估初始脉冲得分
        initial_pulses = self.map_vec(mu)
        initial_score = self.evaluate(initial_pulses, seeds=list(seeds), n_shots=n_shots)
        print(f"初始脉冲得分: {initial_score:.6f}")
        
        # 如果初始脉冲是目前最好的，更新最佳得分
        if initial_score > best_score:
            best_score = initial_score
            best_z = mu.copy()
        
        # 基线初始化（用于减少方差）
        baseline = 0.0
        b_momentum = 0.9
        
        # 学习率调度（可选）
        lr_scheduler_mu = lambda it: lr_mu * np.exp(-it / 100)
        lr_scheduler_logstd = lambda it: lr_logstd * np.exp(-it / 100)
        
        print(f"开始训练REINFORCE优化器...")
        print(f"参数: K={self.K}, batch_size={batch_size}, initial_lr_mu={lr_mu}")
        
        for it in range(max_iter):
            # 记录迭代开始时间
            start_time = time.time()
            
            # 获取当前学习率
            current_lr_mu = lr_scheduler_mu(it)
            current_lr_logstd = lr_scheduler_logstd(it)
            
            # 计算当前标准差
            std = np.exp(log_std)
            
            zs = []
            scores = []
            
            # 采样批次
            for _ in range(batch_size):
                # 采样噪声并生成参数
                eps = self.rng.randn(self.dim)
                z = np.clip(mu + std * eps, -x_clip, x_clip)
                
                # 映射到脉冲并评估
                pulses = self.map_vec(z)
                R = self.evaluate(pulses, seeds=list(seeds), n_shots=n_shots)
                
                zs.append(z)
                scores.append(R)
                
                # 更新最佳结果
                if R > best_score:
                    best_score = R
                    best_z = z.copy()
                    print(f"[新记录] 迭代 {it+1}, 得分: {best_score:.6f}")

                    # 存储当前脉冲
                    np.save("cnot_pulses_rl.npy", pulses)
            
            # 转换为数组便于计算
            zs = np.array(zs)
            scores = np.array(scores)
            
            # 更新基线（指数加权平均）
            baseline = b_momentum * baseline + (1 - b_momentum) * float(scores.mean())
            
            # 计算优势函数
            adv = scores - baseline
            
            # 计算REINFORCE梯度
            g_mu = np.zeros_like(mu)
            g_logstd = np.zeros_like(log_std)
            
            for i in range(batch_size):
                diff = zs[i] - mu
                inv_var = 1.0 / (std** 2 + 1e-12)  # 避免除零
                
                # 均值梯度
                g_mu += adv[i] * (diff * inv_var)
                # 标准差梯度
                g_logstd += adv[i] * ((diff** 2) * inv_var - 1.0)
            
            # 平均梯度
            g_mu /= batch_size
            g_logstd /= batch_size
            
            # 梯度上升更新参数
            mu = np.clip(mu + current_lr_mu * g_mu, -x_clip, x_clip)
            log_std = np.clip(log_std + current_lr_logstd * g_logstd, -4.0, 1.5)  # 限制标准差范围
            
            # 计算迭代时间
            iter_time = time.time() - start_time
            
            # 定期输出进度
            if (it + 1) % 10 == 0:
                print(f"[RL] 迭代={it+1:4d} 得分均值={scores.mean():.6f} 最佳={best_score:.6f} 标准差均值={std.mean():.3f} 学习率={current_lr_mu:.6f} 时间={iter_time:.2f}s")
            else:
                # 对于非10的倍数迭代，也可以输出时间信息
                print(f"[RL] 迭代={it+1:4d} 得分={scores.mean():.6f} 时间={iter_time:.2f}s")
        
        # 返回最优脉冲
        pulses_best = self.map_vec(best_z)
        return pulses_best, best_score


def main():
    """主函数，设置评分器并运行优化"""
    print("初始化两比特评分器...")
    
    # 初始化评分器，使用并行计算加速
    grader = DispersiveCNOTPulseGrader(
        nq_levels=3,
        n_steps=300,
        dt=5e-10,
        omega1_GHz=4.380,
        omega2_GHz=4.614,
        omega_d_GHz=4.498,
        alpha1_GHz=0.210,
        alpha2_GHz=0.215,
        J_GHz=-0.003,
        lambda_coupling=1.03,
        T1_q1=50e-6,
        T1_q2=50e-6,
        Tphi_q1=30e-6,
        Tphi_q2=30e-6,
        nbar_q1=0.0,
        nbar_q2=0.0,
        n_shots=10,
        sigma_detune_q1_Hz=0.5e6,
        sigma_detune_q2_Hz=0.5e6,
        A_penalty=0.1,
        h_a_Hz=200e6,
        h_d_Hz=2.7e6,
        computing_method='parallel'  # 使用并行计算加速
    )
    
    # 创建优化器
    print("创建强化学习优化器...")
    agent = CNOTREINFORCEOptimizer(
        grader,
        K=20,           # 控制点数量
        Amax_MHz=120.0, # 最大幅度限制
        smooth_len=7    # 平滑窗口长度
    )
    
    # 开始训练
    print("开始脉冲优化训练...")
    pulses, best_score = agent.train(
        max_iter=350,
        batch_size=8,
        lr_mu=0.1,
        lr_logstd=0.015,
        n_shots=7,
        seeds=(11, 22, 33, 44),
        x_clip=3.0
    )
    
    # 保存结果
    np.save("pulses_rl.npy", pulses)
    print(f"优化完成！最佳得分: {best_score:.6f}")
    print(f"脉冲已保存到 pulses_rl.npy")
    
    # 最终评估
    print("\n进行最终评估 (n_shots=10)...")
    final_results = grader.grade_submission(pulses, n_shots=10, seed=42, verbose=True)
    grader.save_results(final_results, "cnot_open_system_rl_results.json")
    print("结果已保存到 cnot_open_system_rl_results.json")


if __name__ == "__main__":
    main()