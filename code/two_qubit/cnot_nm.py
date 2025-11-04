import numpy as np
import qutip as qt
from scipy.optimize import minimize
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))

# 可采用多进程并行计算，对原始评分器进行了优化
from two_transmon_grader import DispersiveCNOTPulseGrader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")


class OpenSystemCNOTOptimizer:
    """
    开放系统CNOT门优化
    损失函数由1 - overall_score组成
    """
    def __init__(self, grader):
        self.grader = grader
        self.iteration = 0
        self.best_score = 0.0
        
    def cost_function(self, pulses_flat):
        """
        成本函数：1 - overall_score
        """
        time_1 = time.time()
        pulses = pulses_flat.reshape((self.grader.n_steps, 2))
        
        # 使用评分器计算分数
        results = self.grader.grade_submission(pulses, seed=42, verbose=False)
        
        # 总分越高越好，所以cost = 1 - score
        score = results['overall_score']
        cost = 1.0 - score
        gate_fidelity = results['gate_fidelity']
        leakage = results['leakage']

        time_1 = time.time() - time_1

        if score > self.best_score:
            self.best_score = score
            best_pulses = pulses
            # 保存脉冲
            np.save("pulses_nm.npy", best_pulses)

        print(f"[Nelder-Mead] iter={self.iteration:4d} score={score:.6f} best={self.best_score:.6f} Cost: {cost:.6f} Gate fidelity: {gate_fidelity:.6f} Leakage: {leakage:.6f} iter_time={time_1:.2f}s")
        

        self.iteration += 1
        return cost
    
    def optimize(self, initial_pulses, maxiter=50):
        """
        优化脉冲
        
        参数:
        initial_pulses: 初始脉冲（来自封闭系统优化结果）
        maxiter: 最大迭代次数
        """
        x0 = initial_pulses.flatten()
        
        print("\n" + "="*70)
        print("开始开放系统优化...")
        print(f"使用Nelder-Mead方法进行优化")
        print(f"最大迭代次数: {maxiter}")
        print("="*70 + "\n")
        
        # 无梯度方法优化
        result = minimize(
            self.cost_function,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )


        result.pulses = result.x.reshape((self.grader.n_steps, 2))
        
        # 最终评估
        final_results = self.grader.grade_submission(result.pulses, seed=42, verbose=True)
        result.final_results = final_results
        
        return result

# ============================================================================
# 主程序：开放系统优化
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 初始化评分器
    grader = DispersiveCNOTPulseGrader(
        nq_levels=3,
        n_steps=300,
        dt=5e-10,
        n_shots=10,
        computing_method='parallel'
    )
    
    # 加载封闭系统优化结果
    pulses_closed = np.load('/Users/fangaoming/Desktop/GitHub/PulseOptimization/code/two_qubit/results/pulses_closed.npy')
    print("已加载封闭系统优化脉冲")
    print(f"脉冲形状: {pulses_closed.shape}")
    
    # 评估封闭系统脉冲在开放系统中的性能
    print("\n" + "="*70)
    print("评估封闭系统脉冲在开放系统中的性能...")
    print("="*70)
    results_closed = grader.grade_submission(pulses_closed, seed=42, verbose=False)
    print(f"封闭系统性能:")
    print(f"  Overall score: {results_closed['overall_score']:.6f}")

    # 初始化开放系统优化器
    optimizer = OpenSystemCNOTOptimizer(grader)
    
    # 运行优化
    result_open = optimizer.optimize(
        pulses_closed,
        maxiter=500  # 根据计算资源调整
    )
    
    # 保存最终脉冲
    np.save('cnot_pulses.npy', result_open.pulses)
    grader.save_results(result_open.final_results, 'cnot_results.json')
    
    print("\n" + "="*70)
    print("优化完成!")
    print("="*70)
    print(f"\n开放系统优化后:")
    print(f"  Overall score: {result_open.final_results['overall_score']:.6f}")
    print(f"\n提升: {(result_open.final_results['overall_score'] - results_closed['overall_score']):.6f}")
    
    print(f"\n最终脉冲已保存到: cnot_pulses.npy")
    print(f"评分结果已保存到: cnot_results.json")