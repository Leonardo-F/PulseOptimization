"""
定义一些通用的函数

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official'))
from single_transmon_grader import TransmonPulseGrader

# 定义评分函数，方便多进程调用，及结果对比
def evaluate_pulse(args):
    pulse_data, phi, verbose = args
    # 为每个进程创建独立的评分器实例
    local_grader = TransmonPulseGrader(
        n_levels=4,
        n_steps=30,
        dt=5e-10,
        n_shots=15
    )
    results = local_grader.grade_submission(pulse_data, phi, verbose=verbose)
    return results['overall_score'], results['gate_error'], results["gate_fidelity"],results['leakage_score'], results['penalty_score']
