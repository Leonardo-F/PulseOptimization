<div align=center>
<h1>Pulse Optimization</h1>
</div>

本项目要求解的问题是 2025 量旋杯赛题 [量子门脉冲优化](docs/task.md)，具体来说是要优化单、双量子门脉冲，使得量子电路的执行结果与目标结果尽可能接近。

## 环境准备

本项目基于 `python = 3.11.13` 构建，所需依赖包见 `requirements.txt` 


## 文件说明

我们方案的详细介绍，包含单比特和双比特门，参见 [2025 SpinQ Cup Pulse Optimization](docs/2025_SpinQ_PlusesOptimization.pdf)。

需要注意，我们在项目中的 `.npy` 文件，若命名带有 `_best` 后缀，表示该文件为最佳结果脉冲。同时为了与 [task](docs/task.md) 中的要求一致，我们在各文件夹下，拷贝了最优脉冲文件，均命名为 `pulses.npy` 。

### 项目结构

```
PulseOptimization/
├── code/                           # 代码目录
│   ├── official/                   # 官方示例代码
│   │   └── closed_system_example.ipynb
│   ├── single_qubit/               # 单比特门优化
│   │   ├── solution.ipynb          # 单比特√X门优化解决方案
│   │   ├── compare.ipynb           # 性能对比分析
│   │   ├── single_transmon_grader.py       # 单比特评分器（优化版）
│   │   ├── single_transmon_grader_origin.py # 单比特评分器（原始版）
│   │   ├── spsa_utils.py           # SPSA优化工具函数
│   │   └── results/                # 优化结果文件
│   └── two_qubit/                  # 双比特门优化
│       ├── cnot_solution.ipynb     # 双比特CNOT门优化解决方案
│       ├── cnot_compare.ipynb      # 性能对比分析
│       ├── cnot_closed.py          # 闭合系统GRAPE优化
│       ├── cnot_spsa_utils.py      # SPSA优化工具函数
│       ├── two_transmon_grader.py          # 双比特评分器（优化版）
│       ├── two_transmon_grader_origin.py   # 双比特评分器（原始版）
│       └── results/                # 优化结果文件
├── docs/                           # 文档目录
│   ├── 2025_SpinQ_Cup_PlusesOptimization.pdf  # report 文档
│   ├── task.md                     # 赛题详细说明
│   └── note.md                     # 技术笔记
└── requirements.txt                # Python依赖包列表
```

### 单比特门优化 (single_qubit/)

**主要文件：**
- `solution.ipynb` - 完整的单比特√X门优化实现，使用SPSA算法结合参数化脉冲设计
- `spsa_utils.py` - 包含脉冲生成、优化器、评估工具等核心函数
- `single_transmon_grader.py` - 优化版评分器，支持并行计算加速

### 双比特门优化 (two_qubit/)

**主要文件：**
- `cnot_solution.ipynb` - 双比特CNOT门优化实现，采用两阶段优化策略
- `cnot_closed.py` - 闭合系统GRAPE优化，为开放系统优化提供良好起点
- `cnot_spsa_utils.py` - 双比特优化的工具函数
- `two_transmon_grader.py` - 优化版评分器，支持并行计算
