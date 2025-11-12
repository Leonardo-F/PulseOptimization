<div align=center>
<h1>Pulse Optimization</h1>
</div>

本项目要求解的问题是 2025 量旋杯赛题 [量子门脉冲优化](docs/task.md)，具体来说是要优化单、双量子门脉冲，使得量子电路的执行结果与目标结果尽可能接近。

## 环境准备

本项目基于 `python = 3.11.13` 构建，所需依赖包见 `requirements.txt` 


## 文件说明

需要注意，我们在项目中的 `.npy` 文件，若命名带有 `_best` 后缀，表示该文件为最佳结果脉冲。

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
│   ├── task.md                     # 赛题详细说明
│   └── note.md                     # 技术笔记
└── requirements.txt                # Python依赖包列表
```

### 单比特门优化 (single_qubit/)

**主要文件：**
- `solution.ipynb` - 完整的单比特√X门优化实现，使用SPSA算法结合参数化脉冲设计
- `spsa_utils.py` - 包含脉冲生成、优化器、评估工具等核心函数
- `single_transmon_grader.py` - 优化版评分器，支持并行计算加速

**优化特点：**
- 参数化脉冲设计（30步压缩为10个结点）
- 物理约束处理（tanh函数约束幅度）
- 多进程并行计算（性能提升5倍）
- 鲁棒性设计（多随机种子平均）

### 双比特门优化 (two_qubit/)

**主要文件：**
- `cnot_solution.ipynb` - 双比特CNOT门优化实现，采用两阶段优化策略
- `cnot_closed.py` - 闭合系统GRAPE优化，为开放系统优化提供良好起点
- `cnot_spsa_utils.py` - 双比特优化的工具函数
- `two_transmon_grader.py` - 优化版评分器，支持并行计算

**优化特点：**
- 两阶段优化策略（闭合系统+开放系统）
- 直接脉冲优化（300维参数空间）
- 多进程并行计算（性能提升10倍）
- 针对耦合量子比特系统的特殊处理