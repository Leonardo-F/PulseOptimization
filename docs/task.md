# 量子门优化挑战赛:开放系统下的鲁棒脉冲设计

## 目录

- [赛题背景](#赛题背景)
- [**题目一：单比特门优化（√X门）**](#题目一单比特门优化x门)
  - [物理系统](#物理系统-单比特)
  - [优化目标](#优化目标-单比特)
  - [评分标准](#评分标准-单比特)
  - [提交要求](#提交要求-单比特)
- [**题目二：双比特门优化（CNOT门）**](#题目二双比特门优化cnot门)
  - [物理系统](#物理系统-双比特)
  - [优化目标](#优化目标-双比特)
  - [评分标准](#评分标准-双比特)
  - [提交要求](#提交要求-双比特)
- [快速开始](#快速开始)
- [文件说明](#文件说明)
- [技术支持](#技术支持)

---

## 赛题背景

在实际的量子计算系统中，量子比特不可避免地会受到环境噪声的影响，包括：

- **退相干**：能量弛豫（$T_1$）和纯退相干（$T_\varphi$）
- **系统参数波动**：频率漂移等shot-to-shot fluctuations
- **泄漏**：布居数泄漏到计算子空间之外的能级

传统的闭合系统脉冲优化方法（如GRAPE）在理想系统中可以获得很高的保真度，但将这些脉冲应用到真实的开放系统时，性能会显著下降。本次挑战赛要求参赛者设计**鲁棒的量子门脉冲**，在开放系统环境下仍能保持高保真度。

本挑战赛提供**两个赛题**，参赛者可以选择其中一个或同时完成两个：
- **题目一**：单比特门优化（√X门）
- **题目二**：双比特门优化（CNOT门）

---

## 题目一：单比特门优化（√X门）

### 物理系统 (单比特)

我们考虑一个超导transmon量子比特系统，建模为弱非谐振子：

**裸量子比特哈密顿量**：
$$
H_q/\hbar = \omega_q a^\dagger a + \frac{\alpha}{2} a^\dagger a^\dagger a a
$$

- $\omega_q/(2\pi) \approx 5$ GHz：量子比特频率
- $\alpha/(2\pi) \approx -200$ MHz：非谐性

**微波驱动项**

通过电容耦合的微波驱动用以下哈密顿量描述:

$$
H_{\mathrm{d}} / \hbar=\Omega_{\mathrm{RF}}(t)\left(a^{\dagger}+a\right)
$$

**驱动信号形式:**
$$
\Omega_{\mathrm{RF}}(t)= \Omega_{\mathrm{I}}(t) \cos \left(\omega_{\mathrm{d}} t+\varphi\right)+\Omega_{\mathrm{Q}}(t) \sin \left(\omega_{\mathrm{d}} t+\varphi\right)
$$

**参数说明:**

- $\Omega_{\mathrm{I}}(t)$ 和 $\Omega_{\mathrm{Q}}(t)$ — 同相(I)和正交(Q)分量的驱动幅度
- $\omega_{\mathrm{d}} /(2 \pi)$ — 驱动频率
- $\varphi$ — 由虚拟Z旋转(Virtual Z-rotation)累积的相位

**旋转波近似(RWA)**

为了简化计算,我们转换到以量子比特频率旋转的参考系。使用幺正变换:$$U= \exp \left(\mathrm{i} \omega_{\mathrm{q}} a^{\dagger} a t\right)$$

在旋转参考系下并应用旋转波近似后,得到有效哈密顿量:

$$
\begin{aligned} H_{\mathrm{R}} / \hbar = & \frac{\alpha}{2} a^{\dagger} a^{\dagger} a a+\frac{1}{2}\left\{a^{\dagger} \mathrm{e}^{-\mathrm{i}\left(\omega_{\mathrm{d}}-\omega_{\mathrm{q}}\right) t}\left[\tilde{\Omega}_I(\varphi, t)+\mathrm{i} \tilde{\Omega}_Q(\varphi, t)\right]\right. \\ & \left.+a \mathrm{e}^{\mathrm{i}\left(\omega_{\mathrm{d}}-\omega_{\mathrm{q}}\right) t}\left[\tilde{\Omega}_I(\varphi, t)-\mathrm{i} \tilde{\Omega}_Q(\varphi, t)\right]\right\} \end{aligned}
$$

**其中有效驱动幅度为:**

- $\tilde{\Omega}_I(\varphi, t)=\Omega_{\mathrm{I}}(t) \cos \varphi+\Omega_{\mathrm{Q}}(t) \sin \varphi$
- $\tilde{\Omega}_Q(\varphi, t)= -\Omega_{\mathrm{I}}(t) \sin \varphi+\Omega_{\mathrm{Q}}(t) \cos \varphi$

这两个表达式将累积相位 $\varphi$ 纳入了驱动项中。具体请参考[PRX Quantum 5, 030353](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030353)

**开放系统演化**（Lindblad主方程）：
$$
\frac{d\rho}{dt} = -i[H(t) + \sum_m \delta u_f^{(m)} H_f^{(m)}, \rho(t)] + \mathcal{L}[\rho(t)]
$$

其中：
- $\delta u_f^{(m)}$：频率漂移噪声，$\langle\delta u_f\rangle=0$, $\langle(\delta u_f)^2\rangle=\sigma_f^2$

- $\mathcal{L}[\rho]$：Lindblad超算符，描述退相干过程

  关于频率漂移噪声详细请参考 [Sci. Adv. 11, eadr0875 (2025)](https://www.science.org/doi/10.1126/sciadv.adr0875)

Lindblad超算符定义为：

$$
\mathcal{L}[\rho] = \sum_m \kappa_m\left(L_m \rho L_m^\dagger - \frac{1}{2}L_m^\dagger L_m \rho - \frac{1}{2}\rho L_m^\dagger L_m\right)
$$

跳跃算符（Jump operators）：
- 能量弛豫：$L_{-} = \sqrt{(1+\bar{n})/T_1} \cdot a$
- 热激发：$L_{+} = \sqrt{\bar{n}/T_1} \cdot a^\dagger$
- 纯退相干：$L_{\varphi} = \sqrt{1/T_\varphi} \cdot a^\dagger a$

其中 $\bar{n}$ 是热平衡布居数，$T_1$ 是能量弛豫时间，$T_\varphi$ 是纯退相干时间。

### 优化目标 (单比特)

设计一个**$\sqrt{X}$门**（等价于 $R_X(\pi/2)$），即：

$$
\sqrt{X} = \frac{1}{2}\begin{pmatrix}
1+i & 1-i \\
1-i & 1+i
\end{pmatrix}
$$

要求脉冲在以下条件下仍能保持高性能：
- 退相干：$T_1 = 50$ μs, $T_\varphi = 30$ μs
- 热稳态布居：$\bar{n} = 0.05$
- 频率漂移：$\sigma_f = 0.5$ MHz（shot-to-shot fluctuations）
- 门时间：固定为 **15 ns**
- 时间步长 $\mathrm{dt}$ 固定为 **0.5 ns**

### 评分标准 (单比特)

评分器会从以下几个维度评估你的脉冲：

#### 1. 门保真度（80%权重）

使用6个基准态（cardinal states）评估：

$$
\varepsilon_g \approx 1 - \frac{1}{6} \sum_i \langle\psi_i| R_X(\pi/2)^\dagger \langle\rho(\psi_i)\rangle R_X(\pi/2) |\psi_i\rangle
$$

基准态包括：$|0\rangle$, $|1\rangle$, $|+\rangle$, $|-\rangle$, $|+i\rangle$, $|-i\rangle$

**得分**：$1 - \varepsilon_g$（门保真度）

#### 2. 泄漏抑制（15%权重）

泄漏到高能级的概率：

$$
L \approx \frac{1}{6} \sum_i \left\{1 - \mathrm{Tr}[|0\rangle\langle0| \langle\rho(\psi_i)\rangle] - \mathrm{Tr}[|1\rangle\langle1| \langle\rho(\psi_i)\rangle]\right\}
$$

**得分**：$\max(0, 1 - L \times 5)$

#### 3. 脉冲质量约束（5%权重）

**幅度惩罚 $P_a$**（防止幅度过大）：

$$
P_a = \frac{A_a}{N} \times \sum_{j,k} \left[\exp\left(\frac{\Omega_{j}^{(k)}}{h_a}\right)^2 - 1\right]
$$

- $h_a = 179$ MHz：幅度阈值
- $A_a = 0.1$：惩罚系数
- $\Omega_j^{(k)}$：第 $j$ 个时间步的第 $k$ 个控制幅度（$k=I,Q$）

**微分惩罚 $P_d$**（减少高频分量）：

$$
P_d = \frac{A_d}{N} \times \sum_{j,k} \left[\exp\left(\frac{\Omega_{j+1}^{(k)} - \Omega_{j}^{(k)}}{h_d}\right)^2 - 1\right]
$$

- $h_d = 22.4$ MHz：微分阈值
- $A_d = 0.1$：惩罚系数

**得分**：$\max(0, 1 - (P_a + P_d))$

#### 总分计算

$$
\text{Overall Score} = 0.80 \times (1-\varepsilon_g) + 0.15 \times \text{leakage\_score} + 0.05 \times \text{penalty\_score}
$$

### 提交要求 (单比特)

请提交以下文件（放在`single_qubit/`子目录下）：

1. **`solution.ipynb`**：**必需**，包含你的优化算法和脉冲计算
   - 完整的脉冲优化代码
   - 中间结果和可视化
   - 最终生成的脉冲
   - 必须能够运行并复现结果

2. **`pulses.npy`**：**必需**，优化得到的脉冲数组
   ```python
   # Shape: (n_steps, 2)，其中 n_steps = 30（对应15ns，dt=0.5ns）
   # pulses[:, 0]: Ω_I(t) 以 rad/s 为单位
   # pulses[:, 1]: Ω_Q(t) 以 rad/s 为单位
   np.save('pulses.npy', pulses)
   ```

3. **`report.pdf`**（可选）：说明文档
   - 优化方法说明
   - 关键技术创新
   - 性能分析和结果讨论

---

## 题目二：双比特门优化（CNOT门）

### 物理系统 (双比特)

我们考虑两个通过共享传输线谐振腔（"cavity"）耦合的transmon量子比特系统。

**物理原理**

两个transmon之间的耦合通过共享传输线谐振腔实现。每个transmon量子比特的跃迁能量分别记为$\omega_1$（左比特）和$\omega_2$（右比特）。更高的能级按照Duffing振子模型，具有非谐性$\alpha_1$、$\alpha_2$。每个量子比特与谐振腔的耦合强度分别为$g_1$、$g_2$。

在色散极限$\left|\omega_i-\omega_r\right| \gg\left|g_i\right|$ ($i=1,2$)下，其中$\omega_r$是谐振腔频率，谐振腔可以被绝热消除，得到有效的双比特哈密顿量。每个transmon与谐振腔的耦合转化为有效的比特间耦合：

$$
J \approx \frac{g_1 g_2}{\left(\omega_1-\omega_r\right)}+\frac{g_1 g_2}{\left(\omega_2-\omega_r\right)}
$$

在当前大多数实验装置中，$J \ll\left|\omega_2-\omega_1\right|$，在旋转波近似下，双transmon系统的哈密顿量可以近似为：

**系统参数** （参考 [Quantum 6, 871 (2022)](https://quantum-journal.org/papers/q-2022-12-07-871/)）：

| 参数 | 符号 | 值 |
|------|------|------|
| 左比特频率 | $\omega_1/(2\pi)$ | 4.380 GHz |
| 右比特频率 | $\omega_2/(2\pi)$ | 4.614 GHz |
| 旋转参考系频率 | $\omega_d/(2\pi)$ | 4.498 GHz |
| 左比特非谐性 | $\alpha_1/(2\pi)$ | 210 MHz |
| 右比特非谐性 | $\alpha_2/(2\pi)$ | 215 MHz |
| 有效比特间耦合 | $J/(2\pi)$ | -3 MHz |
| 相对耦合强度 | $\lambda$ | 1.03 |

**有效哈密顿量**（旋转参考系，RWA）：

$$
\hat{H} = \hat{H}_0 + \Omega_{\mathrm{re}}(t) \hat{H}_{d,\mathrm{re}} + \Omega_{\mathrm{im}}(t) \hat{H}_{d,\mathrm{im}}
$$

其中（$\hbar = 1$）：

$$
\begin{aligned}
\hat{H}_0 &= \sum_{q=1,2}\left[\left(\omega_q-\omega_d+\frac{\alpha_q}{2}\right) \hat{b}_q^{\dagger} \hat{b}_q-\frac{\alpha_q}{2}\left(\hat{b}_q^{\dagger} \hat{b}_q\right)^2\right]+J\left(\hat{b}_1^{\dagger} \hat{b}_2+\hat{b}_1 \hat{b}_2^{\dagger}\right) \\
\hat{H}_{d,\mathrm{re}} &= \frac{1}{2}\left[\left(\hat{b}_1^{\dagger}+\hat{b}_1\right)+\lambda\left(\hat{b}_2^{\dagger}+\hat{b}_2\right)\right] \\
\hat{H}_{d,\mathrm{im}} &= \frac{\mathrm{i}}{2}\left[\left(\hat{b}_1^{\dagger}-\hat{b}_1\right)+\lambda\left(\hat{b}_2^{\dagger}-\hat{b}_2\right)\right]
\end{aligned}
$$

- $\hat{b}_q^{\dagger}$, $\hat{b}_q$：transmon $q$ 的产生和湮灭算符
- $\Omega_{\mathrm{re}}(t)$, $\Omega_{\mathrm{im}}(t)$：实部和虚部驱动幅度（rad/s）
- $J$：有效比特间耦合强度
- $\lambda$：第二个比特的相对耦合强度

**说明**：

- Transmon量子比特可以被设计成各种频率、非谐性和耦合强度
- 这里使用的参数来自文献中的典型实验设置
- 驱动哈密顿量$H_{d,\mathrm{re}}$和$H_{d,\mathrm{im}}$允许通过调节$\Omega_{\mathrm{re}}(t)$和$\Omega_{\mathrm{im}}(t)$来同时驱动两个量子比特

**开放系统演化**：与单比特情况类似，包括：
- 每个比特的 $T_1$, $T_\varphi$ 退相干
- 每个比特的频率漂移噪声（shot-to-shot）
- Ensemble averaging over 多次shots

### 优化目标 (双比特)

设计一个**CNOT门**（qubit-1 作为控制比特，qubit-2 作为目标比特）：

$$
\text{CNOT} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

要求脉冲在以下条件下仍能保持高性能：
- 每个比特的退相干：$T_1 = 50$ μs, $T_\varphi = 30$ μs
- 热稳态布居：$\bar{n} = 0$
- 频率漂移：每个比特 $\sigma_f = 0.5$ MHz
- 门时间：**150 ns**
- 时间步长：**0.5 ns**（对应300个时间步）

### 评分标准 (双比特)

评分方法与单比特类似，但使用36个初始态（6×6的张量积态）：

#### 1. 门保真度（80%权重）

使用36个基准态评估（6个单比特态的张量积）：

$$
\varepsilon_g \approx 1 - \frac{1}{36} \sum_i \langle\psi_i| \text{CNOT}^\dagger \langle\rho(\psi_i)\rangle \text{CNOT} |\psi_i\rangle
$$

**得分**：$1 - \varepsilon_g$

#### 2. 泄漏抑制（15%权重）

从计算子空间（$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$）的泄漏：

$$
L \approx \frac{1}{36} \sum_i \left\{1 - \mathrm{Tr}[\hat{P}_{\mathrm{comp}} \langle\rho(\psi_i)\rangle]\right\}
$$

其中 $\hat{P}_{\mathrm{comp}}$ 是计算子空间的投影算符。

**得分**：$\max(0, 1 - L \times 5)$

#### 3. 脉冲质量约束（5%权重）

幅度和微分惩罚的计算方式与单比特相同：
- $h_a = 200$ MHz：幅度阈值
- $h_d = 2.7$ MHz：微分阈值
- $A_a = A_d = 0.1$：惩罚系数

#### 总分计算

$$
\text{Overall Score} = 0.80 \times (1-\varepsilon_g) + 0.15 \times \text{leakage\_score} + 0.05 \times \text{penalty\_score}
$$

### 提交要求 (双比特)

请提交以下文件（放在`two_qubit/`子目录下）：

1. **`solution.ipynb`**：**必需**，包含你的优化算法和脉冲计算
   - 完整的脉冲优化代码
   - 中间结果和可视化
   - 最终生成的脉冲
   - 必须能够运行并复现结果

2. **`pulses.npy`**：**必需**，优化得到的脉冲数组
   ```python
   # Shape: (n_steps, 2)，例如 (300, 2) 对应150ns，dt=0.5ns
   # pulses[:, 0]: Ω_re(t) 以 rad/s 为单位
   # pulses[:, 1]: Ω_im(t) 以 rad/s 为单位
   np.save('pulses.npy', pulses)
   ```

3. **`report.pdf`**（可选）：说明文档
   - 优化方法说明
   - 关键技术创新
   - 性能分析和结果讨论

---

## 快速开始

### 环境配置

```bash
# 安装必要的包
pip install numpy scipy qutip jax jaxlib matplotlib
```

### 1. 理解问题

先运行闭合系统示例，了解GRAPE优化方法：
```bash
# 运行闭合系统示例
jupyter notebook closed_system_example.ipynb
```

### 2. 选择题目并设计优化算法

**方法建议**：

**选项A：扩展GRAPE到开放系统**
- 使用QuTiP的`mesolve`进行演化
- 考虑ensemble averaging over frequency noise
- 优化目标包含门保真度和泄漏

**选项B：其他优化方法**
- CRAB (Chopped Random Basis)
- 参数化脉冲（高斯、DRAG等）+ 优化
- 强化学习方法
- 其他创新方法

### 3. 测试你的脉冲

**对于单比特门（√X）：**

```python
import numpy as np
from single_transmon_grader import TransmonPulseGrader

# 初始化评分器
grader = TransmonPulseGrader(
    n_levels=4,
    n_steps=30,
    dt=5e-10,
    n_shots=15
)

# 加载脉冲
pulses = np.load('pulses.npy')  # Shape: (30, 2)
phi = 0.0

# 评分
results = grader.grade_submission(pulses, phi, verbose=True)
grader.save_results(results, 'results.json')
```

**对于双比特门（CNOT）：**

```python
import numpy as np
from two_transmon_grader import DispersiveCNOTPulseGrader

# 初始化评分器
grader = DispersiveCNOTPulseGrader(
    nq_levels=3,
    n_steps=300,
    dt=5e-10,
    n_shots=10
)

# 加载脉冲
pulses = np.load('pulses.npy')  # Shape: (300, 2)

# 评分
results = grader.grade_submission(pulses, verbose=True)
grader.save_results(results, 'results.json')
```

---

## 文件说明

### Tutorial和示例代码

- **`closed_system_example.ipynb`**：闭合系统GRAPE示例
  - 完整的GRAPE实现（使用JAX自动微分）
  - 从 $|0\rangle$ 到 $|1\rangle$ 的状态转移优化
  - 可视化和分析工具
  - 作为起点参考，但需要扩展到开放系统

### 评分器

- **`single_transmon_grader.py`**：单比特门（√X）官方评分器
  - 使用QuTiP进行开放系统演化仿真
  - 包含ensemble averaging（多次shot平均）
  - 计算门误差、泄漏、惩罚项
  - 提供详细的评分报告

- **`two_transmon_grader.py`**：双比特门（CNOT）官方评分器
  - 基于色散极限的双比特系统模型
  - 评估36个初始态
  - 计算门误差、泄漏、惩罚项
  - 提供详细的评分报告

### 文档

- **`README_EN.md`**：本文档的英文版本

---

## 技术支持

有问题？查看以下资源：

1. **理论背景**：
   - PRX Quantum 5, 030353 (开放系统量子控制)
   - Appl. Phys. Rev. 6, 021318 (2019)
   - Sci. Adv. 11, eadr0875 (2025)
   - Quantum 6, 871 (2022) (双比特门)

2. **工具文档**：
   - [QuTiP Documentation](http://qutip.org/docs/latest/)
   - [JAX Documentation](https://jax.readthedocs.io/)

3. **示例代码**：
   - 本仓库的 `closed_system_example.ipynb`
   - QuTiP的quantum control examples

---

## 评分流程

1. 参赛者提交脉冲文件和参数
2. 评分器加载脉冲，运行多次shot进行ensemble averaging
3. 评估多个基准态的演化，计算门误差和泄漏
4. 计算幅度和微分惩罚
5. 综合得分并生成详细报告

祝各位参赛者取得优异成绩！

> **在噪声环境下保持高保真度，是量子计算走向实用的关键一步。**
