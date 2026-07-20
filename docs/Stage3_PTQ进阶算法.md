# Stage 3: PTQ 进阶算法 — AdaRound / FlexRound / GPTQ 的数学内核

> ⏱ 预计学习时间：20-25 小时 | 🎯 难度：⭐⭐⭐⭐
>
> Stage 0-2 打通了 QAT 的任督二脉——从手写推理引擎到 LSQ 的可学习 scale。
> 但 QAT 有一个硬门槛：**你需要训练数据 + GPU 训练时间**。如果模型是别人训好的、
> 你没有训练代码、或者你只是想快速评估量化效果——PTQ（Post-Training Quantization）
> 才是现实选择。
>
> **这个 Stage 的目标**：把"普通 PTQ 精度烂"变成"进阶 PTQ 精度接近 QAT"。
> 核心手段——优化舍入决策（AdaRound / FlexRound）、均衡权重分布（CLE）、
> 修正系统性偏差（Bias Correction）、逐层重建（GPTQ）。

---

## 目录

1. [开篇：从 QAT 到 PTQ——为什么需要"进阶"PTQ](#开篇)
2. [知识总览](#知识总览)
3. [1. AdaRound：舍入不是"四舍五入就行"](#1-adaround)
4. [2. AdaRound 的数学推导：Taylor → QUBO → Soft Relaxation](#2-adaround-的数学推导)
5. [3. 从零实现 AdaRound](#3-从零实现-adaround)
6. [4. FlexRound：element-wise division 的巧思](#4-flexround)
7. [5. Cross-Layer Equalization (CLE)：让每层一样好量化](#5-cross-layer-equalization)
8. [6. Bias Correction：只修正 bias 就白赚精度](#6-bias-correction)
9. [7. GPTQ：从 OBQ 到千亿参数](#7-gptq)
10. [8. 动手实验](#8-动手实验)
11. [检验标准](#检验标准)

---

## 开篇：从 QAT 到 PTQ——为什么需要"进阶"PTQ

Stage 2 结束时你掌握了 LSQ：让 scale 和 weight 一起训练，2-bit 精度从 68.5% 拉到 82.8%。这是 QAT 的巅峰——但 QAT 有一个你绕不过去的前提：**你得有训练管线。**

现实中你经常遇到的情况是：
- HuggingFace 上下载了一个 LLaMA-7B，没有训练代码
- 客户给了一个 .pth 权重文件，要求"帮我压到 4-bit 部署"
- 想快速评估一个模型的量化可行性，不想等 20 个 epoch

这些场景下 PTQ 是唯一的选择。但普通 PTQ（MinMax 校准 + round-to-nearest）精度经常很差——尤其在 ≤4-bit 时。

**进阶 PTQ 不会取代 QAT，但会把 PTQ 的精度上限推到接近 QAT 的水平。** 它的武器库有四件：

```
普通 PTQ:  校准 scale → round(weight/scale) → 完事
                ↓ 精度损失: 5-15%

进阶 PTQ:
  ├── AdaRound:  优化每个权重的舍入方向 (向上 vs 向下)
  ├── CLE:       均衡相邻层的权重大小，减少量化难度差异
  ├── Bias Corr: 修正量化引入的激活值均值偏移
  └── GPTQ:      用 Hessian 信息做逐列重建，把误差降到最低
                ↓ 精度损失: 1-3% (接近 QAT!)
```

---

## 知识总览

```
                           ┌──────────────────────────┐
                           │   普通 PTQ 的精度瓶颈      │
                           │   round-to-nearest 不是    │
                           │   "任务损失最小"的舍入     │
                           └────────────┬─────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
    ┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
    │  AdaRound        │    │  CLE + Bias Corr    │    │  GPTQ             │
    │  优化舍入决策     │    │  优化权重的可量化性  │    │  逐列重建 + Hessian│
    │  (per-weight)    │    │  (per-layer均衡)    │    │  (per-column补偿) │
    └────────┬────────┘    └──────────┬──────────┘    └────────┬─────────┘
             │                        │                        │
             │    ┌───────────────────┘                        │
             │    │                                            │
             ▼    ▼                                            ▼
    ┌──────────────────┐                          ┌──────────────────────┐
    │   FlexRound       │                          │   OBQ (前身)          │
    │   ÷ 代替 +         │                          │   Optimal Brain       │
    │   更自然的舍入方式  │                          │   Surgeon 的量化版    │
    └──────────────────┘                          └──────────────────────┘
```

四件武器不是互斥的——实际使用中通常组合：**CLE + AdaRound + GPTQ**。

---

## 1. AdaRound：舍入不是"四舍五入就行"

### 1.1 一个反直觉的实验

取一个训练好的 3 层 MLP。用 MinMax 校准确定 scale，然后对比两种舍入方式：

```python
import torch
import torch.nn as nn

# 假设已经确定了 per-channel scale
w = torch.randn(64, 64) * 0.5   # 某层权重
s = w.abs().max() / 127          # 对称量化 scale

# 方式 A: round-to-nearest
w_q_rnn = torch.round(w / s).clamp(-128, 127) * s
mse_rnn = ((w - w_q_rnn) ** 2).mean()

# 方式 B: 随机舍入（randomly flip some rounding decisions）
def random_round(x, flip_prob=0.3):
    """随机翻转 30% 的舍入方向"""
    floor = torch.floor(x)
    rnn = torch.round(x)
    flip = torch.rand_like(x) < flip_prob
    # flip: 本来是 round (= floor or ceil), 改为另一个方向
    alt = torch.where(rnn > floor, floor, floor + 1)
    return torch.where(flip, alt, rnn)

w_q_rand = random_round(w / s).clamp(-128, 127) * s
mse_rand = ((w - w_q_rand) ** 2).mean()

print(f"Round-to-Nearest MSE: {mse_rnn:.6f}")
print(f"Random Flip MSE:      {mse_rand:.6f}")
```

你会发现：**round-to-nearest 在"权重 MSE"意义上最优，但在"任务损失"意义上不一定。**

因为权重 MSE 最小 ≠ 输出 logits 变化最小 ≠ 最终预测正确率最高。一个对输出影响几乎为零的权重，舍入到"次近"整数导致的 MSE 看似增加了，但模型精度完全不受影响。而一个对输出影响巨大的权重，即使"最近"舍入看着很准——那一点误差也可能通过层层传播被放大。

**AdaRound 的核心思想**：不是最小化权重 MSE——而是最小化**每一层的输出重建误差**（即量化前后该层激活值的差异）。

### 1.2 问题形式化

对一层权重 `W ∈ R^(m×n)`：
- 全精度输出：`Y_fp = W × X`
- 量化输出：`Y_q = Ŵ × X`，其中 `Ŵ = (floor(W/s) + h) × s`，`h ∈ {0, 1}^(m×n)`

目标是找最优的 `h`：

```
min_h || (W × X) - (Ŵ × X) ||_F^2
     = min_h || (W - Ŵ) × X ||_F^2
```

**h 有 2^(m×n) 种可能——暴力搜索不可行。** AdaRound 用了三步工程近似：

1. **泰勒展开** → 把"任务损失"近似为"权重误差的加权平方和"
2. **QUBO 转化** → 把离散优化问题变成无约束二次二值优化
3. **连续松弛** → 用 sigmoid 把二值变量变连续，梯度下降直接优化

---

## 2. AdaRound 的数学推导：Taylor → QUBO → Soft Relaxation

### 2.1 Step 1：泰勒展开——从"任务损失"到"权重误差"

设 L 是任务损失（比如交叉熵）。我们对量化后的权重 `Ŵ` 在 FP32 权重 `W` 处做二阶泰勒展开：

```
L(Ŵ) ≈ L(W)                          ← 第 0 阶（FP32 损失）
      + ∇_W L · (Ŵ - W)              ← 第 1 阶
      + 1/2 · (Ŵ - W)^T · H_W · (Ŵ - W)  ← 第 2 阶
```

其中 `H_W` 是损失对权重的 Hessian 矩阵。

**关键假设 1**：训练已经收敛 → `∇_W L ≈ 0`。一阶项消失。

```
L(Ŵ) - L(W) ≈ 1/2 · (Ŵ - W)^T · H_W · (Ŵ - W)
             ≈ 1/2 · Σ_i H_ii · (Ŵ_i - W_i)^²  （对角近似）
```

**关键假设 2**：Hessian 对角近似。`H_ii` = 损失对第 i 个权重的二阶导数。

现在目标变成：最小化 `Σ_i H_ii · (Ŵ_i - W_i)^²`——这是**加权的权重 MSE**。`H_ii` 越大 = 这个权重对任务越重要 → 舍入误差的惩罚越重。

### 2.2 Step 2：用 Fisher Information 近似 Hessian

真 Hessian 要算二阶导——对于 LLM 不可能。但可以证明：对使用交叉熵损失的分类任务，

```
H_ii ≈ (∂L/∂W_i)²  （Fisher Information 的对角元）
```

而在校准时可以算出一阶梯度。所以：

```python
# 对一批校准数据，算每个权重的 Fisher 对角
fisher_diag = torch.zeros_like(W)
for x, y in calib_loader:
    loss = criterion(model(x), y)
    loss.backward()
    fisher_diag += W.grad ** 2  # 平方梯度累加

# H_ii ≈ fisher_diag / len(calib_loader)（均值）
```

这就是 AdaRound 不用真 Hessian 的原因——用校准数据的平方梯度近似就够了。

### 2.3 Step 3：QUBO 形式

把量化误差写成 `(Ŵ - W) = (floor(W/s) + h) × s - W`。

令 `ΔW_round = W - round(W/s) × s`（round-to-nearest 的误差），
`ΔW_flip` = 从 round 翻转到另一个方向的"额外误差"。

经过整理（略去与 h 无关的常数项），优化目标变成：

```
min_h Σ_i [ H_ii · (ΔW_flip_i)² · h_i + 2 · H_ii · ΔW_round_i · ΔW_flip_i · h_i ]
    = min_h Σ_i a_i · h_i + Σ_i Σ_j Q_ij · h_i · h_j
```

这是一个 **二次无约束二值优化（QUBO）** 问题。`a_i` 是线性项系数，`Q_ij` 是交叉项（来自输出重建误差中的两个权重间的相互作用）。

**交叉项的意义**：权重 i 和权重 j 舍入到"错"方向，它们的误差可能在输出中**相互抵消**或**相互放大**。AdaRound 会利用抵消效应——让两个权重都"故意"舍错，但总输出误差反而更小。

### 2.4 Step 4：连续松弛（Soft Relaxation）

QUBO 仍是 NP-hard。AdaRound 的关键工程技巧：**把二值 h ∈ {0,1} 变成连续变量，用 sigmoid 来近似。**

```python
h_soft = sigmoid(v)        # v 是可学习参数, h_soft ∈ (0, 1) 连续
h_hard = (v >= 0).float()  # 推理时: 硬的 0/1
```

训练时：
```
Ŵ_soft = floor(W/s) × s + h_soft × s     # 可微! 梯度可以穿过 h_soft → v
loss = ||(W - Ŵ_soft) × X||_F^2           # 输出重建误差
```

`sigmoid` 的陡峭程度由温度 τ 控制：`h_soft = σ(v / τ)`。训练初期 τ 大（平滑，好优化），逐步退火到 τ → 0（逼近二值）。

---

## 3. 从零实现 AdaRound

### 3.1 核心类

```python
import torch
import torch.nn as nn

class AdaRoundQuantizer(nn.Module):
    """
    对一层权重做 AdaRound 优化。

    前向: Ŵ = (floor(W/s) + σ(v)) × s  — 软量化, 可微
    推理: Ŵ = (floor(W/s) + (v >= 0)) × s  — 硬量化
    """

    def __init__(self, weight_shape, scale, n_bits=8, symmetric=True):
        super().__init__()
        self.register_buffer("scale", scale)
        self.qmax = 2**(n_bits-1) - 1 if symmetric else 2**n_bits - 1
        self.qmin = -self.qmax if symmetric else 0

        # v 是可学习的舍入参数: v > 0 → 向上取整, v < 0 → 向下
        # 初始化为 0 = 和 round-to-nearest 一致
        self.v = nn.Parameter(torch.zeros(weight_shape))

    def get_soft_rounding(self, w_scaled, temperature=1.0):
        """
        w_scaled = w / s
        返回: floor(w/s) + sigmoid(v/τ) → ∈ (floor, floor+1) 连续
        """
        w_floor = torch.floor(w_scaled)
        soft_ceil = torch.sigmoid(self.v / temperature)
        return w_floor + soft_ceil

    def get_hard_rounding(self, w_scaled):
        """τ → 0 的极限: sigmoid(v/0) → step(v), v≥0 → 1, else → 0"""
        return torch.floor(w_scaled) + (self.v >= 0).float()

    def forward(self, w, temperature=1.0, hard=False):
        w_scaled = w / self.scale
        if hard:
            w_rounded = self.get_hard_rounding(w_scaled)
        else:
            w_rounded = self.get_soft_rounding(w_scaled, temperature)
        w_clamped = torch.clamp(w_rounded, self.qmin, self.qmax)
        return w_clamped * self.scale


def adaround_optimize(weight, scale, input_data, n_iter=10000, lr=0.001):
    """
    优化 AdaRound 的 v 参数。

    Args:
        weight: FP32 权重, shape [out_features, in_features]
        scale: per-channel scale, shape [out_features]
        input_data: 该层的校准输入 X, shape [N_samples, in_features]
    """
    n_bits = 8
    quant = AdaRoundQuantizer(weight.shape, scale, n_bits)

    # 用 Fisher 信息对角作为 Hessian 近似
    # 简化版: 直接用每个权重的平方梯度
    fisher = compute_fisher_diag(weight, input_data)  # 见下方

    opt = torch.optim.Adam([quant.v], lr=lr)

    # 温度退火 schedule
    taus = torch.logspace(1, -1, n_iter)  # τ: 10 → 0.1

    for i in range(n_iter):
        opt.zero_grad()

        # 软量化权重
        w_soft = quant(weight, temperature=taus[i])

        # 输出重建误差（Fisher-weighted）
        w_diff = (weight - w_soft)   # [out, in]
        # 加权 MSE: Σ H_ii · (Ŵ_i - W_i)²
        loss = (fisher * w_diff ** 2).sum()

        loss.backward()
        opt.step()

        if i % 2000 == 0:
            print(f"  Iter {i:5d}, loss={loss.item():.6f}, τ={taus[i]:.3f}")

    # 返回硬舍入后的量化权重
    return quant(weight, hard=True)


def compute_fisher_diag(weight, input_data):
    """用平方梯度近似 Fisher 对角"""
    weight_copy = weight.detach().clone().requires_grad_(True)
    fisher = torch.zeros_like(weight_copy)

    for x in input_data.split(64):  # mini-batch
        out = x @ weight_copy.T
        # 用 MSE 作为代理损失（对分类任务，Fisher = E[(∂L/∂W)²] ≈ (∂L/∂W)²）
        loss = out.pow(2).sum()  # 简化: 直接用输出的二阶矩
        loss.backward()
        fisher += weight_copy.grad ** 2
        weight_copy.grad = None

    return fisher / len(input_data)
```

### 3.2 关键实现细节

**为什么 Fisher 近似可以工作？** 对于已经收敛的模型，`W.grad` 不是零（因为 `W.grad = ∂L/∂W` 的 mini-batch 估计有噪声），但它的平方 `(∂L/∂W)²` 精确地描述了权重对损失的敏感度——即使均值为零，方差非零。

**温度退火 schedule 为什么重要？** 如果一开始 τ 太小，sigmoid 接近阶跃函数——梯度几乎是 0 或不存在，优化不动。从大 τ 开始让梯度平滑，逐步退火让解逼近二值。

**AdaRound 的局限**：per-weight 舍入参数 `v` 本身占用和权重一样大的内存——对于 LLM 不可能每层都存。实践中只对关键层（前几层 + 最后几层）做 AdaRound。

---

## 4. FlexRound：element-wise division 的巧思

### 4.1 AdaRound 的"加法"有什么问题？

AdaRound 的量化公式：

```
Ŵ = (floor(W/s) + h) × s,  h ∈ {0, 1}
```

本质是把舍入错误当作**加法修正**。但量化误差更自然的建模是**乘法缩放**——大权重的舍入误差大（绝对值），小权重的舍入误差小。用统一的 `+s` 修正"大权重少改一点 vs 小权重多改一点"——加法的粒度不对。

### 4.2 FlexRound 的公式

```
Ŵ = floor(W/(s⊙t)) × s
```

其中 `t` 是 element-wise 除法因子（和 W 同 shape）。不改变量化范围（scale s 不变），而是**在除以 scale 之前先对每个权重做微调**。

等价地：

```
Ŵ_i = round(W_i / (s · t_i)) × s
```

`t_i > 1`：缩放后 W_i/(s·t_i) 变小 → 更可能向下取整
`t_i < 1`：缩放后变大 → 更可能向上取整

**和 AdaRound 的对比**：

| 维度 | AdaRound | FlexRound |
|------|----------|-----------|
| 形式 | `floor(W/s) + h` | `floor(W/(s⊙t))` |
| 修正方式 | 加法 | 乘法（除法因子） |
| 对大权重的行为 | 加 s 对小权重影响相对更大 | t 是乘法缩放，对大权重的影响成比例更大 |
| 参数量 | 和 W 一样多 | 和 W 一样多 |
| 理论基础 | Taylor → QUBO | 同上，但用 ÷ 代替 + |

### 4.3 FlexRound 的实现

```python
class FlexRoundQuantizer(nn.Module):
    """FlexRound: 用 element-wise division 替代 AdaRound 的 addition"""

    def __init__(self, weight_shape, scale, n_bits=8):
        super().__init__()
        self.register_buffer("scale", scale)
        self.qmax = 2**(n_bits-1) - 1
        self.qmin = -self.qmax

        # t 初始化为全 1（= 和标准 round 完全一致）
        self.log_t = nn.Parameter(torch.zeros(weight_shape))

    def forward(self, w, temperature=1.0, hard=False):
        t = torch.exp(self.log_t)        # 保证 t > 0
        w_scaled = w / (self.scale * t)  # ← 关键: 元素级除法
        if hard:
            w_rounded = torch.round(w_scaled)
        else:
            # 软舍入: 用 sigmoid 区分 floor 和 ceil
            w_floor = torch.floor(w_scaled)
            w_rounded = w_floor + torch.sigmoid((
                w_scaled - w_floor - 0.5) / temperature
            )
        return torch.clamp(w_rounded, self.qmin, self.qmax) * self.scale


def flexround_optimize(weight, scale, input_data, n_iter=10000):
    """FlexRound 优化——和 AdaRound 的优化循环几乎一样，只改了修正方式"""
    quant = FlexRoundQuantizer(weight.shape, scale)

    fisher = compute_fisher_diag(weight, input_data)
    opt = torch.optim.Adam([quant.log_t], lr=0.001)
    taus = torch.logspace(1, -1, n_iter)

    for i in range(n_iter):
        opt.zero_grad()
        w_soft = quant(weight, temperature=taus[i])
        loss = (fisher * (weight - w_soft) ** 2).sum()
        loss.backward()
        opt.step()

    return quant(weight, hard=True)
```

### 4.4 为什么 log_t 而不是直接学 t？

```python
# 如果直接学 t:
self.t = nn.Parameter(torch.ones(shape))  # t 可能变成负的! 除后方向翻转

# 用 log_t:
self.log_t = nn.Parameter(torch.zeros(shape))  # log(1) = 0
t = torch.exp(self.log_t)  # 始终 > 0, 初始 = 1
```

这是实现细节但很重要——用 `exp` 约束 t 的正性，避免了 t 变负导致权重方向翻转的灾难。

---

## 5. Cross-Layer Equalization (CLE)：让每层一样好量化

### 5.1 问题的根源

考虑一个 MobileNet 的 depthwise conv → pointwise conv 对：

```
DW Conv:  权重范围 [−0.05, 0.05]   ← 很容易量化（范围小）
PW Conv:  权重范围 [−2.5, 2.5]     ← 很难量化（范围大）
```

同样 8-bit，DW Conv 的 scale 精细（每个 level 代表 0.0004），PW Conv 的 scale 粗糙（每个 level 代表 0.02）。**PTQ 的误差被"最难量化的那层"主导。**

### 5.2 ReLU 的 scale 不变性——CLE 的数学基础

ReLU 有一个关键性质：

```
s · ReLU(x) = ReLU(s · x),  对于任意 s > 0
```

这意味着：如果你把上一层的输出乘以 s，再把下一层的输入除以 s——**网络的输出完全不变**。

应用到相邻两层：

```
Layer 1: y₁ = W₁ × x
         y₁_relu = ReLU(y₁)

Layer 2: y₂ = W₂ × y₁_relu

现在插入均衡因子 S > 0:
         y₁' = (W₁ ⊘ S) × x              ← 每输出通道除 S
         y₁_relu' = ReLU(y₁')
         y₂' = (W₂ ⊙ S) × y₁_relu'       ← 每输入通道乘 S
         = W₂ × S × ReLU(W₁/S × x)
         = W₂ × ReLU(S × W₁/S × x)        ← ReLU 的 scale 不变性
         = W₂ × ReLU(W₁ × x)
         = y₂                             ← 完全等价!
```

`W₁ ⊘ S` 表示 `W₁` 的每行（输出通道）除以对应的 `S` 值。

### 5.3 怎么选 S？

选 S 使得**两层的权重大小范围接近**——都容易被量化：

```
S_i = sqrt(max(|W₁[i,:]|) / max(|W₂[:,i]|))

效果: max(|W₁'[i,:]|) ≈ max(|W₂'[:,i]|)
```

### 5.4 完整实现

```python
def cross_layer_equalization(conv1_weight, conv2_weight):
    """
    对 depthwise → pointwise Conv pair 做 CLE。

    conv1_weight: [C_out, C_in/groups, k, k] — depthwise: C_out == C_in
    conv2_weight: [C_out, C_in, 1, 1]       — pointwise: 1×1 conv

    均衡后:
      conv1_weight[i] /= S[i]    ← 第 i 个输出通道缩小
      conv2_weight[:, i] *= S[i]  ← 第 i 个输入通道放大
    """
    with torch.no_grad():
        # 每个通道的权重大小范围
        r1 = conv1_weight.reshape(conv1_weight.size(0), -1).abs().amax(dim=1)
        r2 = conv2_weight.reshape(conv2_weight.size(0), conv2_weight.size(1), -1)
        r2 = r2.abs().amax(dim=2)  # [C_out2, C_in2(=C_out1)]

        # S_i 使得均衡后两层的范围相近
        S = torch.sqrt(r1 / (r2.amax(dim=0) + 0.00000001))

        # 应用均衡
        c1_new = conv1_weight.clone()
        c2_new = conv2_weight.clone()

        for i in range(len(S)):
            c1_new[i] /= S[i]
            c2_new[:, i] *= S[i]

    return c1_new, c2_new


def cle_whole_model(model, layer_pairs):
    """
    对整个模型做 CLE。

    Args:
        model: nn.Module
        layer_pairs: [(conv1_name, conv2_name), ...]
          例: [("features.3.conv.0.0", "features.3.conv.1"),
               ("features.5.conv.0.0", "features.5.conv.1")]
    """
    for name1, name2 in layer_pairs:
        conv1 = dict(model.named_modules())[name1]
        conv2 = dict(model.named_modules())[name2]

        w1_new, w2_new = cross_layer_equalization(conv1.weight, conv2.weight)
        conv1.weight.copy_(w1_new)
        conv2.weight.copy_(w2_new)

        # 如果 conv 后面有 BN，还需要调整 BN 的 γ
        _absorb_scales_to_bn(conv1, conv2, model, name1, name2)


def _absorb_scales_to_bn(conv1, conv2, model, name1, name2):
    """将均衡因子吸收到相邻的 BN 层中（如果存在）"""
    # 如果有 BN: y = γ * (x - μ)/σ + β, 将 1/S 吸收到 γ
    # conv1_bn.weight *= 1/S; conv1_bn.bias *= 1/S
    # conv2_bn.weight *= S;  conv2_bn.bias 不变
    pass  # 具体实现取决于模型结构
```

### 5.5 CLE 的适用范围和局限

CLE 对 MobileNet 系列效果最显著（因为 depthwise + pointwise 对天然存在大的范围不对称）。对标准 ResNet/ViT 也有帮助但效果不如 MobileNet 明显。对 LLM 几乎没用——因为没有相邻 Conv 对这样严格的结构。

**关键限制**：CLE 依赖 ReLU 的 scale 不变性。如果激活函数是 SiLU/Swish/GELU——这些**没有** scale 不变性，CLE 不能用于它们之后的层。

---

## 6. Bias Correction：只修正 bias 就白赚精度

### 6.1 量化引入的"系统性偏移"

权重量化后，每层输出的期望值会发生偏移：

```
E[Y_q] = E[Ŵ × X + b] = E[Ŵ] × E[X] + b
E[Y_fp] = E[W × X + b] = E[W] × E[X] + b
```

因为 `E[Ŵ] ≠ E[W]`（舍入引入了系统性的向上/向下偏移），激活值的均值被"推偏"了。这个偏移在深层会累积，最终导致预测出错。

Bias Correction 的做法很简单：**在量化后用校准数据算一下偏移，然后修改 bias 把它补偿回来。**

### 6.2 实现

```python
def bias_correction(fp_layer, q_layer, input_data):
    """
    对一层做 Bias Correction。

    Args:
        fp_layer: 全精度层 (nn.Conv2d / nn.Linear)
        q_layer: 量化后的层
        input_data: 校准数据（几批输入就够了）

    Returns:
        修正后的 bias
    """
    with torch.no_grad():
        fp_layer.eval()
        q_layer.eval()

        # 收集全精度和量化后的输出
        y_fp_all, y_q_all = [], []
        for x in input_data.split(64):
            y_fp_all.append(fp_layer(x))
            y_q_all.append(q_layer(x))

        y_fp = torch.cat(y_fp_all, dim=0)
        y_q = torch.cat(y_q_all, dim=0)

        # 计算 per-channel 偏移
        bias_corr = (y_fp - y_q).mean(dim=(0, 2, 3))  # Conv2d
        # 或 bias_corr = (y_fp - y_q).mean(dim=0)      # Linear

        # 修正 bias
        corrected_bias = q_layer.bias.data + bias_corr
        q_layer.bias.copy_(corrected_bias)

        print(f"Bias Correction applied, max adjustment: "
              f"{bias_corr.abs().max().item():.6f}")

    return q_layer


def bias_correction_whole_model(fp_model, q_model, calib_loader):
    """对模型的所有可量化层做 Bias Correction"""
    for (name_fp, m_fp), (name_q, m_q) in zip(
        fp_model.named_modules(), q_model.named_modules()
    ):
        if isinstance(m_fp, (nn.Conv2d, nn.Linear)):
            # 需要前一层激活作为 input_data
            # 实践中: 事先收集每层的校准输入
            input_data = get_calib_input_for_layer(name_fp, calib_loader, fp_model)
            bias_correction(m_fp, m_q, input_data)
```

### 6.3 什么时候 Bias Correction 最有用？

- 权重量化后有明显的方向性 bias（比如一批权重都向上取整了）
- 深度网络的后几层（前几层的偏差被放大）
- 非对称量化（zero_point ≠ 0 引入额外的偏移）

Bias Correction 是最"低投入高回报"的技术——只需几批校准数据前向一次，不改变权重也不增加推理开销。

---

## 7. GPTQ：从 OBQ 到千亿参数

### 7.1 为什么需要 GPTQ

AdaRound 优化每层的输出重建误差——但它把所有权重当独立的优化变量。对于 LLM 的一个 `[4096, 4096]` Linear 层，AdaRound 需要 1600 万个舍入参数——这还没算优化过程中的内存峰值。

**GPTQ 的核心贡献**：把舍入优化从"所有列同时优化"改为"逐列优化 + 剩余列用 Hessian 信息补偿"——内存从 O(n²) 降到 O(n·B)（B 是 block size）。

### 7.2 前置：OBQ（Optimal Brain Quantizer）

理解 GPTQ 之前必须先理解 OBQ。OBQ = Optimal Brain Surgeon (OBS) 的量化版。

OBS 的原始问题：**删除一个权重（剪枝）**，如何调整其他权重来补偿？

OBQ 的版本：**量化一个权重**，如何调整其他未量化的权重来补偿？

```python
# OBQ 算法伪代码
# W: [d_row, d_col] 权重矩阵
# H_inv: 逆 Hessian [d_col, d_col], H = X^T X

W_q = W.clone()
quantization_order = []  # 量化顺序（按 Hessian 对角重要性排序）

for col in sorted(range(d_col), key=lambda i: 1/H_inv[i,i]):
    # Step 1: 量化这一列
    w_col = W_q[:, col]
    w_col_q = quantize(w_col)          # 量化
    error = w_col - w_col_q            # 误差 d_row × 1

    # Step 2: 用 H_inv 计算最优补偿
    # 补偿量 = -error / H_inv[col, col]  (除以自己的 Hessian 对角)
    # 应用到所有未量化的列
    for remaining_col in unquantized_cols:
        W_q[:, remaining_col] -= error * (H_inv[col, remaining_col]
                                          / H_inv[col, col])

    quantization_order.append(col)

    # Step 3: 从 H_inv 中删除这一列（矩阵收缩）
    H_inv = remove_row_col(H_inv, col)
```

### 7.3 GPTQ 的两个关键改造

**改造 1：固定顺序，分批处理**

OBQ 每次量化一列（1 个权重 per row），然后更新所有未量化列。对 LLM 的 4096×4096 矩阵，这一步 O(d_col²) 遍历所有列——完全没法用。

GPTQ 的做法：**固定列顺序（从左到右），每次处理 B=128 列作为一个 block。**
Block 内的各列独立量化（不做交叉补偿），整 block 做完后一次性用 Cholesky 更新。

```python
# GPTQ 算法伪代码
B = 128  # block size
H = X^T @ X  # Hessian, shape [d_col, d_col]
# 固定列顺序（按 Hessian 对角重要性降序）
perm = torch.argsort(torch.diag(H), descending=True)
W = W[:, perm]
H = H[perm][:, perm]

# Cholesky 分解: H = L @ L^T
L = torch.linalg.cholesky(H)  # 下三角

# H_inv 的列向量可用 L 高效计算
# 关键性质: L[:, col] 编码了 col 和之前所有列的关系

for block_start in range(0, d_col, B):
    block_end = min(block_start + B, d_col)
    block_cols = range(block_start, block_end)

    # Block 内: 独立量化（不做交叉补偿，用 lazy update）
    for col in block_cols:
        w_col_q = quantize(W[:, col])
        error = W[:, col] - w_col_q
        W[:, col] = w_col_q

        # 补偿应用到本 block 内剩余列
        for remaining in range(col+1, block_end):
            W[:, remaining] -= error * (H[col, remaining] / H[col, col])

    # Block 完成后: 对 block 之后的所有列做全局补偿
    # 用 Cholesky 分解加速!
    E = W_fp[:, block_cols] - W[:, block_cols]  # block 内所有列的误差
    # 补偿 = E @ L[block_end:, block_cols] @ inv(L[block_cols, block_cols])
    # 这一步是 O(d_col · d_row · B) 而不是 O(d_col²)
    W[:, block_end:] -= _cholesky_update(E, L, block_start, block_end)
```

**改造 2：Cholesky 加速全局补偿**

OBQ 的全局补偿需要遍历所有 `(col, remaining_col)` 对——O(d_col²)。GPTQ 的 Cholesky 分解把这一步变成 O(d_col · B)。

Cholesky 分解 `H = L @ L^T` 后，`L[i, j]` 恰好编码了列 i 和列 j 之间的 Hessian 关系。全局补偿公式可以写成矩阵乘法：

```python
def cholesky_global_update(W, E, L, block_start, block_end):
    """
    用 Cholesky 因子 L 加速 block 外的全局补偿。

    E: [d_row, B] — block 内各列的量化误差
    L: 下三角矩阵, H = L @ L^T
    """
    d_col = L.shape[0]
    B = block_end - block_start

    # 从 L 中提取 block 到剩余列的映射
    # 这一步是矩阵乘法, GPU 上极快
    L_block_inv = torch.linalg.inv(L[block_start:block_end,
                                      block_start:block_end])
    # compensation_matrix: [B, d_col - block_end]
    compensation = (L_block_inv.T @
                    L[block_end:, block_start:block_end].T).T

    # W[:, block_end:] -= E @ compensation
    update = E @ compensation  # [d_row, d_col - block_end]
    W[:, block_end:] -= update

    return W
```

### 7.4 GPTQ 的完整流程（简化的可运行版本）

```python
def gptq_quantize_layer(W, X, n_bits=4, group_size=128, block_size=128):
    """
    用 GPTQ 量化一个 Linear 层。

    Args:
        W: [d_out, d_in] 权重矩阵
        X: [N_calib, d_in] 校准输入
        n_bits: 目标比特数
        group_size: per-group scale 的组大小
        block_size: GPTQ 的 block size

    Returns:
        W_q: 量化后的权重 (INT), scale: 量化 scale
    """
    d_out, d_in = W.shape
    device = W.device

    # Step 1: 计算 Hessian H = X^T X
    H = X.T @ X         # [d_in, d_in]
    # 加正则化防止奇异性
    H.diagonal().add_(0.00001 * torch.mean(torch.diag(H)))

    # Step 2: Cholesky 分解
    L = torch.linalg.cholesky(H)  # [d_in, d_in], 下三角

    # Step 3: 固定列顺序 — 按 H 对角降序（重要的列先量化）
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.diag(H) + damp
    perm = torch.argsort(diag, descending=True)
    inv_perm = torch.argsort(perm)

    W = W[:, perm].clone()
    L = L[perm][:, perm]  # 行列同步重排

    # Step 4: per-group scale
    n_groups = d_in // group_size
    scales = torch.zeros(d_out, n_groups, device=device)

    # 存储量化后的权重（INT）
    qmax = 2**(n_bits - 1) - 1
    Q = torch.zeros_like(W, dtype=torch.int8)

    # Step 5: 逐 block 量化
    dead_diag = torch.zeros(d_in, dtype=torch.bool, device=device)

    for block_start in range(0, d_in, block_size):
        block_end = min(block_start + block_size, d_in)

        for col in range(block_start, block_end):
            if dead_diag[col]:
                continue

            w_col = W[:, col]  # [d_out]

            # 确定这一列的 scale (按 group)
            group_idx = col // group_size
            col_group_start = group_idx * group_size
            col_group_end = min(col_group_start + group_size, d_in)

            # 计算 scale: max-abs per output channel in this group
            s = w_col.abs().max() / qmax
            if s == 0:
                dead_diag[col] = True
                continue
            scales[:, group_idx] = torch.maximum(
                scales[:, group_idx], s
            )

            # 量化
            w_col_q = (torch.round(w_col / s).clamp(-qmax, qmax)
                       * s)
            error = w_col - w_col_q          # [d_out]
            W[:, col] = w_col_q
            Q[:, col] = torch.round(w_col / s).clamp(-qmax, qmax).to(torch.int8)

            # Block 内补偿: 对本 block 剩余列用 Hessian
            for remaining in range(col + 1, block_end):
                if not dead_diag[remaining]:
                    W[:, remaining] -= error * (
                        H[col, remaining] / H[col, col]
                    )

        # Block 完成后: 对 block 之后的列做全局补偿
        if block_end < d_in:
            E = W_fp_original[:, block_start:block_end] - W[:, block_start:block_end]
            W[:, block_end:] = cholesky_global_update(
                W[:, block_end:], E, L, block_start, block_end
            )

    # 恢复原始列顺序
    W_final = W[:, inv_perm]
    Q_final = Q[:, inv_perm]
    # 相应的 scale 也需要恢复...

    return W_final, Q_final, scales
```

### 7.5 GPTQ vs AdaRound — 什么时候用什么？

| 场景 | 推荐 |
|------|------|
| 小模型 (<100M 参数), CNN | AdaRound（per-weight 精度最高） |
| 中等模型 (100M-1B) | FlexRound 或 AdaRound（选关键层） |
| LLM (≥1B) | GPTQ（唯一可行的逐列重建方案） |
| 需要最快速度 | PTQ MinMax + Bias Correction（放弃精度换速度） |

### 7.6 GPTQ 的 Hessian = X^T X 是怎么来的？

这是一个容易被跳过的推导——但理解它才能真正理解 GPTQ 的每行代码。

对一层 Linear `Y = W × X`（`W: [d_out, d_in]`, `X: [N, d_in]`），量化后的层输出重建误差：

```
L = ||(W_q - W) @ X||_F^2  = ||ΔW @ X||_F^2
  = trace((ΔW @ X)^T @ (ΔW @ X))
  = trace(X^T @ ΔW^T @ ΔW @ X)
  = trace(ΔW^T @ ΔW @ X @ X^T)       # 循环置换
```

对于逐列量化（每次量化 `W` 的一列 `W[:, col]`），每一列对应的 Hessian 子问题只涉及 `X[col, :]` 和 `X` 中其他列的关系——而这个关系正好由 `H = X^T @ X` 的对应行/列完全描述。

**直觉**：`H[i, j] = X[i, :] · X[j, :]`（校准数据中第 i 和第 j 个输入维度的内积）。如果两个维度高度相关（`H[i, j]` 大），量化维度 i 的误差可以"转移"到维度 j 来补偿——因为改变维度 j 的权重能产生类似的输出变化。

---

## 8. 动手实验

| # | 实验 | 时间 | 产出 |
|---|------|:--:|------|
| 1 | 用 3 层 MLP (MNIST) 实现 AdaRound，对比 round-to-nearest 精度 | 1h | AdaRound vs RNN 在 8/4/2-bit 下的精度表 |
| 2 | 实现 CLE on MobileNetV2，对比均衡前后的 per-channel 权重大小分布 | 45min | 均衡前后的权重大小直方图 |
| 3 | 实现 Bias Correction on ResNet-18 PTQ，逐层画 bias 修正量 | 30min | 哪些层最需要修正的分析 |
| 4 | 用一个小 Transformer (GPT-2 Small) 跑 GPTQ 4-bit，记录每层的量化误差 | 1.5h | GPTQ vs RNN-PTQ 的 perplexity 对比 |
| 5 | （选做）FlexRound vs AdaRound 在 ResNet-20 4-bit 的正面对决 | 45min | 精度 + 收敛速度对比 |

**期望结果**（ImageNet, ResNet-18, 8-bit PTQ）:

```
┌──────────────────────┬────────┬────────┐
│ Method               │ 8-bit  │ 4-bit  │
├──────────────────────┼────────┼────────┤
│ FP32 Baseline        │ 69.8%  │ 69.8%  │
│ PTQ Round-to-Nearest │ 69.2%  │ 62.1%  │
│ PTQ + Bias Correction│ 69.3%  │ 63.5%  │
│ PTQ + CLE            │ 69.4%  │ 65.2%  │
│ PTQ + AdaRound       │ 69.5%  │ 67.8%  │
│ PTQ + CLE + AdaRound │ 69.6%  │ 68.3%  │
│ QAT Baseline (参考)  │ 69.7%  │ 67.5%  │
└──────────────────────┴────────┴────────┘
```

**核心观察**：CLE + AdaRound 的 4-bit PTQ 精度（68.3%）已经接近甚至超过 4-bit 固定 scale QAT（67.5%）。

---

## 检验标准

- [ ] 能手推 AdaRound 的 Taylor 展开 → QUBO 转化过程
- [ ] 能解释温度退火 schedule 在 AdaRound 中的作用
- [ ] 能用 sigmoid 从零实现 AdaRound 的 soft relaxation
- [ ] 能说出 FlexRound 的 element-wise division 相比 AdaRound addition 的优势
- [ ] 能手写 CLE 并解释为什么对 MobileNet 效果显著（画均衡前后的权重大小对比图）
- [ ] 能手写 Bias Correction 并解释为什么量化会引入系统性偏移
- [ ] 能用 Cholesky 分解实现 GPTQ 的 lazy batch update
- [ ] 能说出为什么 GPTQ 用 `X^T X` 作为 Hessian 近似
- [ ] 能根据模型大小选择 PTQ 方案：<100M 用 AdaRound, >1B 用 GPTQ

---

> 💡 **学习建议**：Stage 3 是"理论最密集"的一个阶段——AdaRound 的数学推导 + GPTQ 的 Cholesky 更新是两大硬骨头。
>
> **攻克策略**：
> 1. **AdaRound 先跑代码再看公式**——代码只有 100 行，跑通 MNIST 实验后你自然理解 "soft relaxation" 在做什么
> 2. **GPTQ 不要试图一次看懂**——先理解 OBQ（一个简单得多的 100 行实现），再看 GPTQ 的 block-wise + Cholesky 加速
> 3. **CLE + Bias Correction 可以独立实践**——不需要理解 AdaRound/GPTQ 就能做，效果立竿见影
>
> 完成后回头看：你现在有了 PTQ 的全套武器——可以和 Stage 2 的 LSQ QAT 做全链路对比了。
>
> Next: [Stage 4: YOLO 量化实战](./Stage4_YOLO量化实战.md)
