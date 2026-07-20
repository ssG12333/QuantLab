# Stage 0: 量化基础与硬件基石

> ⏱ 预计学习时间：15-25 小时 | 🎯 难度：⭐
>
> **目标**：从零建立量化直觉——知道浮点数怎么变成整数、为什么 8-bit 能替代 32-bit、量化误差从哪来、
> 硬件上 INT8 矩阵乘法怎么跑的、3-bit/2-bit 的低比特极限在哪、QLoRA 的 NF4 为什么是"非均匀"量化。
> **这个 Stage 不碰一行 PyTorch 量化 API**——Observer、FakeQuantize、QuantStub 等全部留给 Stage 1。
> 这里只做一件事：让你从"浮点数"走到"量化后的整数推理"，每一步都亲手算过。

---

## 目录

1. [开篇：为什么深度学习量化能工作](#开篇)
2. [知识总览](#知识总览)
3. [1. 浮点数在内存中长什么样](#1-浮点数在内存中长什么样)
4. [2. 量化公式：一步一步推导](#2-量化公式一步一步推导)
5. [3. 对称量化 vs 非对称量化](#3-对称量化-vs-非对称量化)
6. [4. 量化粒度：一个 scale 管多大范围](#4-量化粒度)
7. [5. 校准算法：scale 和 zero_point 怎么算](#5-校准算法)
8. [6. 舍入策略：四舍五入就够了吗](#6-舍入策略)
9. [7. 硬件层的真相：VNNI / DP4A / TensorCore 的指令级拆解](#7-硬件层的真相)
10. [8. 量化误差分析：SNR 与误差预算](#8-量化误差分析)
11. [9. Sub-byte 量化：当 8 个值都不够用](#9-sub-byte-量化)
12. [10. NF4 与 Double Quantization：QLoRA 的地基](#10-nf4-与-double-quantization)
13. [11. 从零手写一个完整的 INT8 推理引擎](#11-从零手写)
14. [12. 动手实验](#12-动手实验)
15. [检验标准](#检验标准)

---

## 开篇：为什么深度学习量化能工作

在进入任何公式之前，先回答一个根本性问题：**一个 FP32 权重被四舍五入到 INT8 之后，模型为什么还能用？**
FP32 可以表示约 42 亿个不同值，INT8 只有 256 个。从 42 亿降到 256——差了 7 个数量级。

答案有三个层次。

**第一层——冗余。** 一个 ResNet50 有 2500 万个 FP32 参数，但绝大多数参数对最终预测的贡献微乎其微。
256 个值去近似 42 亿个可能值——99% 的参数精度是"过剩"的。

**第二层——噪声免疫。** 网络在训练中经历过各种噪声：Dropout 随机丢弃 50% 神经元、数据增强随机扰动、
BatchNorm 用 mini-batch 统计量近似总体分布。量化误差不过是另一种噪声——网络天然有容错能力。

**第三层——训练补偿（QAT 的关键）。** 如果在训练中注入量化噪声，模型会把权重"推"到恰好落在量化网格上。
不做训练的 PTQ 只能"忍受"误差，QAT 能"主动避开"误差。这就是 Stage 1.5 和 Stage 2 要讲的内容。

但要做到这一切，你得先理解量化的数学基础。这就是 Stage 0。

---

## 知识总览

```
[1] 浮点数表示 ──→ [2] 量化公式推导 ──→ [3] 对称 vs 非对称
                      │
                      ├──→ [4] 量化粒度 (tensor/channel/group)
                      │
                      ├──→ [5] 校准算法 (MinMax / MSE / KL / Percentile)
                      │
                      └──→ [6] 舍入策略 (RNN / Stochastic Rounding)

[7] 硬件加速原理 (内存带宽 / VNNI 指令级 / DP4A / TensorCore 分块)

[8] 量化误差分析 (SNR 公式 / 误差预算)

[9] Sub-byte 量化 (3-bit / 2-bit 的极限)  ──→ 为什么需要 LSQ (Stage 2)

[10] NF4 + Double Quantization ──→ QLoRA 的地基 (Stage 7)

[11] 手写完整 INT8 推理引擎 → MNIST 验证
```

---

## 1. 浮点数在内存中长什么样

> 学这节时，把自己想象成一个正在设计量化芯片的硬件工程师。你要把 FP32 乘法器替换成 INT8 乘法器——
> 在做这件事之前，必须精确知道"FP32 是什么"和"它浪费了什么"。

### 1.1 画四张位布局图（闭眼能默写）

FP32、FP16、BF16、INT8 是四个不同的 trade-off 点：

```
FP32 (Single Precision):  1 + 8 + 23 = 32 bits
┌─────┬──────────┬─────────────────────────────┐
│ sign│ exponent │         mantissa             │
│1 bit│  8 bits  │         23 bits              │
└─────┴──────────┴─────────────────────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
  范围: [~0.0000000000000000000000000000000000000118, ~340000000000000000000000000000000000000.0], 精度: ~7 位有效十进制

FP16 (Half Precision):  1 + 5 + 10 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  5 bits  │   10 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-15) × (1.mantissa)
  范围: [0.0000000596, 65504] ← exponent 只有 5 bits, 容易溢出/下溢

BF16 (Brain Float 16):  1 + 8 + 7 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  8 bits  │    7 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
  范围同 FP32（exponent 同宽！）, BF16↔FP32 互转只需截断低 16 位

INT8 (量化后的表示):  8 bits, 无 sign/exponent/mantissa 分界
┌──────────────────────────────────┐
│          8-bit integer           │
└──────────────────────────────────┘
  unsigned: [0, 255]   — 非对称量化的范围
  signed:   [-128, 127] — 对称量化的范围
  ★ 关键区别: INT8 没有 exponent！每个相邻值的间距是固定的 = scale
    而 FP32 的相邻值间距随 magnitude 变化 (小值密, 大值疏)
```

**核心洞察**：
- exponent 的宽度决定了**动态范围**，mantissa 的宽度决定了**精度**
- BF16：宽 exponent，窄 mantissa——牺牲精度换范围，适合训练
- FP16：窄 exponent，中等 mantissa——牺牲范围换精度，适合推理
- INT8：没有 exponent，没有 mantissa——范围/精度全部由 scale 决定
- **深度学习不需要 7 位有效数字**——3-4 位就够了。这就是量化能工作的根因

### 1.2 FP8 格式（预告 Stage 9）

FP8 是 2023-2024 年最热的新格式——H100 支持原生 FP8 计算。两种变体：

```
E4M3 (训练前向):  1 + 4 + 3 = 8 bits
  范围: [0.00195, 448.0], 精度: ~1 位有效十进制
  4-bit exponent — 范围比 FP16 窄 (容易溢出), 但 3-bit mantissa 提供了基本的精度

E5M2 (反向梯度):  1 + 5 + 2 = 8 bits
  范围: [0.000015, 57344.0], 精度: ~0.5 位有效十进制
  5-bit exponent — 范围大 (防止梯度溢出), 2-bit mantissa — 梯度不需要高精度
```

FP8 量化相比 INT8 的优势：不需要校准数据来确定 scale——因为 FP8 保留了一个小 exponent，可以自适应数据的动态范围。但 FP8 的硬件成本高于 INT8（需要浮点乘法器）。Stage 9 会详细展开。

### 1.3 动手验证

```python
import struct

def float32_to_bits(f: float) -> str:
    """FP32 浮点数 → 32-bit 二进制"""
    packed = struct.pack('>f', f)
    as_int = int.from_bytes(packed, byteorder='big')
    return f'{as_int:032b}'

def analyze_fp32(f: float) -> dict:
    """分解 FP32 的 sign + exponent + mantissa"""
    bits = float32_to_bits(f)
    sign = int(bits[0])
    exponent_raw = int(bits[1:9], 2)
    mantissa_bits = bits[9:]
    mantissa_val = 1.0
    for i, b in enumerate(mantissa_bits):
        if b == '1':
            mantissa_val += 2**(-(i+1))
    return {
        'bits': bits, 'sign': sign,
        'exponent': exponent_raw - 127,
        'mantissa': mantissa_val,
        'value': (-1)**sign * mantissa_val * 2**(exponent_raw - 127),
    }

# 观察 FP32 的精度随 magnitude 变化
for f in [1.0, 1000.0, 0.0001]:
    r = analyze_fp32(f)
    spacing = 2**(r['exponent'] - 23)  # 相邻 FP32 的间距
    print(f"f={f:8g}: exp={r['exponent']:3d}, "
          f"相邻间距={spacing:.10f}")

# f=1.0:     exp=0,   相邻间距=0.000000119  (小值密)
# f=1000.0:  exp=9,   相邻间距=0.0000610  (大值疏!)
# f=0.0001:  exp=-14, 相邻间距=0.00000000000728 (非常密!)

print(f"\nINT8 只有 FP32 的 {256/2**32*100:.8f}% 表示能力——但配合 scale 就够了")
```

---

## 2. 量化公式：一步一步推导

### 2.1 从实数到整数的映射

量化的本质是找一个**线性映射**，把连续实数域映射到有限整数域：

```
实数 r ∈ [r_min, r_max]  ↔  整数 q ∈ [q_min, q_max]

推导:
  第 1 步: r - r_min                         → 范围 [0, r_max - r_min]
  第 2 步: (r - r_min) / (r_max - r_min)     → 范围 [0, 1]
  第 3 步: × (q_max - q_min) + q_min         → 范围 [q_min, q_max]
  第 4 步: round()                            → 整数

标准形式:
  q = round(r / S + Z)

其中:
  S = (r_max - r_min) / (q_max - q_min)     ← scale，量化步长
  Z = q_min - round(r_min / S)               ← zero_point

反量化:
  r ≈ S × (q - Z)
```

这个"≈"是整个量化的核心——反量化回来的 r 不等于原始值，差值就是**量化误差**。

### 2.2 用具体数值跑一遍

把 [-2.5, 4.7] 量化为 INT8 的 [-128, 127]：

```python
r_min, r_max = -2.5, 4.7
q_min, q_max = -128, 127

S = (r_max - r_min) / (q_max - q_min)  # = 7.2 / 255 ≈ 0.028235
Z = q_min - round(r_min / S)           # = -128 - (-89) = -39

print(f"S = {S:.6f}, Z = {Z}")
# 验证: [-128, 127] → [S×(q_min-Z), S×(q_max-Z)] ≈ [-2.51, 4.67]
```

### 2.3 量化误差的两个来源

量化公式 `q = round(r / S + Z).clamp(q_min, q_max)` 中的两个操作——`round()` 和 `clamp()`——各自产生一种误差。

**首先理解"量程"是什么**：INT8 能表示的整数只有 `[q_min, q_max]`（对称为 [-128, 127]，非对称为 [0, 255]）。反量化回实数域后，INT8 能表示的实数范围是 `[S×(q_min-Z), S×(q_max-Z)]` —— 这就是"量程"。任何落在这个范围之外的实数，int8 都表示不了。

**Round Error（舍入误差）**：`round()` 丢掉的零头。

```
量化: r = 2.503, scale = 0.02, zp = 0
  r / scale = 125.15
  round(125.15) = 125                    ← 丢掉了 0.15
  反量化: 125 × 0.02 = 2.50              ← 原始 2.503 → 变成 2.50

每个元素最多损失 S/2（本例中 0.02/2 = 0.01）。
大批量中正负误差倾向抵消——有些向上舍、有些向下舍。
```

**Clip Error（截断误差）**：`clamp()` 把超出量程的值"一刀切"到边界。

量程 = `[S×(q_min-Z), S×(q_max-Z)]`。对于对称量化 zp=0：
`量程 = [S×(-128), S×127] = [-2.56, 2.54]`

```
1. 一个很大的正数被截断:
  r = 3.0, scale = 0.02, zp = 0
  r / scale = 150.0
  clamp(150.0, -128, 127) = 127         ← 一刀切! 150 → 127
  反量化: 127 × 0.02 = 2.54             ← 原始 3.0 → 变成 2.54，误差 0.46

2. 一个很小的负数被截断:
  r = -3.0, scale = 0.02, zp = 0
  r / scale = -150.0
  clamp(-150.0, -128, 127) = -128       ← 一刀切! -150 → -128
  反量化: -128 × 0.02 = -2.56           ← 原始 -3.0 → 变成 -2.56，误差 0.44
```

**截断的是什么？把谁截断了？** 截断的是"除以 scale 后的值"。当 `r/scale` 超出 `[q_min, q_max]` 时，`clamp()` 把它硬按到边界上。被截断的是**数据值超过量程的那部分**——正值太大被压到 127，负值太小被压到 -128。

**为什么 clip error 比 round error 严重得多？**

```
Round Error:
  每个元素最多损失 S/2 (0.01)，正负随机 → 大批量中抵消
  100 个元素，平均损失 ≈ 0（正负抵消后接近零）

Clip Error:
  一个 outlier 可能损失几十倍的 S：
  原始值 3.0 → 截断后 2.54，损失 0.46 = 23 × (S/2)
  且 clip 永远是同向的（正值只截到上界，负值只截到下界）→ 不存在抵消!
  100 个元素中有 1 个 outlier → 该元素的损失不会和任何其他元素抵消
  → 这个损失直接进入下游层的输入，被后续层的权重矩阵放大
```

**这就是为什么校准算法的核心目标是最小化 clip error**——它不是要减少每个元素的舍入（舍入自己会抵消），而是要决定`r_max`和`r_min`取多大，使得"被 clip 的数据比例×clip 损失的严重程度"最小化。选太小 → 太多元素被截断。选太大 → 量化网格变粗 → 每个元素的舍入误差变大。校准就是在这个权衡中找最优解。

---

## 3. 对称量化 vs 非对称量化

### 3.1 选择的核心逻辑

| | 对称 (Symmetric) | 非对称 (Asymmetric) |
|---|-----------------|-------------------|
| 公式 | `q = round(r / S)` | `q = round(r / S + Z)` |
| Zero Point | **强制为 0** | **可以不为 0** |
| 典型场景 | **权重**（以 0 为中心对称） | **ReLU 激活**（全 ≥ 0，偏态） |
| Scale | `S = max(|min|, |max|) / 127` | `S = (max - min) / 255` |

### 3.2 可视化

```
对称量化 — 适合权重:
     分布:      ----***|***----
              -0.5   0   0.5
     量化尺:  [-128 ......... 127] × S   两头都用上了 ✓

非对称量化 — 适合 ReLU 激活:
     分布:      ****|                   (全 ≥ 0)
                0   2.5
     如果错用对称: [-128 ... 0 ... 127]  负半轴全浪费! ✗
     非对称:      [0 ......... 255]      256 个值全用上 ✓
```

ReLU 后激活全在 [0, 3.0]。用**对称量化**：因为对称强制零点在中间，范围必须取 [-3, 3] 覆盖两头 → INT8 的 [-128, 127] 被映射到 [-3, 3]。数据全 ≥0，所以 [-128, 0) 这 128 个整数等级**永远用不上**——没有数据会落进去。剩下 [0, 127] 这 128 个等级去覆盖 [0, 3.0]，每个等级间距 = 3.0/128 ≈ 0.0234，精度被"稀释"了。

用**非对称量化**：零点可以不对齐 0，直接取 [0, 3.0] 映射到 [0, 255]。全部 256 个等级都在有效数据范围内，每个等级间距 = 3.0/255 ≈ 0.0118 —— **精细了一倍**。

**浪费的不是数据，是等级**：256 个 INT8 等级，对称量化只能用到 128 个，另一半等级"空转"。这并不是说对称量化有什么问题——对称量化是为"数据以 0 为中心"设计的。问题是把**它用在不以 0 为中心的数据上**——就像用一把尺子量桌子宽度，尺子的一半刻度在桌子外边。

### 3.2.1 那个 "0" 到底是什么？—— zero_point 的数学直觉

在可视化图中，对称量化的 `[-128 ... 0 ... 127]` 中间那个 **"0"** 就是 **zero_point (Z)**，它是整个量化公式的灵魂：

```
q = round(r / S + Z)      ← 这是主公式
      ↑ 整数÷实数        ↑ 这是 Z
```

**Z 的含义：INT8 领域里的"整数 0"对应实数领域的哪个位置？**

对称量化 Z = 0 时：
```
INT8 域:   -128    -127    ...    0    ...    126    127
           ↓       ↓              ↓            ↓      ↓
实数域:  -128S  -127S   ...   0.000   ...   126S   127S
                              ↑
                          INT8 的 0 = 实数 0
                          网格以实数 0 为中心，正负完全对称
```

**这就是"对称"两个字的来源**——不是数据对称，而是**量化网格以实数 0 为对称中心**。INT8 的 0 对应实数的 0，正负两侧的刻度完全对称：`+N×S` 对应 `+N`，`-N×S` 对应 `-N`。

非对称量化 Z ≠ 0 时（假设 Z = 39）：
```
INT8 域:   0      ...   39    ...    255
           ↓            ↓            ↓
实数域:  -39S    ...   0.000  ...   216S
           ↑            ↑
    INT8 的 0 ≠ 实数 0    INT8 的 39 = 实数 0
    零点"偏移"到 39 的位置
```

**非对称的"非对称"**在于：INT8 的 0 不再对应实数 0，而是对应实数 `-Z×S`。整个网格"偏移"了 Z 个刻度，使得 256 个等级可以压在 [min, max] 的数据范围内，而不是以实数 0 为中心。

**用数值举例——ReLU 激活 [0, 3.0]：**

对称量化（Z=0）：
```
S = 3.0 / 127 ≈ 0.02362

INT8:   -128   -127   ...    0    ...    126    127
实数:  -3.023  -3.000  ...  0.000  ...  2.976  3.000
        ↑___________________↑   ↑___________________↑
        这 128 个值永远用不到      这 128 个值覆盖 [0, 3.0]
        因为 ReLU 输出 ≥0         每格 0.02362
```

非对称量化（Z=0, 因为 min=0 ⇒ Z = round(-0/0.01176) = 0...其实也是 0）：
等等——当 min=0 时，Z = round(-0/S) = 0，非对称也退化成了 Z=0？没错！但关键区别是：

非对称量化（假设数据是 [1.5, 3.0]，min ≠ 0）：
```
S = (3.0 - 1.5) / 255 ≈ 0.005882
Z = round(-1.5 / 0.005882) ≈ -255 → clamp 到 0（INT8 非对称范围是 [0,255]）
→ 实际 Z = 0（min=1.5 导致零点被迫拉到 0）

但用 [0, 3.0] 时：
S = (3.0 - 0) / 255 ≈ 0.01176
Z = round(-0 / 0.01176) = 0

INT8:   0       1      ...   127    ...   255
实数:  0.000  0.01176  ...  1.494  ...  3.000
       ↑
  INT8 的 0 = 实数 0（巧合，因为数据 min=0）
  但量程是 0..255（全正整），不需要负半轴！
```

**对称 vs 非对称的本质区别不是 Z=0 还是 Z≠0——而是 INT8 的取值范围**：
- 对称：INT8 范围 `[-128, 127]`（有正有负，以 0 为中心）
- 非对称：INT8 范围 `[0, 255]`（全正整数，0 在边界）

即使非对称 Z=0，它也能用上全部 256 个值——因为网格从 0 开始，不需要覆盖负数区域。这就是非对称量化对 ReLU 激活更好的根本原因：**网格的起点（INT8 的 0）可以对齐到数据的最小值**，而不是像对称量化那样强制对齐到实数 0。

### 3.3 代码验证

```python
import torch

acts = torch.relu(torch.randn(10000) * 2.0 + 0.5)
wts  = torch.randn(10000) * 0.3

def quant_sym(x, n=8):
    S = x.abs().max() / (2**(n-1)-1)
    q = torch.round(x / S).clamp(-2**(n-1)+1, 2**(n-1)-1)
    return S*q, S

def quant_asym(x, n=8):
    S = (x.max() - x.min()) / (2**n - 1)
    Z = torch.round(-x.min() / S).clamp(0, 2**n - 1)
    q = torch.round(x / S + Z).clamp(0, 2**n - 1)
    return S*(q-Z), S

# mse_sym_act ≈ 5× mse_asym_act  →  非对称对 ReLU 激活好得多
# mse_sym_wt ≈ mse_asym_wt        →  对称对权重够好甚至更好
```

---

## 4. 量化粒度：一个 scale 管多大范围

### 4.1 三种粒度的直观对比

假设 Conv2d weight 为 `[64, 3, 3, 3]`（64 个输出通道，每个 3×3×3=27 个权重）：

```
Per-Tensor:  所有权重共 1 个 (S,Z)
  → 存储: 8B (1个 scale × 4B + 1个 zero_point × 4B)
  → 问题: 64 个通道的权重大小可能差 50 倍——共用一个 scale → 小通道被大通道"稀释"

Per-Channel:  64 个 (S,Z)，每输出通道一个
  → 存储: 512B (64 × 8B)
  → 权重本身: 1728 × 4B = 6912B, scale 占 7.4%
  → ★ 权重量化的默认选择——性价比最高

Per-Group(128): 每 128 个权重一组，共 1728/128≈14 组
  → 存储: 14 × 8B = 112B
  → ★ LLM 权重量化的默认 (GPTQ/AWQ 的 group_size=128)
  → 精度最高的 per-tensor 量变种——"介于 per-tensor 和 per-channel 之间"
```

### 4.2 Per-Group 为什么是 LLM 的最优解

LLM 的一个 `[4096, 4096]` Linear 权重有 16M 个元素。如果做 per-channel（4096 个 scale）——已经很好，但还不够：

```
Per-Channel: 4096 个 scale — 每组 4096 个元素
Per-Group(128): 16384 个 scale — 每组 128 个元素

组越小 → scale 越"局部" → 对 outlier 的容忍度越高
但不小于 128 的原因: scale 存储开销开始超过精度收益
  group_size=64: 32768 个 scale → 256KB 额外存储 (开始显著了)
  group_size=32: 65536 个 scale → 512KB
```

### 4.3 代码验证

```python
W = torch.randn(64, 3, 3, 3)
# Per-Tensor
S_t = W.abs().max() / 127
mse_t = ((W - torch.round(W/S_t).clamp(-128,127)*S_t)**2).mean()
# Per-Channel
S_c = W.reshape(64,-1).abs().max(dim=1).values / 127
W_q = (torch.round(W.reshape(64,-1) / S_c.unsqueeze(1))
       .clamp(-128,127) * S_c.unsqueeze(1)).reshape_as(W)
mse_c = ((W - W_q)**2).mean()
# Per-Group(9): 每 3×3 group 一个 scale
W_flat = W.reshape(64, -1)  # [64, 9]
n_groups = 9 // 3  # 3 groups per channel
S_g = W_flat.reshape(64, n_groups, -1).abs().max(dim=2).values / 127
W_qg = (torch.round(W_flat.reshape(64, n_groups, -1) /
        S_g.unsqueeze(2)).clamp(-128,127) *
        S_g.unsqueeze(2)).reshape_as(W)
mse_g = ((W - W_qg)**2).mean()
print(f"Per-Tensor: {mse_t:.2e} | Per-Channel: {mse_c:.2e} "
      f"| Per-Group: {mse_g:.2e} | 改进: {mse_t/mse_c:.1f}x / {mse_t/mse_g:.1f}x")
```

---

## 5. 校准算法：scale 和 zero_point 怎么算

校准是一个优化问题：找一对 (S, Z) 使量化误差最小。

### 5.1 四种校准器一览

| 校准器 | 做法 | 优点 | 缺点 |
|--------|------|------|------|
| **MinMax** | 覆盖全部范围 | 最快 | outlier 毁一切 |
| **Percentile** | 覆盖 99.9% 范围 | 抗 outlier | 丢弃 0.1% 数据 |
| **MSE** | 网格搜索最小化 MSE | 理论最优 | 慢 |
| **KL Divergence** | 最小化分布差异 | TensorRT 默认 | 需要直方图 |

### 5.2 完整实现

```python
import torch

class MinMaxCalibrator:
    def calibrate(self, x, symmetric=False, n_bits=8):
        q_max = 2**(n_bits-1)-1 if symmetric else 2**n_bits-1
        if symmetric:
            self.scale = x.abs().max() / q_max
            self.zero_point = torch.tensor(0.)
        else:
            self.scale = (x.max() - x.min()) / q_max
            self.zero_point = torch.round(-x.min()/self.scale).clamp(0, q_max)

class PercentileCalibrator:
    def calibrate(self, x, pct=0.999, n_bits=8):
        s = x.flatten().sort().values; n = len(s)
        lo, hi = s[int(n*(1-pct))], s[int(n*pct)]
        self.scale = (hi - lo) / (2**n_bits - 1)
        self.zero_point = torch.round(-lo / self.scale).clamp(0, 2**n_bits - 1)

class MSECalibrator:
    def calibrate(self, x, n_bits=8, n_bins=100):
        q_max = 2**(n_bits-1)-1
        best_mse, best_amax = float('inf'), x.abs().max().item()
        for amax in torch.linspace(x.abs().max()/100, x.abs().max(), n_bins):
            s = amax / q_max
            mse = ((x - torch.round(x/s).clamp(-q_max,q_max)*s)**2).mean().item()
            if mse < best_mse: best_mse, best_amax = mse, amax.item()
        self.scale = best_amax / q_max; self.zero_point = torch.tensor(0.)

class KLCalibrator:
    """TensorRT 默认。建直方图 → 搜截断点 → 最小化 KL(P||Q_quantized)。"""
    def calibrate(self, x, n_bits=8, n_bins=2048):
        x_abs = x.abs()
        hist = torch.histc(x_abs, bins=n_bins, min=0, max=x_abs.max())
        hist = hist / hist.sum()
        q_max = 2**(n_bits-1)-1
        best_kl, best_t = float('inf'), x_abs.max().item()

        def smooth(p, eps=0.0000000001):
            z, nz = (p==0).float(), (p!=0).float()
            e = eps*z.sum()/nz.sum() if nz.sum()>0 else eps
            return (p + z*e + nz*eps) / (p + z*e + nz*eps).sum()

        for i in range(n_bins//2, n_bins):
            P = smooth(torch.cat([hist[:i], hist[i:].sum().unsqueeze(0)]))
            n_per = i/(q_max+1)
            Q_exp = torch.zeros(i)
            for j in range(q_max+1):
                s, e = int(j*n_per), int((j+1)*n_per)
                if e>s: Q_exp[s:e] = hist[:i][s:e].sum()/(e-s)
            Q = smooth(torch.cat([Q_exp, hist[i:].sum().unsqueeze(0)]))
            kl = (P * (P.log() - Q.log())).sum()
            if kl < best_kl: best_kl, best_t = kl.item(), (i/n_bins)*x_abs.max().item()
        self.scale = best_t / q_max; self.zero_point = torch.tensor(0.)

# ===== 对比 =====
x = torch.randn(10000); x[0], x[1] = 100.0, -50.0  # outlier
for name, cal in [("MinMax",MinMaxCalibrator()),("Percentile",PercentileCalibrator()),
                   ("MSE",MSECalibrator()),("KL",KLCalibrator())]:
    cal.calibrate(x, symmetric=True)
    mse = ((x - torch.round(x/cal.scale).clamp(-127,127)*cal.scale)**2).mean().item()
    print(f"  {name:12s}: scale={cal.scale:.4f}, MSE={mse:.4f}")
# MinMax 被 outlier 拖垮；Percentile/MSE/KL 表现好
```

### 5.3 四种校准器的选择 —— 从数据特征出发

不同的校准器不是"谁更好"，而是"谁更适合你的数据分布"：

- **MinMax**：数据干净、outlier 少的场景（如 ResNet 的中间层）
- **Percentile**：已知有 outlier 但 outlier 是噪声（不是信号）——如输入图片的传感器坏点
- **KL Divergence**：数据分布未知时——工业默认（TensorRT 用这个），通用性最好
- **MSE**：追求 MSE 最优，但不知道是否是"任务最优"

**一个常见的坑**：对 LLM 的激活值不要用 Percentile——LLM 的 outlier channel 不是噪声，是模型"精心"学到的注意力信号。删掉它们 = 删掉核心能力。这就是 Stage 6 的 SmoothQuant 和 AWQ 试图解决的问题。

### 5.4 串联：scale 的前世今生 —— 从公式到 LSQ

上面四种校准器都在做同一件事：**找到一对 (S, Z)，让量化后的数据尽可能接近原始数据**。但你在后面每个 Stage 里看到的 "scale" 是同一个概念的不同演化形态。这里画一条完整的演化线——读完这节，后面学到任何阶段你都知道 "scale 从哪来、怎么被用、最后怎么被取代"。

```
[1] 纯数学阶段 — 一个公式
─────────────────────────
  Stage 0 §2: S = (r_max - r_min) / (q_max - q_min)
               Z = q_min - round(r_min / S)

  这是 scale 的"定义"。你知道 S 和 Z 怎么从数据范围算出来。
  但你只有数据，没有框架——S 和 Z 只是你手写在纸上的两个数。


[2] 校准阶段 — 数据驱动的 S 确定
───────────────────────────────
  Stage 0 §5: 四种校准器 (MinMax / Percentile / MSE / KL)

  现在你有校准数据了。你从训练集中抽几百张图（不需要标签），
  跑一遍模型，记录每层激活值的分布。
  四种校准器提供四种"怎么从数据分布中选 r_max / r_min"的策略。

  核心权衡: 覆盖范围 vs 分辨率
    范围太大 → 每个量化等级间距大 → 精度粗 → Round Error 大
    范围太小 → 太多数据被 clip → Clip Error 大

  ★ 校准器做的事 = 找到 r_max 和 r_min 的最优值，
    然后代入 [1] 的公式算出 S 和 Z。


[3] 框架阶段 — Observer 自动化
──────────────────────────────
  Stage 1 §9: PyTorch 的 Observer 类体系

  不能每一层手动校准——Observer 把 [2] 的校准过程自动化了。
  Observer 的 forward() 在每个 batch 中收集 min/max，
  calculate_qparams() 用 [1] 的公式算出 scale 和 zero_point。

  MovingAverageMinMaxObserver:
    self.min_val += α × (observed_min - self.min_val)      ← EMA 平滑
    self.max_val += α × (observed_max - self.max_val)
    scale = max(|min_val|, |max_val|) / 127                 ← ❄ [1] 的公式!
    zero_point = 0

  ★ Observer 做的事 = 把 [2] 的校准算法嵌入到 PyTorch Module 中，
    让 scale 在每个 forward 时自动更新。


[4] QAT 阶段 — scale 固定
─────────────────────────
  Stage 1.5: 固定 scale QAT

  Observer 在校准阶段收集了 min/max，算出了 scale。
  然后 observer_enabled 被设为 0 —— scale 冻结。
  后续 7-8 个 epoch QAT 训练中，这个被冻结的 scale 全程不变。
  weight 在变，但 scale 不跟着变。

  8-bit (256 个等级): 没事——grid 够密，scale 差一点也能凑合
  4-bit (16 个等级): 开始出问题——scale 微小的偏差被放大
  2-bit (4 个等级):  崩溃——scale 几乎不可能"正好"对

  ★ 固定 scale 的上限在 4-bit —— Stage 1.5 的实验数据证明了这一点。


[5] LSQ 阶段 — scale 可学习
───────────────────────────
  Stage 2: LSQ (Learned Step Size Quantization)

  [4] 的固定 scale 在低比特下不够。LSQ 的解决方案：
  把 register_buffer("scale") 改成 nn.Parameter(scale_init)。

  scale 不再是 Observer 一次性算出来然后冻住的——它和 weight
  一起被梯度下降优化。weight 分布变了 → scale 自动跟着调。

  LSQ 的 backward:
    ∂v̂/∂s = -v/s + round(v̄)                    ← scale 的梯度, ❄ 公式 (6)
    grad_scale = grad_scale / √(N × Q_P)        ← Gradient Scaling, ❄ 公式 (13)

  ★ LSQ 做的事 = 把 scale 从 "Observer 的一次性输出" 升级为
    "和 weight 平级的可学习参数"——量化网格永远对齐当前 weight 分布。
```

**这条线的核心信息**：scale 从一个数学公式 (`S = range / 255`) 开始，经过校准器确定最优范围、Observer 自动化收集统计、QAT 中冻结使用、最终在 LSQ 中变成可学习参数。**后面所有 Stage 里提到的 "scale"，都是这同一个概念——只是它在不同阶段的"状态"不同。** 当你看到 Stage 3 的 GPTQ 在 "逐列优化 weight 的量化 scale"、Stage 7 的 QLoRA 在 "对 scale 做 Double Quantization" 时——你已经在 Stage 0 见过它的原点了。

---

## 6. 舍入策略：四舍五入就够了吗

### 6.1 Round-to-Nearest (RNN)

99% 的场景用 `std::nearbyint`（round-half-to-even，又名 banker's rounding）。批量中正负误差倾向抵消。

### 6.2 Stochastic Rounding — 低比特的救命稻草

**问题**：RNN 的期望不等于原始值。例如 `2.3 → 2.0`（永远向下），`2.7 → 3.0`（永远向上）。

**Stochastic Rounding**：用概率舍入，保证 `E[round_sr(x)] = x`。

```python
def stochastic_round(x):
    """以概率 (x - floor(x)) 向上取整，以概率 (ceil(x) - x) 向下取整"""
    f = torch.floor(x)
    r = torch.rand_like(x)
    return torch.where(r < x - f, f + 1, f)  # 概率 = x的小数部分

x = torch.tensor([2.7, 2.3, 2.5, 2.1])
# RNN:  [3, 2, 3, 2]  均值=2.5  ≠ 2.4 (偏!)
# SR:   ≈ [2.7, 2.3, 2.5, 2.1]  期望=2.4  ✓

# 验证无偏性
rnn = torch.round(x).float()
sr_mean = sum(stochastic_round(x) for _ in range(10000)) / 10000
print(f"RNN: {rnn.tolist()}, 均值={rnn.mean():.2f}")
print(f"SR:  {sr_mean.tolist()}, 均值={sr_mean.mean():.2f}")
# SR 的均值接近 [2.7, 2.3, 2.5, 2.1]
```

**为什么低比特下 SR 变得重要？** 在 8-bit 下，RNN 误差 ≤ `S/2`，在大批量中抵消。在 2-bit 下只有 4 个量化等级——每个等级的间距是 `S`（不是 `S/2`），RNN 误差 ≤ `S/2`，但 4 个等级的采样不足以让误差抵消（因为数据分布通常高度集中在 1-2 个等级）。SR 保证无偏——即使采样不足，期望也不偏。

---

## 7. 硬件层的真相：VNNI / DP4A / TensorCore 的指令级拆解

加速不主要来自"INT8 乘法本身比 FP32 快"。真相是三个层次：**内存带宽 → 指令吞吐 → 矩阵专用硬件**。

### 7.1 第一层：内存带宽瓶颈

```
推理流程:
  HBM (显存) → 搬运权重到寄存器 → 计算单元 → 矩阵乘法 → 写回 HBM
               ↑ 这一步通常比计算慢 3-5× ("带宽受限")

具体算 ResNet50 的第 1 层 Conv (7x7, 3→64, 224×224):
  FP32 weight: [64, 3, 7, 7] = 9408 个 float = 37.6 KB
  INT8 weight: [64, 3, 7, 7] = 9408 个 int8  = 9.4 KB

  FP32 input:  [1, 3, 224, 224] = 602 KB
  INT8 input:  [1, 3, 224, 224] = 602 KB  ← input 通常不量化到 INT8 (需要 FP32 精度)

  每次 Conv 需要搬运: FP32 = 640 KB, INT8 = 611 KB
  带宽瓶颈: 不是 weight (37KB vs 9KB), 而是 input activation (602KB)
  → INT8 加速主要来自: (1) weight 在 L2 cache 中多用 4× 空间; (2) 指令吞吐更高
```

### 7.2 第二层：VNNI / DP4A 指令

**Intel VNNI（`_mm256_dpbusd_epi32`）**：

```
一条指令完成: 4 个 INT8 × UINT8 → INT32 累加

寄存器布局:
  src1 (INT8):  |a3|a2|a1|a0| × 4 组 = 16 个 INT8/lane
  src2 (UINT8): |b3|b2|b1|b0| × 4 组 = 16 个 UINT8/lane
  dst  (INT32): |a3*b3+a2*b2+a1*b1+a0*b0| × 4 组 = 4 个 INT32 累加

一条 dpbusd = 4 条 INT8 mul + 3 条 add + 1 条 accumulate = 等效 8 条 FP 指令
在 1 个 CPU cycle 内完成!
```

**ARM DP4A（`SDOT` 指令）**：ARM 的等价实现，用于移动端（qnnpack 后端），4 个 INT8 乘加 → INT32。

**为什么是"dpbusd"这个名字？** `dp` = dot product, `bu` = byte × unsigned byte, `sd` = signed dword accumulate。指令名直接告诉你数据类型组合。

### 7.3 第三层：TensorCore 的矩阵分块

NVIDIA TensorCore 不是"通用计算单元"——它是**专用于 4×4 矩阵乘加的硬件块**。

```
FP16 TensorCore (A100):
  2 个 4×4 FP16 矩阵 → 4×4 FP16 输出 = 128 FMA/cycle/warp

INT8 TensorCore (A100):
  2 个 4×4 INT8 矩阵 → 4×4 INT32 输出 = 128 FMA/cycle/warp
  → 但 INT8 的 instruction issue 效率更高 (更少 decode overhead)
  → A100: FP16 = 312 TFLOPS, INT8 = 624 TFLOPS → 2× 吞吐!

关键: INT8 TensorCore 的 2× 加速不是因为 "INT8 比 FP16 快两倍"
     而是因为 "INT8 lane 可以塞两倍的 op → 同一 cycle 做双倍工作"
     且 INT32 accumulator 可以用更窄的 datapath (节省功耗和面积)
```

### 7.4 加速的三维度汇总

```
维度 1: 内存带宽  → INT8 weight 是 FP32 的 1/4 → 同一带宽下搬运 4× 更多权重
                      → L2 cache 命中率提高 → 等效加速 2-3×

维度 2: 指令吞吐  → VNNI/DP4A: 1 条指令 = 4 个 INT8 乘 + 累加
                      → 等效 8× 指令效率 (vs 标量 FP32 mul + add)

维度 3: 计算吞吐  → TensorCore INT8: 2× FP16 的 FLOPS
                      → 同一 SM 可以并发更多 warp

总加速 = 带宽增益 (2-3×) × 指令增益 (约 7× 等效) × 吞吐增益 (2×)
       ≈ 不是乘法关系, 而是"谁最瓶颈决定最终加速"
       对大多数推理场景: 瓶颈是带宽 → INT8 加速 ≈ 2-4×
       对计算密集型: 加速可达 10×+
```

---

## 8. 量化误差分析：SNR 与误差预算

### 8.1 均匀量化的 SNR 公式

对于一个在范围 `R` 内均匀分布的信号，用 `n` 比特均匀量化：

```
SNR(dB) ≈ 6.02 × n + 1.76    (均匀量化, 全范围信号)

具体值:
  8-bit: SNR ≈ 49.9 dB  →  信号能量 / 噪声能量 ≈ 98,000
  6-bit: SNR ≈ 37.9 dB  →  信号能量 / 噪声能量 ≈ 6,200
  4-bit: SNR ≈ 25.8 dB  →  信号能量 / 噪声能量 ≈ 380
  3-bit: SNR ≈ 19.8 dB  →  信号能量 / 噪声能量 ≈ 96
  2-bit: SNR ≈ 13.8 dB  →  信号能量 / 噪声能量 ≈ 24
```

**注意**：这是"scale 完美"的理论上限。实际中 scale 有 5-10% 的估计误差 → 实际 SNR 比理论值低。

### 8.2 误差在层间的累积

量化误差不是独立的——它在层间传播和放大：

```
第 1 层输出: y₁ = Ŵ₁ × x₁ + ε₁    (ε₁ = 量化误差)
第 2 层输入: x₂ = σ(y₁) = σ(W₁×x₁ + ε₁)    ← ε₁ 进入了 ReLU!
第 2 层输出: y₂ = Ŵ₂ × x₂ + ε₂
                 = Ŵ₂ × σ(W₁×x₁ + ε₁) + ε₂  ← ε₁ 被 Ŵ₂ 放大!

如果 ReLU 的 threshold 被 ε₁ 跨过 → 一个原本为正的激活变负 → 完全丢失
→ 前几层的量化精度对最终精度的影响远大于后几层
→ 这就是为什么"第一层通常不量化"(或保持 FP32)
```

### 8.3 误差预算分配：哪些层能承受更多误差

```
经验规律（基于 ResNet/BERT 的逐层量化分析）:

  Input layer:  误差容忍度 0%     → 永不量化 (原始像素/embedding)
  Layer 1-2:    误差容忍度 0.5%   → 高精度 (8-bit)
  Layer 3-7:    误差容忍度 2%     → 标准 (8-bit, 可以用更激进的校准)
  Layer 8-N:    误差容忍度 3-5%   → 激进 (可以尝试 4-bit)
  Output layer: 误差容忍度 0%     → 永不量化 (logits/bbox 精度敏感)

★ CLE (Stage 3) 的工作就是让"难以量化"的层变得"容易量化"
   ——通过层间均衡, 把误差容忍度的最低值拉高。
```

---

## 9. Sub-byte 量化：当 8 个值都不够用

### 9.1 低比特的量化网格

当比特数降到 8 以下，量化等级急剧减少：

```
n_bits   等级数    对称范围         相当于……
──────────────────────────────────────────
8        256      [-128, 127]      标准 INT8
6        64       [-32, 31]        可用，大多数任务不掉精度
4        16       [-8, 7]          关键拐点——scale 的精确性变得生死攸关
3        8        [-4, 3]          极低——只有 8 个值，每个 weight 被"四舍五入"到最近的 1/8 等级
2        4        [-2, 1]          只有 4 个值！几乎无法独立表示任何信息
1        2        [-1, 0]          二值化——权重只有 -s 和 0 (或 -s 和 +s)
```

### 9.2 信息论下限

一个 `n` 比特的量化值承载 **n 比特的信息**。一个 FP32 权重承载 32 比特的信息。但大多数权重的"有效信息"远小于 32 比特——因为权重的后 16 位通常是噪声。

```
问题: 一个权重需要多少比特?

ResNet50 / ViT:  通常 6-8 比特就够了 (4-bit 开始明显掉精度)
LLM:             4-6 比特 (因为有 outlier channel, 需要更多比特保护少数通道)
BitNet b1.58:    1.58 比特 (三元: -1, 0, +1, log2(3) ≈ 1.58)
                 → 证明: 只要有 QAT, 权重可以被压缩到接近理论极限
```

### 9.3 Stage 2 的铺垫：为什么低比特需要 LSQ

在 4-bit 下只有 16 个量化等级。如果 scale 偏了 10%，8 个等级中可能有 1 个完全跳过最优权重值——量化网格"对不准"了。而固定 scale QAT（Stage 1.5）的 scale 只靠前几个 epoch 校准——在训练初期权重还在快速变化时做出的决定，到后期可能完全不对。

**这就是 LSQ（Stage 2）要解决的问题**：让 scale 变成可学习的参数，和 weight 一起被梯度优化。scale 永远不会"过时"——它始终跟随当前 weight 分布。

---

## 10. NF4 与 Double Quantization：QLoRA 的地基

### 10.1 均匀量化的浪费——为什么需要"非均匀"

前面所有的量化都假设"量化等级均匀分布"。但数据分布是不均匀的：

```
权重的实际分布（大多数 LLM 层）:
      ***
     *   *
    *     *
  **       **
--+----------+--→ weight value
               ↑ 长尾——有些 weight 很大，但很少

均匀格子: |---|---|---|---|---|---|---|---|
         浪费! ↑ 大部分格子集中在小值区 (数据密)
               ↑ 长尾区格子太少 (数据疏但 outlier 敏感)
```

**NormalFloat (NF)** 的思路：**不做均匀格子。** 假设数据服从正态分布 → 找一个"量化格子的分位点"，使得每个格子的"概率质量"相等。

### 10.2 NF4 的直观理解

```
NF4: 16 个量化等级, 但格子的位置由正态分布的分位数决定:

  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇  Q₈  Q₉  Q₁₀ Q₁₁ Q₁₂ Q₁₃ Q₁₄ Q₁₅
  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
  ↑              间距密 (小值区概率密度高)               间距疏 (大值区概率密度低)↑

关键: 小值区 (大部分 weight) → 精细的格子 → 高精度
      大值区 (少数 outlier) → 粗糙的格子 → 但对 outlier 来说, 粗格子也够了
```

**NF4 的实际 16 个值**（归一化到 [-1, 1]）：

```python
# QLoRA 的 NF4 量化等级 (已预计算, 来自正态分布的分位数)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])
# 注意: 这些值不是均匀的! 靠近 0 处密集, 靠近 ±1 处稀疏
```

**量化过程**：不是 `round(x/S) × S`，而是 **"找最近的 NF4 等级"**：

```python
def quantize_nf4(x, scale):
    """NF4 量化: 归一化 → 找最近的 NF4 等级 → 反归一化"""
    x_norm = x / scale                    # 归一化到 [-1, 1]
    # 对每个元素: 找距离最近的 NF4_LEVELS 中的值
    idx = torch.abs(x_norm.unsqueeze(-1) -
                    NF4_LEVELS.to(x.device)).argmin(dim=-1)
    x_q_norm = NF4_LEVELS.to(x.device)[idx]
    return x_q_norm * scale
```

### 10.3 Double Quantization — 对 scale 再量化

Per-group 量化给每个 group_size=64 的组一个 scale。4096×4096 的权重 → 262,144 个 64 元素组 → 262,144 个 scale（FP32 = 1MB 存储！）。

**Double Quantization 的思路**：这些 scale 值本身也是"可以用量化压缩的"——对 scale 做一次 8-bit 量化。

```
Step 1: 记录 scale 的 FP32 值     → S_fp32  (1MB)
Step 2: 对 S_fp32 做 8-bit 量化   → S_i8 = round((S_fp32 - Z_s) / S_s)
                                      (需额外存 S_s + Z_s, 2×FP32 = 8B)
Step 3: 推理时: S_fp32 ≈ S_i8 × S_s + Z_s
           → 存储: 262K × 1B (INT8) + 8B (FP32) ≈ 256KB + 8B
           → 从 1MB 降到 256KB → 4× 压缩!
```

**QLoRA 的组合**：NF4（数据的非均匀量化）+ Double Quantization（scale 的再次量化）= 一个 65B 的模型只需 ~35GB 显存（FP32 需要 260GB）。

---

## 11. 从零手写一个完整的 INT8 推理引擎

把前面 10 节的知识串起来——手写一个量化 MLP，在 MNIST 上验证。

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

# ===== Step 1: 训练一个 FP32 MLP（省略训练代码）=====
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.view(-1, 784))))

# ===== Step 2: 手写量化层（不依赖 torch.ao）=====
class QuantizedLinear(nn.Module):
    def __init__(self, fp_linear, n_bits=8):
        super().__init__()
        w = fp_linear.weight.data
        # 权重量化 (对称 per-tensor)
        self.S_w = w.abs().max() / (2**(n_bits-1) - 1)
        self.register_buffer("q_w",
            torch.round(w/self.S_w).clamp(-128,127).to(torch.int8))
        self.register_buffer("bias", fp_linear.bias.data.clone()
            if fp_linear.bias is not None
            else torch.zeros(fp_linear.out_features))
        self.S_a = None  # 激活 scale — 由校准数据确定

    def calibrate(self, x_sample):
        """用校准数据确定激活值的 scale"""
        with torch.no_grad():
            out = F.linear(x_sample, self.q_w.float() * self.S_w, self.bias)
            self.S_a = out.abs().max() / 127.0

    def forward(self, x):
        # 激活量化 → 矩阵乘法 → 反量化
        q_a = torch.round(x / self.S_a).clamp(-128, 127).to(torch.int8)
        return F.linear(q_a.float(), self.q_w.float(), None) * \
               (self.S_a * self.S_w) + self.bias

class QuantizedMLP(nn.Module):
    def __init__(self, fp_model, n_bits=8):
        super().__init__()
        self.q_fc1 = QuantizedLinear(fp_model.fc1, n_bits)
        self.q_fc2 = QuantizedLinear(fp_model.fc2, n_bits)

    def calibrate(self, loader):
        x, _ = next(iter(loader))
        with torch.no_grad():
            h = F.relu(self.q_fc1(x.view(-1, 784)))
            self.q_fc2.calibrate(h)
            self.q_fc1.calibrate(x.view(-1, 784))

    def forward(self, x):
        x = F.relu(self.q_fc1(x.view(-1, 784)))
        return self.q_fc2(x)

# ===== Step 3: 验证 =====
model_int8 = QuantizedMLP(trained_fp_model, n_bits=8)
model_int8.calibrate(test_loader)
fp_acc = evaluate(trained_fp_model, test_loader)
int8_acc = evaluate(model_int8, test_loader)
print(f"FP32: {fp_acc:.2f}%  →  INT8: {int8_acc:.2f}%"
      f"  →  Δ = {fp_acc-int8_acc:.2f}%")
# 期望: Δ < 0.5%
```

---

## 12. 动手实验

| # | 实验 | 时间 | 产出 |
|---|------|:--:|------|
| 1 | 用 `QuantizedMLP` 完成 MNIST 8-bit 量化推理 | 30min | 第一个从零完成的量化项目 |
| 2 | 四种校准器在 MLP 上的消融对比（MinMax/Percentile/MSE/KL） | 20min | 精度差异表 |
| 3 | 对比 RNN vs Stochastic Rounding 在 2/3/4-bit 下的误差 | 20min | 理解 SR 的低比特价值 |
| 4 | 计算 ResNet50 第 1 层 Conv 的 FP32 vs INT8 内存搬运量 | 15min | 带宽瓶颈的量化直觉 |
| 5 | 手写 NF4 量化器，对比 4-bit 均匀量化 vs NF4 的 MSE | 30min | NF4 优于均匀量化的实验证据 |
| 6 | 实现 Double Quantization：对 256 个 scale 做 8-bit 压缩，记录压缩比和误差 | 20min | 理解 QLoRA 的 4× 存储压缩 |

### 实验记录模板

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ 实验     │ 方法      │ 比特     │ MSE      │ 备注     │
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ 校准器   │ MinMax   │ 8-bit    │ 0.0123   │ outlier  │
│ ...      │ ...      │ ...      │ ...      │ ...      │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 检验标准

- [ ] 能徒手画 FP32 / FP16 / BF16 / INT8 / FP8(E4M3/E5M2) 的位布局图
- [ ] 能手推 `S = (rmax-rmin)/(qmax-qmin), Z = qmin - round(rmin/S)`
- [ ] 能手写 MinMax / Percentile / MSE / KL 四种校准器的 Python 实现
- [ ] 能解释为什么权重用对称、ReLU 激活用非对称
- [ ] 能写出 per-tensor / per-channel / per-group 三种粒度的 scale 计算代码
- [ ] 能说出 VNNI 指令 `_mm256_dpbusd_epi32` 的一个 cycle 做了什么
- [ ] 能解释为什么 INT8 TensorCore 有 2× FP16 的吞吐
- [ ] 能手算 SNR 公式在 8/6/4/3/2-bit 下的值，解释"拐点"
- [ ] 能手写 stochastic rounding 并证明其无偏性
- [ ] 能说出 NF4 和均匀 4-bit 的根本区别（量化格子的分布不同）
- [ ] 能实现 Double Quantization：对一个 scale 数组做 8-bit 压缩
- [ ] 能从零手写 QuantizedLinear + calibrate → 在 MNIST 上推理

---

> 💡 **学习建议**：这个 Stage 的核心目标是"形成量化直觉"——不碰任何 PyTorch 量化 API。
> 所有公式旁边都有可运行的 Python 代码，看完公式马上跑代码。
>
> §7（硬件）、§9（sub-byte）、§10（NF4）是三个"知识钩子"——不需要完全精通，
> 但要知道它们分别在哪一章被展开（硬件→Stage 4/5/8、sub-byte→Stage 2、
> NF4→Stage 7）。
>
> 这样进入 Stage 1 时，你已经有完整的底层认知，不会被 API 淹没。
>
> Next: [Stage 1: PyTorch QAT PT2E 深度拆解](./Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md)
