# Stage 0: 数值量化基础与硬件基石

> ⏱ 预计学习时间：15-25 小时 | 🎯 难度：⭐
>
> **目标**：**从零建立量化直觉**——知道一个浮点数怎么变成整数，知道为什么 8-bit 能替代 32-bit，知道量化误差从哪里来，知道硬件上 INT8 矩阵乘法怎么跑。**并且深入 PyTorch 源码，理解 Observer、FakeQuantize、QuantStub、fuse_modules 这些关键模块的内部机制。**
>
> **阅读建议**：这篇不是浮点数的 Wikipedia 搬运，而是用深度学习量化场景来打量数值格式。Stage 0 的第 1-7 节建立理论地基，第 8-11 节是 PyTorch 源码深潜——理解每个关键模块的内部结构。建议先通读一遍建立全局认知，然后逐节动手跑代码。

---

## 目录

1. [开篇：为什么深度学习量化能工作](#开篇为什么深度学习量化能工作)
2. [知识总览](#知识总览)
3. [1. 浮点数在内存中长什么样](#1-浮点数在内存中长什么样)
4. [2. 量化公式：一步一步推导](#2-量化公式一步一步推导)
5. [3. 对称量化 vs 非对称量化：什么场景选哪个](#3-对称量化-vs-非对称量化什么场景选哪个)
6. [4. 量化粒度：一个 scale 管多大范围](#4-量化粒度一个-scale-管多大范围)
7. [5. 校准算法：scale 和 zero_point 怎么算](#5-校准算法scale-和-zero_point-怎么算)
8. [6. 舍入策略：四舍五入就够了吗](#6-舍入策略四舍五入就够了吗)
9. [7. 为什么量化能加速：硬件层的真相](#7-为什么量化能加速硬件层的真相)
10. [8. PyTorch 源码深潜：Observer 家族](#8-pytorch-源码深潜observer-家族)
11. [9. PyTorch 源码深潜：FakeQuantize 内核](#9-pytorch-源码深潜fakequantize-内核)
12. [10. PyTorch 源码深潜：QuantStub / DeQuantStub 到底是什么](#10-pytorch-源码深潜quantstub--dequantstub-到底是什么)
13. [11. PyTorch 源码深潜：fuse_modules 内部全流程](#11-pytorch-源码深潜fuse_modules-内部全流程)
14. [12. PyTorch 源码深潜：prepare_qat 和 convert 如何消费这些模块](#12-pytorch-源码深潜prepare_qat-和-convert-如何消费这些模块)
15. [13. 动手实验](#13-动手实验)
16. [检验标准](#检验标准)

---

## 开篇：为什么深度学习量化能工作

在进入任何公式之前，先回答一个根本性的问题：**深度学习模型为什么能忍受 INT8 只有 256 个值？**

答案有三个层次：

**第一层——冗余**。一个 FP32 的 ResNet50 有约 2500 万个参数，每个 4 字节。但其中绝大部分参数对最终预测的贡献很小。你可以把 90% 的参数四舍五入到 1 位小数，模型精度几乎不变。深度学习模型被过度参数化了，这是量化能工作的根本原因。

**第二层——噪声免疫**。神经网络在训练过程中已经学会了应对各种噪声——dropout 随机丢弃神经元、数据增强加噪声、BN 层的统计波动。量化引入的误差只是另一种"噪声"，模型可以适应它。

**第三层——训练时补偿（QAT 的关键）**。如果你在训练过程中就注入量化噪声，网络会把权重的值"推"到刚好落在量化网格上，让量化误差最小。这就是 QAT 比 PTQ 好很多的原因——不做训练的 PTQ 只能"忍受"量化误差，QAT 能"主动避开"它。

但要做到这一切，你得先理解量化的数学基础——浮点数怎么表示、scale 怎么算、校准怎么做。这就是 Stage 0 要解决的问题。

---

## 知识总览

```
[1] 浮点数表示 ──→ [2] 量化公式推导 ──→ [3] 对称 vs 非对称选择
                      │
                      ├──→ [4] 量化粒度 (tensor/channel/group)
                      │
                      ├──→ [5] 校准算法 (MinMax / MSE / KL / Percentile)
                      │
                      └──→ [6] 舍入策略 (RNN / Stochastic Rounding)

[7] 硬件加速原理 (内存带宽 / VNNI / DP4A / Tensor Core)

[8-12] PyTorch 源码深潜:
  [8] Observer 家族 (ObserverBase → MinMaxObserver → HistogramObserver → PerChannel)
  [9] FakeQuantize 内核 (FakeQuantizeBase → 状态机 → forward 实现)
  [10] QuantStub / DeQuantStub (占位符 → Observer → 量化/反量化节点的生命周期)
  [11] fuse_modules (模块树遍历 → fuser_method_mapping → 数学折叠 vs 封装)
  [12] prepare_qat / convert (消费 stubs + qconfig → 插入 FakeQuantize → 替换为真量化节点)
```

---

## 1. 浮点数在内存中长什么样

> 学这一节时，把你自己想象成一个正在设计量化芯片的硬件工程师。你要把 FP32 的乘法器替换成 INT8 的乘法器——在做这件事之前，你必须精确地知道"FP32 是什么"和"它浪费了什么"。

### 1.1 画三张位布局图（闭眼能默写）

FP32、FP16、BF16 是三个不同的 trade-off 点。它们都是 IEEE 754 标准下的定义，但各有侧重。画这三张图，它们的核心差别就能一目了然：

```
FP32 (Single Precision):  1 + 8 + 23 = 32 bits
┌─────┬──────────┬─────────────────────────────┐
│ sign│ exponent │         mantissa             │
│1 bit│  8 bits  │         23 bits              │
└─────┴──────────┴─────────────────────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
    exponent-127 的意思是：8-bit exponent 可表示 0~255，减去 127（bias）
    得到实际指数范围 [-127, +128]
    1.mantissa 是"隐含前导 1"——因为有 23 bits mantissa，实际上表示的是 24 bits 精度

FP16 (Half Precision):  1 + 5 + 10 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  5 bits  │   10 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-15) × (1.mantissa)
  实际指数范围: [-14, +15]  →  表示范围 [2^-14, 2^15] ≈ [6.1e-5, 32768]
  这就是 FP16 容易梯度下溢的原因——2^-14 以下的值直接变成 0

BF16 (Brain Float 16):  1 + 8 + 7 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  8 bits  │    7 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
  指数范围: 同 FP32! → 表示范围 [~1.18e-38, ~3.40e38]
  精度只有 7-bit mantissa（相当于 ~2 位有效十进制数字）
  但和 FP32 互转只需截断低 16 位——硬件实现极简单
```

**这三张图的本质差别只在一件事**：exponent 的宽度决定了"你能表示多宽的范围"（动态范围），mantissa 的宽度决定了"你能表示多精确"（精度）。BF16 用窄 mantissa 换取了宽 exponent，而 FP16 做了相反的 trade-off。没有谁绝对好——看场景。

### 1.2 为什么深度学习天然适合低精度

这就要说到一个事实：**深度学习模型的参数不需要那么多位精度。**

一个 FP32 的权重 `0.0039215689`——那后面的 `5689` 对 loss 的影响几乎为零。在 FP32 中，这个数字的 23-bit mantissa 意味着有约 7 位有效十进制数字。但大量研究（和直觉）表明，对于训练好的权重，3-4 位有效数字就足够了。

而且，网络是由层层非线性激活函数（ReLU、SiLU、GELU）串联而成的。非线性操作有天然的"噪声压制"效果——输入的一个小误差在经过非线性变换后往往会被"压低"。Sigmoid 把非常大的范围映射到 [0, 1] 的区间，量化误差在输出端几乎看不出来。

### 1.3 动手验证

```python
import struct
import torch

def float32_to_bits(f: float) -> str:
    """将一个 FP32 浮点数转为其精确的 32-bit 二进制表示"""
    packed = struct.pack('>f', f)
    as_int = int.from_bytes(packed, byteorder='big')
    return f'{as_int:032b}'

def analyze_fp32(f: float) -> dict:
    """把 FP32 分解为 sign + exponent + mantissa"""
    bits = float32_to_bits(f)
    sign = int(bits[0])
    exponent_raw = int(bits[1:9], 2)
    exponent_actual = exponent_raw - 127
    mantissa_bits = bits[9:]

    # 重建 mantissa 值（隐含 1. + 小数部分）
    mantissa_val = 1.0
    for i, b in enumerate(mantissa_bits):
        if b == '1':
            mantissa_val += 2 ** (-(i + 1))

    value = (-1)**sign * mantissa_val * (2**exponent_actual)
    return {
        'bits': bits,
        'sign': sign,
        'exponent_raw': exponent_raw,
        'exponent_actual': exponent_actual,
        'mantissa_bits': mantissa_bits,
        'mantissa_val': mantissa_val,
        'value': value,
    }

# 实验 1: 看 3.14159 的 FP32 内部
r = analyze_fp32(3.1415926)
print(f"值: {r['value']}")
print(f"二进制: {r['bits']}")
print(f"  Sign: {r['sign']} | Exponent: {r['exponent_raw']} (实际={r['exponent_actual']})")
print(f"  Mantissa bits: {r['mantissa_bits']}")
print(f"  Mantissa value: {r['mantissa_val']:.12f}")

# 实验 2: FP32 能精确表示多大的数？
# 23-bit mantissa → 约 2^23 = 8388608 个不同的值在 [1, 2) 区间
# → 两个相邻 FP32 的间距约 1.19e-7
print(f"\nFP32 的间距: {2**(-23):.2e}  (在 magnitude=1 附近)")

# 实验 3: INT8 可以表示 256 个不同的值
# 这听起来很少，但考虑一个量化场景：
# 一个激活值的范围是 [-3.0, 3.0]，用 INT8 量化
# scale = 6.0 / 255 ≈ 0.0235
# 这意味着两个相邻 INT8 值之间的距离是 0.0235
# 任何在 [-0.012, 0.012] 之间的变化 INT8 都无法分辨
# 但对于一个经过 ReLU 的激活值来说，这个精度通常够了
```

---

## 2. 量化公式：一步一步推导

### 2.1 把问题说清楚

量化要做的事情是——**找一个线性映射，把连续的实数域映射到有限的整数域**。这里面有两个核心参数：**scale（步长 S）**和**zero_point（零点 Z）**。

为什么要用"线性映射"而不是其他映射？因为线性映射在硬件上几乎零开销——就是一次乘法和一次加法。如果用非线性映射（比如对数量化），你要查表，延迟和功耗都不可预测。

### 2.2 推导

```
问题：把任意实数 r ∈ [r_min, r_max] 映射到整数 q ∈ [q_min, q_max]

第 1 步：平移到原点
  r - r_min     → 范围变为 [0, r_max - r_min]

第 2 步：缩放到 [0, 1]
  (r - r_min) / (r_max - r_min)    → 范围 [0, 1]

第 3 步：缩放到 [q_min, q_max]
  q' = (r - r_min) / (r_max - r_min) * (q_max - q_min) + q_min

第 4 步：取整
  q = round(q')

整理为标准形式：
  q = round(r / S + Z)

  其中：
    S = (r_max - r_min) / (q_max - q_min)    ← scale，量化步长
    Z = q_min - round(r_min / S)              ← zero_point

反量化：
  r ≈ S × (q - Z)
```

在代码中跑一遍具体数值有助于建立直觉：

```python
import numpy as np

# 场景：把 [-2.5, 4.7] 这个区间量化到 INT8 [-128, 127]
r_min, r_max = -2.5, 4.7
n_bits = 8
q_min, q_max = -128, 127

S = (r_max - r_min) / (q_max - q_min)
Z = q_min - round(r_min / S)
# 更精确: Z = int(round(-r_min / S + q_min))

print(f"S = ({r_max} - {r_min}) / ({q_max} - {q_min})")
print(f"  = {r_max - r_min} / {q_max - q_min}")
print(f"  = {S:.6f}")
print(f"Z = {q_min} - round({r_min} / {S:.6f}) = {Z}")

# 验证：量化范围两端映射回实数
r_low = S * (q_min - Z)
r_high = S * (q_max - Z)
print(f"量化范围 [{q_min}, {q_max}] 映射回实数: [{r_low:.4f}, {r_high:.4f}]")
print(f"原始范围: [{r_min}, {r_max}]")

# 量化一个具体值
r = 1.8
q = int(round(r / S + Z))
q_clamped = np.clip(q, q_min, q_max)
r_recovered = S * (q_clamped - Z)

print(f"\n真实值 r = {r}")
print(f"量化值 q = round({r}/{S:.6f} + {Z}) = {q}")
if q != q_clamped:
    print(f"⚠️  但 q 超出 [{q_min}, {q_max}]，被 clamp 到 {q_clamped}")
    print(f"   → Clip Error = |{r} - {r_recovered:.4f}| = {abs(r - r_recovered):.4f}")
else:
    print(f"恢复值 r̂ = {r_recovered:.6f}")
    print(f"量化误差 = |{r} - {r_recovered:.6f}| = {abs(r - r_recovered):.6f}")
```

### 2.3 量化误差的两个来源

量化误差来自两个独立的过程：

1. **Round Error（舍入误差）**：`round()` 操作丢掉的小数部分。对于 8-bit，最大舍入误差是 `S / 2`。这是不可避免的，但通常可控。

2. **Clip Error（截断误差）**：超出 `[q_min, q_max]` 的值被硬截断。这比 round error 严重得多——一个异常值被 clip 掉可能改变整层的输出。**校准算法的核心目标就是最小化 clip error**。

你作为量化工程师的核心 work 就是找到那对 `(S, Z)`，让这两种误差加起来最小。这是第 5 节的内容。

---

## 3. 对称量化 vs 非对称量化：什么场景选哪个

### 3.1 一张表把选择说清楚

| 对比维度 | 对称 (Symmetric) | 非对称 (Asymmetric) |
|----------|-----------------|-------------------|
| 公式 | `q = round(r / S)` | `q = round(r / S + Z)` |
| Zero Point | **强制为 0** | **可以不为 0** |
| 硬件开销 | 不需要减 Z → 省一个操作 | 需要减 Z → 多一次加法 |
| 适用的数据分布 | 以 0 为中心对称分布 | 有偏分布（如 ReLU 后全 ≥ 0） |
| 典型场景 | **权重**（均值 ≈ 0，近似对称） | **激活值**（ReLU 输出全 ≥ 0） |
| scale 计算 | `S = max(|r_min|, |r_max|) / 127` | `S = (r_max - r_min) / 255` |

### 3.2 可视化——这是建立直觉的关键

想象你在铺一个"尺子"来测量数据：

```
对称量化 (Symmetric) — 适合权重:
     数据的分布:     ----***|***----
                   -0.5   0   0.5

     量化尺子:    [-128 ... 0 ... 127] × S
                    ↑ 尺子以 0 为中心，两边对称
                    ↑ 所有量化值都用来覆盖 [-abs_max, abs_max]
                    利用率高，没有浪费


非对称量化 (Asymmetric) — 适合 ReLU 激活值:
     数据的分布:    ****|          (全部 ≥ 0)
                   0   2.5

     如果用对称量化:  [-128 ... 0 ... 127] × S
                      ↑ 负数部分（-128 到 0）完全被浪费！
                      ↑ 实际只用到了 [0, 127] 这一段

     非对称量化:     [0 ... 255] × S
                      ↑ 所有 256 个值都用在 [0, max]
                      ↑ 利用率 100%
```

**一个具体的浪费算例**：ReLU 之后的值全在 [0, 5.0]。对称量化会把 [-5.0, 5.0] 映射到 [-128, 127]，实际上只用了 [0, 127] 这 128 个值（50%）。非对称量化把 [0, 5.0] 映射到 [0, 255]，全部 256 个值都用上了。对于 8-bit 量化，这 50% 的浪费在精度上是致命的。

### 3.3 用代码验证

```python
import torch

activations = torch.relu(torch.randn(10000) * 2.0 + 0.5)  # ReLU 后全 ≥0
weights = torch.randn(10000) * 0.3                          # 近似对称

def quantize_symmetric(x, n_bits=8):
    q_max = 2**(n_bits-1) - 1
    abs_max = x.abs().max()
    S = abs_max / q_max
    q = torch.round(x / S).clamp(-q_max, q_max)
    return q, S, torch.tensor(0.)

def quantize_asymmetric(x, n_bits=8):
    q_max = 2**n_bits - 1
    x_min, x_max = x.min(), x.max()
    S = (x_max - x_min) / q_max
    Z = torch.round(-x_min / S).clamp(0, q_max)
    q = torch.round(x / S + Z).clamp(0, q_max)
    return q, S, Z

def mse(x, q_fn, **kw):
    q, S, Z = q_fn(x, **kw)
    x_rec = S * (q - Z)
    return ((x - x_rec) ** 2).mean().item()

print(f"权重量化: 对称={mse(weights, quantize_symmetric):.2e} | 非对称={mse(weights, quantize_asymmetric):.2e}")
print(f"激活量化: 对称={mse(activations, quantize_symmetric):.2e} | 非对称={mse(activations, quantize_asymmetric):.2e}")
# 结论：对权重，对称量化通常够好甚至更优；对 ReLU 激活，非对称显著更优
```

---

## 4. 量化粒度：一个 scale 管多大范围

### 4.1 直觉：一根 scale 管多少数据

这是在精度和存储之间做 trade-off。scale 越精细（每个小群体有自己的 scale），量化误差越小，但存储 scale 的开销越大。

```
假设一个 Conv2d 的 weight 形状是 [64, 3, 3, 3]（64 个输出通道）：

Per-Tensor:
  所有权重共用一个 (S, Z)
  存储: 2 个 float32 = 8 bytes
  问题: 通道 0 的权重范围是 [-0.1, 0.1]，通道 63 的范围是 [-0.5, 0.5]
        → 共用 scale 的话，通道 0 的量化分辨率被浪费了 5 倍

Per-Channel:
  每个输出通道有自己的 (S_i, Z_i)
  存储: 64 × 2 = 128 floats = 512 bytes
  权重本身: 64×3×3×3 = 1728 floats = 6912 bytes
  存储开销: 512/6912 = 7.4% ← 完全可接受！
  （而且对于 INT8 模型，权重本身只有 1728 bytes，scale 占 512/1728 = 30%，
   这是 INT8 量化模型大小不会恰好是 FP32 的 1/4 的原因）

Per-Group (比如 group_size=128):
  把所有权重分成 (1728/128) ≈ 14 个组
  每组有自己的 scale
  存储: 14 × 2 = 28 floats = 112 bytes
  精度: 最高，但计算复杂度也更高
```

**为什么权重量化几乎都用 Per-Channel？** 因为 per-channel 的 scale 存储开销通常不到权重的 8%，但精度提升是巨大的——尤其是对于 depthwise separable convolution 这种不同通道差异很大的结构。

### 4.2 动手验证

```python
W = torch.randn(64, 3, 3, 3)

# Per-Tensor
S_t = W.abs().max() / 127
W_q_tensor = (torch.round(W / S_t).clamp(-128, 127) * S_t).reshape_as(W)

# Per-Channel
W_c = W.reshape(64, -1)
S_c = W_c.abs().max(dim=1).values / 127
W_q_channel = (torch.round(W_c / S_c.unsqueeze(1)).clamp(-128, 127)
               * S_c.unsqueeze(1)).reshape_as(W)

mse_t = ((W - W_q_tensor)**2).mean().item()
mse_c = ((W - W_q_channel)**2).mean().item()
print(f"Per-Tensor MSE:  {mse_t:.2e}")
print(f"Per-Channel MSE: {mse_c:.2e}")
print(f"改进倍数: {mse_t/mse_c:.1f}x")
```

---

## 5. 校准算法：scale 和 zero_point 怎么算

校准是一个优化问题：找一对 `(S, Z)`，使得量化后的数据和原始数据的"差异"最小。不同的"差异"定义衍生出不同的校准器。

### 5.1 四种校准器的设计思路

| 校准器 | 度量目标 | 优点 | 缺点 |
|--------|----------|------|------|
| **MinMax** | 覆盖全部范围 | 最快、最简单 | 对 outlier 极度敏感 |
| **Percentile** | 覆盖 99.9% 的数据 | 抗 outlier | 丢弃了 0.1% 的数据点（可能包含关键信号） |
| **MSE** | 最小化均方误差 | 理论最优（MSE 意义下） | 需要网格搜索，校准慢 |
| **KL Divergence** | 最小化分布差异 | 业界标准（TensorRT 默认） | 需要构建直方图，对 bin 数量敏感 |

**核心直觉**：MinMax 把所有数据包住，不管有没有 outlier。百分之九十九点九的数据都很正常，但一个 outlier 就能把 scale 拉得非常大——导致量化网格的"分辨率"变得极差。Percentile 直接删掉 outlier，用 99.9% 分位数作为 max，让绝大部分数据的量化分辨率更好——但万一 outlier 包含关键信息（这是 LLM 中的重要问题），删掉它就坏了。MSE 和 KL 是两者的折中。

### 5.2 手写四种校准器

```python
import torch
import numpy as np

class MinMaxCalibrator:
    def __init__(self, symmetric=False, n_bits=8):
        self.symmetric = symmetric
        self.n_bits = n_bits

    def calibrate(self, x):
        q_max = 2**(self.n_bits - 1) - 1 if self.symmetric else 2**self.n_bits - 1
        if self.symmetric:
            abs_max = x.abs().max()
            self.scale = abs_max / q_max
            self.zero_point = torch.tensor(0.)
        else:
            x_min, x_max = x.min(), x.max()
            self.scale = (x_max - x_min) / q_max
            self.zero_point = torch.round(-x_min / self.scale).clamp(0, q_max)
        return self

class PercentileCalibrator:
    def __init__(self, pct_low=0.999, pct_high=0.999, n_bits=8):
        self.pct_low, self.pct_high = pct_low, pct_high
        self.n_bits = n_bits

    def calibrate(self, x):
        s = x.flatten().sort().values
        n = len(s)
        x_min = s[int(n * (1 - self.pct_low))]
        x_max = s[int(n * self.pct_high)]
        self.scale = (x_max - x_min) / (2**self.n_bits - 1)
        self.zero_point = torch.round(-x_min / self.scale).clamp(0, 2**self.n_bits - 1)
        return self

class MSECalibrator:
    def __init__(self, n_bins=100, n_bits=8, symmetric=True):
        self.n_bins, self.n_bits = n_bins, n_bits
        self.symmetric = symmetric

    def calibrate(self, x):
        q_max = 2**(self.n_bits - 1) - 1
        x_abs = x.abs()
        best_mse, best_absmax = float('inf'), x_abs.max().item()

        for absmax in torch.linspace(x_abs.max()/100, x_abs.max(), self.n_bins):
            scale = absmax / q_max
            q = torch.round(x / scale).clamp(-q_max, q_max)
            mse = ((x - q * scale) ** 2).mean().item()
            if mse < best_mse:
                best_mse, best_absmax = mse, absmax.item()

        self.scale = best_absmax / q_max
        self.zero_point = torch.tensor(0.)
        print(f"  MSE Calibrator: best_absmax={best_absmax:.4f}, MSE={best_mse:.6e}")
        return self

class KLCalibrator:
    """
    KL 校准器的核心思路：
    1. 把 FP32 激活值做成一个直方图（2048 bins）
    2. 尝试各种截断阈值，对每个阈值：
       a. 截断并重新归一化 → "参考分布" P
       b. 把截断后的 bin 压缩到 128 个量化 bin → "量化分布" Q
       c. 计算 KL(P||Q)
    3. 选 KL 散度最小的阈值 → 计算 scale
    """
    def __init__(self, n_bits=8, n_bins=2048):
        self.n_bits, self.n_bins = n_bits, n_bins

    def _smooth_distribution(self, p, eps=1e-10):
        is_zeros = (p == 0).float()
        is_nonzeros = (p != 0).float()
        n_zeros = is_zeros.sum().float()
        n_nonzeros = is_nonzeros.sum().float()
        eps1 = eps * n_zeros / n_nonzeros if n_nonzeros > 0 else eps
        p = p + is_zeros * eps1 + is_nonzeros * eps
        return p / p.sum()

    def calibrate(self, x):
        x_abs = x.abs()
        hist = torch.histc(x_abs, bins=self.n_bins, min=0, max=x_abs.max())
        hist = hist / hist.sum()
        q_max = 2 ** (self.n_bits - 1) - 1
        best_kl, best_threshold = float('inf'), x_abs.max().item()

        for i in range(self.n_bins // 2, self.n_bins):
            threshold = (i / self.n_bins) * x_abs.max().item()
            P = torch.cat([hist[:i], hist[i:].sum().unsqueeze(0)])
            P = self._smooth_distribution(P)

            # 压缩 i bins → q_max + 1 bins
            num_per_bin = i / (q_max + 1)
            Q_quant = torch.zeros(q_max + 1)
            for j in range(q_max + 1):
                start, end = int(j * num_per_bin), int((j + 1) * num_per_bin)
                Q_quant[j] = hist[:i][start:end].sum()

            # 扩展回 i bins
            Q_expand = torch.zeros(i)
            for j in range(q_max + 1):
                start, end = int(j * num_per_bin), int((j + 1) * num_per_bin)
                if end > start:
                    Q_expand[start:end] = Q_quant[j] / (end - start)
            Q = torch.cat([Q_expand, hist[i:].sum().unsqueeze(0)])
            Q = self._smooth_distribution(Q)

            kl = (P * (torch.log(P) - torch.log(Q))).sum()
            if kl < best_kl:
                best_kl, best_threshold = kl.item(), threshold

        self.scale = best_threshold / q_max
        self.zero_point = torch.tensor(0.)
        print(f"  KL Calibrator: best_threshold={best_threshold:.4f}, KL={best_kl:.6f}")
        return self
```

### 5.3 异常值场景下的对比实验

```python
# 构造含 outlier 的数据来对比四种校准器
x = torch.randn(10000) * 1.0
x[0], x[1] = 100.0, -50.0   # 注入极端 outlier

print("=== 含 outlier 场景下四种校准器的表现 ===")
for name, calib in [
    ("MinMax    ", MinMaxCalibrator(symmetric=True)),
    ("Percentile", PercentileCalibrator()),
    ("MSE       ", MSECalibrator(n_bins=100)),
    ("KL        ", KLCalibrator(n_bins=2048)),
]:
    calib.calibrate(x)
    q = torch.round(x / calib.scale).clamp(-127, 127)
    x_rec = q * calib.scale
    mse = ((x - x_rec)**2).mean().item()
    print(f"  {name}: scale={calib.scale:.4f}, MSE={mse:.4f}")

# 预期输出:
#   MinMax 被 outlier 拖垮，scale 被拉到 ~0.78，MSE 很高
#   Percentile 丢弃 outlier，MSE 最低
#   MSE / KL 介于中间
#
# ⚠️ 但注意：如果 outlier 包含关键信息（LLM 的 outlier channel），
#   Percentile 丢弃它们会导致模型性能严重下降
#   这就是 SmoothQuant / AWQ 等方法的动机——
#   不丢弃 outlier，而是用数学方法把它"平滑掉"
```

---

## 6. 舍入策略：四舍五入就够了吗

### 6.1 Round-to-Nearest (RNN) — 99% 的场景用这个

RNN 简单且无偏（从统计意义上）。对 8-bit 量化，round error 在有大量权重时倾向于互相抵消——正误差和负误差的概率相等。

### 6.2 Stochastic Rounding — 低比特的救命稻草

当比特数很低（≤3-bit），RNN 的误差变得有偏且不可忽略。Stochastic Rounding 通过概率舍入使期望值等于原始值：

```python
def stochastic_round(x):
    floor_val = torch.floor(x)
    frac = x - floor_val
    # 以 frac 为概率向上取整，以 1-frac 为概率向下取整
    rand = torch.rand_like(frac)
    return torch.where(rand < frac, floor_val + 1, floor_val)

# 实验：大量样本下的期望值对比
x = torch.tensor([2.7, 2.3, 2.5, 2.1])
rnn_results = torch.round(x).float()   # [3, 2, 3, 2]  → 均值=2.5
sr_results_avg = torch.zeros_like(x)
for _ in range(10000):
    sr_results_avg += stochastic_round(x).float() / 10000
# sr_avg ≈ [2.7, 2.3, 2.5, 2.1]  → 均值=2.4 (无偏!)

# RNN 把 2.3 永远舍入到 2（有偏）
# SR 让 2.3 在 70% 的情况下 = 2，30% 的情况下 = 3（期望 = 2.3）
```

SR 的核心价值：在大批量数据的统计意义上，量化误差的均值为零。对于推理来说，单个 token 的误差可能大，但整体分布和 FP 一致。RNN 做不到这一点——它的一致性误差会累积。

---

## 7. 为什么量化能加速：硬件层的真相

### 7.1 瓶颈是搬运，不是计算

很多人以为 INT8 比 FP32 快是因为 INT8 乘法更快。这只对了一半。**真正的瓶颈在内存带宽**。

```
推理过程（简化）：
  1. 从 HBM（显存）搬运权重到计算单元
  2. 做矩阵乘法
  3. 写回结果

step 1 的耗时通常比 step 2 长 3-5 倍。
这是一个"带宽受限"的场景。

FP32 权重: [1024, 1024] → 4 MB
INT8 权重: [1024, 1024] → 1 MB  ← 搬运量减少到 1/4
```

### 7.2 DP4A / VNNI：一条指令，4 次乘法

```
传统 INT8 向量乘加 (4对):
  for i in range(4):
      acc += a[i] * b[i]
  → 4 条乘法指令 + 3 条加法指令

DP4A / VNNI:
  _mm256_dpbusd_epi32(accum, a_8bit, b_8bit)
  → 1 条指令 = 4 次 INT8 乘 + 1 次 INT32 累加
  → 指令数减少到 1/7
```

### 7.3 Tensor Core INT8 — Ampere 架构的数据

```
NVIDIA A100:
  FP16: 312 TFLOPS
  INT8: 624 TFLOPS  → 2×  FP16 吞吐
  INT4: 1248 TFLOPS → 4×  FP16 吞吐

  每次 Tensor Core 操作: 处理 16×16×16 的 MMA (matrix-multiply-accumulate)
  → 4096 次乘法+加法，一个时钟周期
```

量化加速的三个维度：**内存带宽（2-4x） × 计算吞吐（2-4x） × 功耗效率（INT8 操作功耗远低于 FP32）**。

---

## 8. PyTorch 源码深潜：Observer 家族

> 从这一节开始，我们进入 PyTorch 源码。不只是调用 API，而是理解每一个类的继承关系、每一个 forward 的实现细节、每一个 buffer 的生命周期。

### 8.1 Observer 的继承树

```
ObserverBase  (nn.Module)
  └── _ObserverBase  (添加 eps、dtype、qscheme 等量化属性)
        └── MinMaxObserver  (直接记录 min/max)
        │     └── MovingAverageMinMaxObserver  (EMA 更新 min/max)
        │           └── MovingAveragePerChannelMinMaxObserver  (EMA + per-channel)
        │     └── PerChannelMinMaxObserver  (per-channel，无 EMA)
        └── HistogramObserver  (用直方图做 KL/MSE 校准)
```

**设计原则**：`ObserverBase` 负责"观察"（collect stats），`FakeQuantize` 继承自 Observer 并额外负责"量化"（apply quant/dequant）。这种继承关系确保了 FakeQuantize 既是一个 Observer（有 scale/Z），又是一个可训练模块（有 forward 时量化逻辑）。

### 8.2 MinMaxObserver：最简实现

```python
# ===== MinMaxObserver 的简化源码 =====
# 对应文件: torch/ao/quantization/observer.py
#
# 核心状态: min_val / max_val 两个 buffer
# 核心逻辑: forward 时更新 min/max，不改变数据

class MinMaxObserver(torch.nn.Module):
    """
    最基础的 observer: 记录输入值的绝对最小值和最大值。

    关键设计决策:
    1. 继承了 nn.Module——这样它可以作为子模块注册到模型中
    2. min_val 和 max_val 用 register_buffer——跟随设备迁移、保存到 state_dict
    3. forward 只是"看"数据不修改——返回的是原始输入
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=0, quant_max=255, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps  # 防止 scale=0 导致除零

        # register_buffer: 不是可训练参数，但会保存到 state_dict
        # 会跟随 model.to(device)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        """只观察，不修改。x_orig 直接返回。"""
        if x_orig.numel() == 0:
            return x_orig

        # detach: 不想让 min/max 的更新参与梯度计算
        x = x_orig.detach()

        if self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            self.min_val = torch.min(self.min_val, x.min())
            self.max_val = torch.max(self.max_val, x.max())

        elif self.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            # channel 是第 0 维
            x_flat = x.reshape(x.shape[0], -1)
            self.min_val = torch.min(self.min_val, x_flat.min(dim=1).values)
            self.max_val = torch.max(self.max_val, x_flat.max(dim=1).values)

        return x_orig  # ← 注意：返回原始输入

    def calculate_qparams(self):
        """根据 min/max 计算 scale 和 zero_point"""
        if self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric):
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            scale = abs_max / float(self.quant_max - self.quant_min) * 2
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.max_val - self.min_val) / float(self.quant_max - self.quant_min)
            zero_point = self.quant_min - torch.round(self.min_val / scale)
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return scale, zero_point
```

### 8.3 MovingAverageMinMaxObserver：EMA 版本

```python
class MovingAverageMinMaxObserver(MinMaxObserver):
    """
    为什么需要 EMA（指数滑动平均）？
    因为每批数据的 min/max 可能波动。直接用整个 epoch 的 abs min/max
    容易被某批异常数据污染。EMA 给最近的 batch 更大权重，
    让统计量更平滑。

    EMA: new = old + α × (observed - old)
          当 α=0.01 时，最近 100 个 batch 的权重占 ~63%
    """

    def __init__(self, averaging_constant=0.01, **kwargs):
        super().__init__(**kwargs)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach()

        if self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            # EMA 更新——比直接取 min/max 更"柔和"
            self.min_val = self.min_val + self.averaging_constant * (x.min() - self.min_val)
            self.max_val = self.max_val + self.averaging_constant * (x.max() - self.max_val)
        # ... per-channel 逻辑类似

        return x_orig
```

### 8.4 HistogramObserver：TensorRT 的默认选择

```python
class HistogramObserver(MinMaxObserver):
    """
    不用 min/max，而是构建一个直方图，然后做 KL Divergence 或 MSE 搜索
    找到最优截断点。

    为什么？因为 min/max 对 outlier 太敏感了。直方图能平衡"覆盖范围"
    （少截断 outlier）和"分辨率"（scale 尽可能小）。
    """

    def __init__(self, bins=2048, upsample_rate=128, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.upsample_rate = upsample_rate
        self.register_buffer("histogram", torch.zeros(self.bins))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().abs()

        # 合并新旧直方图
        # 用 min/max 跟踪总范围
        super().forward(x)

        # 构建新的直方图并和旧的合并
        new_hist = torch.histc(x, bins=self.bins,
                               min=self.min_val.item(),
                               max=self.max_val.item())
        self.histogram = self.histogram + new_hist

        return x_orig

    def calculate_qparams(self):
        """使用 _non_linear_param_search 做 KL 或 MSE 优化"""
        # _non_linear_param_search:
        #   1. 把 self.histogram 归一化
        #   2. 遍历可能的截断点，每个点计算 KL(P||Q) 或 MSE
        #   3. 返回最优的 min/max 范围
        new_min, new_max = _non_linear_param_search(
            self.histogram, self.bins, self.quant_min, self.quant_max,
            self.dtype, self.qscheme,
        )
        # 用 optimal min/max 计算 scale/Z（和父类一样）
        self.min_val, self.max_val = new_min, new_max
        return super().calculate_qparams()
```

---

## 9. PyTorch 源码深潜：FakeQuantize 内核

### 9.1 FakeQuantize 的继承关系

```
ObserverBase → _ObserverBase → MinMaxObserver → ...
                                              ↘
FakeQuantizeBase (nn.Module)                    ↘
  └── FakeQuantize  ←── 同时继承了 Observer 的统计能力和
                          FakeQuantizeBase 的量化能力
```

**FakeQuantize 是多继承的精妙设计**。它继承自 Observer（有 min/max/scale/Z 的收集和计算能力），同时也继承了 FakeQuantizeBase（有 enable/disable 状态机制）。这样它就是一个完整的"观察→量化→反量化"单元。

### 9.2 FakeQuantizeBase：状态管理

```python
# ===== FakeQuantizeBase 的简化源码 =====
# 位置: torch/ao/quantization/fake_quantize.py

class FakeQuantizeBase(torch.nn.Module):
    """
    FakeQuantize 的基类。
    定义了 observer_enabled / fake_quant_enabled 两个核心开关。

    为什么用 uint8 tensor 而不是 bool？
    → TorchScript 兼容性。Bool tensor 在 torchscript 中的行为可能不一致。
    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self):
        super().__init__()
        # 两个独立 flag——这是整个 QAT 状态机的基础
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))

    def enable_observer(self):
        self.observer_enabled[0] = 1

    def disable_observer(self):
        self.observer_enabled[0] = 0

    def enable_fake_quant(self):
        self.fake_quant_enabled[0] = 1

    def disable_fake_quant(self):
        self.fake_quant_enabled[0] = 0
```

### 9.3 FakeQuantize：完整内核

```python
class FakeQuantize(FakeQuantizeBase, MovingAverageMinMaxObserver):
    """
    完整的内核：Observer 的统计能力 + FakeQuantizeBase 的状态管理

    forward 的逻辑：
      if observer_enabled:  更新 min/max → 重新计算 scale/Z
      if fake_quant_enabled: 量化 → clamp → 反量化 → 返回 float

    STE（Straight-Through Estimator）在这里体现：
    torch.round() 的前向做真正的舍入（输出是离散值），
    但 PyTorch 对 round() 的反向使用 STE——梯度直接穿透。
    这意味着模型能通过梯度下降来"学习"补偿量化误差。
    """

    def __init__(self, observer=MovingAverageMinMaxObserver,
                 quant_min=-128, quant_max=127, **observer_kwargs):
        FakeQuantizeBase.__init__(self)
        MovingAverageMinMaxObserver.__init__(self, quant_min=quant_min,
                                             quant_max=quant_max, **observer_kwargs)

    def forward(self, X):
        # 阶段 1: 观察（更新 scale / zero_point）
        if self.observer_enabled[0] == 1:
            # self.activation_post_process 是指向自己的引用——
            # 因为在很多场景中，FakeQuantize 被赋值给
            # module.activation_post_process，所以它需要在自己上做 observer 更新
            self.activation_post_process(X)

        # 阶段 2: 量化-反量化（FakeQuantize）
        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.calculate_qparams()

            if zero_point is not None:
                zero_point = zero_point.to(torch.int32)

            # ★ 核心操作用于 ONNX 导出和 TorchScript 兼容的 C++ 实现
            # torch.fake_quantize_per_tensor_affine 在 cpp 层有高效实现
            if self.qscheme == torch.per_tensor_affine:
                X = torch.fake_quantize_per_tensor_affine(
                    X, scale.item(), int(zero_point.item()),
                    self.quant_min, self.quant_max)
            elif self.qscheme == torch.per_tensor_symmetric:
                X = torch.fake_quantize_per_tensor_affine(
                    X, scale.item(), 0, self.quant_min, self.quant_max)
            elif self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch.fake_quantize_per_channel_affine(
                    X, scale, zero_point, self.quant_min, self.quant_max)

        return X
```

**关于 `torch.fake_quantize_per_tensor_affine`**：这是 PyTorch 的 C++ 扩展函数（在 `torch/_C` 中定义）。它做了三件事：量化（round + clamp）、反量化（转回 float）、STE 梯度处理。和纯 Python 实现不同，它针对每一种量化模式（per-tensor affine, per-tensor symmetric, per-channel affine）有独立的优化。

---

## 10. PyTorch 源码深潜：QuantStub / DeQuantStub 到底是什么

> 这一节回答一个常见困惑：为什么 PyTorch 量化模型的开头和结尾总能看到 `QuantStub()` 和 `DeQuantStub()`？它们是做什么的？prepare_qat 和 convert 如何使用它们？

### 10.1 Stubs 的本质：占位符

`QuantStub` 和 `DeQuantStub` 的源码（`torch/ao/quantization/stubs.py`）极其简单，因为它们本身**不做任何计算**——它们只是在模块树中占据一个位置，告诉 prepare_qat 和 convert "在这里插入量化/反量化节点"。

```python
# ===== QuantStub 和 DeQuantStub 的源码（完整） =====
# 位置: torch/ao/quantization/stubs.py
import torch.nn as nn

class QuantStub(nn.Module):
    """
    标记"量化开始"位置的占位符。

    在量化过程中，它的生命周期:
    1. 用户创建: QuantStub() → 只是一个 nn.Module，forward 是 identity
    2. prepare_qat(): QuantStub 被替换为一个 Quantize 操作（实际上是一个
       observer/FakeQuantize 节点）
    3. convert(): observer 的信息被用来生成真正的 Quantize 节点
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x  # 完全直通，不做任何操作


class DeQuantStub(nn.Module):
    """
    标记"反量化"位置的占位符。

    生命周期:
    1. 用户创建: DeQuantStub() → identity
    2. prepare_qat(): 同上
    3. convert(): 生成真正的 DeQuantize 节点
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

### 10.2 为什么需要 Stubs？

考虑一个典型的量化模型：

```python
class QuantizableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()       # 标记：这是量化入口
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.relu2 = nn.ReLU()
        self.dequant = DeQuantStub()   # 标记：这是量化出口
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.quant(x)        # ← 这里之前: FP32 输入，之后: FakeQuantize
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dequant(x)      # ← 这里之前是量化的，之后回到 FP32
        x = self.fc(x)           # 分类头用 FP32（精度敏感）
        return x
```

没有 stubs 的话，框架怎么知道"入口在哪里"和"出口在哪里"？它不能假设第一个 op 是量化入口——也许第一层想保持 FP32 精度。Stubs 是用户显式告诉框架"从这里开始量化、到这里结束量化"的机制。

### 10.3 prepare_qat 如何消费 Stubs

```
prepare_qat() 遍历模型时遇到 QuantStub:
  1. 读取 QuantStub 所在模块的 qconfig
  2. 创建一个 activation_post_process (FakeQuantize) 实例
  3. 用这个 FakeQuantize 替换 QuantStub 模块
     → QuantStub() 被替换为 FakeQuantize()

prepare_qat() 遍历模型时遇到 DeQuantStub:
  1. 同上
  2. DeQuantStub() 被替换为 FakeQuantize()
  （在 QAT 阶段，QuantStub 和 DeQuantStub 都被替换为 FakeQuantize——
   因为 QAT 阶段不需要区分 Q/DQ，两者都是 FakeQuantize）

convert() 遍历模型时:
  QuantStub → 读取其训练后的 scale/Z → 生成真正的 Quantize 节点
  DeQuantStub → 读取其训练后的 scale/Z → 生成真正的 DeQuantize 节点
```

**所以 Stubs 的生命周期是**：
```
用户创建: QuantStub() (identity)
    ↓ prepare_qat
QAT 阶段: FakeQuantize() (observer + fake quant)
    ↓ convert
INT8 模型: Quantize 节点 (真正的量化)
```

---

## 11. PyTorch 源码深潜：fuse_modules 内部全流程

> 这是 Stage 0 最硬的一节。`fuse_modules` 看起来只是一个简单函数，但它的内部机制涉及模块树遍历、钩子迁移、数学折叠、和 QAT/PTQ 双路径。

### 11.1 fuse_modules 做了什么（一句话版）

**把 `['conv1', 'bn1', 'relu1']` 变成 `[ConvBnReLU2d, Identity, Identity]`**，其中 `ConvBnReLU2d` 内部是融合后的 Conv+BN+ReLU，原来的 `bn1` 和 `relu1` 被替换为 `nn.Identity()`。

### 11.2 完整流程

```
fuse_modules(model, [['conv1', 'bn1', 'relu1']], inplace=False)
    │
    ├── Step 0: inplace=False → model = copy.deepcopy(model)
    │
    ├── Step 1: 解析路径
    │     _get_module(model, 'conv1')  → Conv2d 对象
    │     _get_module(model, 'bn1')    → BatchNorm2d 对象
    │     _get_module(model, 'relu1')  → ReLU 对象
    │
    ├── Step 2: 查找融合函数
    │     types = (Conv2d, BatchNorm2d, ReLU)
    │     在 DEFAULT_OP_LIST_TO_FUSER_METHOD 字典中查找
    │     → 找到: fuse_conv_bn_relu
    │
    ├── Step 3: 执行融合
    │     如果 is_qat=True:
    │       返回 nniqat.ConvBnReLU2d(conv, bn, relu)
    │         — QAT 模块，内部包含 BN 的实时统计 + FakeQuantize 节点
    │     如果 is_qat=False (PTQ):
    │       a. fuse_conv_bn_eval(conv, bn) → 数学折叠 BN 到 Conv 权重中
    │          W' = W × γ / √(σ² + ε)
    │          b' = (b - μ) × γ / √(σ² + ε) + β
    │          返回一个普通的 Conv2d（不再是 Conv+BN）
    │       b. nni.ConvReLU2d(folded_conv, relu) → 融合的 Conv+ReLU 模块
    │
    ├── Step 4: 迁移 hooks
    │     把原始 conv1 的 pre_forward_hooks → fused module
    │     把原始 relu1 的 forward_hooks → fused module
    │     （bn1 的 hooks 被丢弃——因为 BN 的 forward 不再执行了）
    │
    ├── Step 5: 替换模块
    │     _set_module(model, 'conv1', fused_module)  → ConvBnReLU2d
    │     _set_module(model, 'bn1',   nn.Identity())  → 直通
    │     _set_module(model, 'relu1', nn.Identity())  → 直通
    │
    │     ⚠️ 注意：bn1 和 relu1 模块仍然存在于 model 中！
    │     它们是 nn.Identity()，forward 时不做任何操作。
    │     这是为了保持模型结构不变——如果有其他代码引用 model.bn1，
    │     它不会报错，只是失去了原来的功能。
```

### 11.3 fuse_conv_bn_eval 的数学折叠

```python
# ===== 关键：PTQ 路径下 BN 被折叠进 Conv =====
# 位置: torch/nn/utils/fusion.py

def fuse_conv_bn_eval(conv, bn):
    """
    在 PTQ 模式下，BN 的权重和偏置被"折入" Conv 的权重和偏置。
    折叠后不再需要一个独立的 BN 模块——因为计算已经合并。
    这是一个精确的数学等价变换，不引入任何近似（在 FP32 精度下）。
    """
    # γ / √(σ² + ε)
    gamma_over_std = bn.weight / torch.sqrt(bn.running_var + bn.eps)

    # 融合权重: W' = W * γ / √(σ² + ε)
    fused_weight = conv.weight * gamma_over_std.view(-1, 1, 1, 1)

    # 融合偏置: b' = (b - μ) * γ / √(σ² + ε) + β
    if conv.bias is not None:
        fused_bias = (conv.bias - bn.running_mean) * gamma_over_std + bn.bias
    else:
        fused_bias = bn.bias - bn.running_mean * gamma_over_std

    return torch.nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
        bias=True, padding_mode=conv.padding_mode,
    ).to(fused_weight.device)  # 返回一个新 Conv，权重和偏置已融合
```

### 11.4 支持的融合模式完整列表

```python
# ===== 融合模式映射表 =====
# 位置: torch/ao/quantization/fuser_method_mappings.py

_DEFAULT_OP_LIST_TO_FUSER_METHOD = {
    (nn.Conv1d, nn.BatchNorm1d):             fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU):    fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d):             fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU):    fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d):             fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU):    fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU):                    fuse_conv_relu,
    (nn.Conv2d, nn.ReLU):                    fuse_conv_relu,
    (nn.Conv3d, nn.ReLU):                    fuse_conv_relu,
    (nn.Linear, nn.ReLU):                    fuse_linear_relu,
    (nn.Linear, nn.BatchNorm1d):             fuse_linear_bn,
    (nn.BatchNorm2d, nn.ReLU):               fuse_bn_relu,
    (nn.BatchNorm3d, nn.ReLU):               fuse_bn_relu,
}
# 每个 (pattern) → fuser_method 的映射
```

---

## 12. PyTorch 源码深潜：prepare_qat 和 convert 如何消费这些模块

至此，你已经理解了 Observer、FakeQuantize、QuantStub、fuse_modules 各自是什么。现在把它们串起来——**prepare_qat 和 convert 是如何使用这些模块来完成 QAT 管线的**。

### 12.1 prepare_qat 的完整流程

```
prepare_qat(model)
    │
    ├── 遍历 model 的所有子模块 (model.named_modules())
    │
    ├── 对每个有 .qconfig 属性的模块 m:
    │   │
    │   ├── 如果 m 是 QuantStub / DeQuantStub:
    │   │     → 替换为一个 FakeQuantize 实例 (m.qconfig.activation())
    │   │
    │   ├── 如果 m 是融合后的模块 (如 ConvBnReLU2d):
    │   │     → 模块内部已经包含了 weight_fake_quant 和 activation_post_process
    │   │     → 不需要额外插入 FakeQuantize
    │   │
    │   ├── 如果是普通的 Conv2d / Linear:
    │   │     → 创建一个 weight_fake_quant = m.qconfig.weight()
    │   │     → 创建一个 activation_post_process = m.qconfig.activation()
    │   │     → 附加到模块 m 上
    │   │     → 修改 m 的 forward 方法（wrapping），在计算前后插入量化操作
    │   │
    │   └── 如果 m.qconfig 是 None:
    │         → 跳过，不量化这个模块
    │
    └── 返回 model_prepared（现在可以在 FakeQuantize 下训练了）

关键点: prepare_qat 不修改计算图（Eager Mode），它修改的是 forward 方法。
       这就是 FakeQuantize 在 Eager 模式下"不可见"的原因——
       它在模块内部，不是独立的图节点。
```

### 12.2 convert 的完整流程

```
convert(model_prepared)
    │
    ├── 遍历模型的所有子模块
    │
    ├── 对每个模块 m:
    │   │
    │   ├── 如果 m 是 FakeQuantize（原 QuantStub/DeQuantStub 变来的）:
    │   │     读取 m.scale 和 m.zero_point
    │   │     → QuantStub → 替换为 torch.nnq.Quantize(scale, zero_point)
    │   │     → DeQuantStub → 替换为 torch.nnq.DeQuantize(scale, zero_point)
    │   │
    │   ├── 如果 m 是融合后的 QAT 模块（如 nniqat.ConvBnReLU2d）:
    │   │     → 从模块内部读取 weight_fake_quant.scale 和 act_fake_quant.scale
    │   │     → 替换为量化版模块（如 torch.nnq.ConvReLU2d）
    │   │
    │   ├── 如果 m 是 Conv2d + 有 weight_fake_quant:
    │   │     读取 weight_fake_quant.scale → 创建 torch.nnq.Conv2d(scale=...)
    │   │
    │   └── 移除所有不再需要的 observer / fake_quant 模块
    │
    └── 返回: 一个"真正的" INT8 推理模型
              所有量化参数 (scale/zero_point) 现在是固定常量
              所有计算在被支持的 backend 上是真正的 INT8
```

### 12.3 数据流的完整生命周期

```
FP32 输入
    │
    │  QuantStub (在 prepare_qat 中被替换为 FakeQuantize)
    │  ┌─ observer 观察输入范围 → 计算 scale/Z
    │  └─ fake_quant: 量化输入 → clamp → 反量化回 float
    │
    ▼
  [融合的 QAT Conv 模块]
    │  ┌─ weight_fake_quant: 量化 weight
    │  │  Conv2d(quantized_weight, quantized_input)  ← 还是 FP32 卷积！
    │  └─ activation_post_process: 量化输出
    │
    ▼
  ... 更多层 ...
    │
    │  DeQuantStub (在 prepare_qat 中被替换为 FakeQuantize)
    │  └─ fake_quant: 最后一次量化 → 确保输出范围一致
    │
    ▼
FP32 输出 (但值已经被量化过，落在离散网格上)
```

---

## 13. 动手实验

### 实验 1：手写完整量化器 + MNIST 推理（30 分钟）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 目标：从零实现一个 QuantizedLinear，比较 FP32 vs INT8 推理精度
class QuantizedLinear(nn.Module):
    def __init__(self, fp_linear: nn.Linear, n_bits=8):
        super().__init__()
        # 量化 weight
        S_w = fp_linear.weight.data.abs().max() / (2**(n_bits-1) - 1)
        q_w = torch.round(fp_linear.weight.data / S_w).clamp(-128, 127).to(torch.int8)

        self.register_buffer("S_w", S_w)
        self.register_buffer("q_w", q_w)
        self.register_buffer("bias",
            fp_linear.bias.data.clone() if fp_linear.bias is not None else None)
        self.S_a = None

    def calibrate(self, x_sample):
        # 跑一批数据，收集激活值的范围
        with torch.no_grad():
            out = F.linear(x_sample, self.q_w.float() * self.S_w, self.bias)
            self.S_a = out.abs().max() / 127.0

    def forward(self, x):
        # 量化激活 → INT8 矩阵乘法 → 反量化
        q_a = torch.round(x / self.S_a).clamp(-128, 127).to(torch.int8)
        q_out = F.linear(q_a.float(), self.q_w.float(), None)
        out = q_out * (self.S_a * self.S_w)
        if self.bias is not None:
            out = out + self.bias
        return out

# 加载 MNIST，训练一个 2-layer MLP，然后用 QuantizedLinear 替换每一层，
# 比较 FP32 和 INT8 的 test accuracy
```

### 实验 2：四种校准器消融实验（20 分钟）

```python
# 在同一个模型上，分别用 MinMax / Percentile / MSE / KL 做 PTQ
# 记录 Top-1 Accuracy 和 SNR，画对比表

# 期望结果：
# ┌──────────────┬──────────┬──────────┐
# │ Calibrator   │ Top-1    │ Δ (vs FP) │
# ├──────────────┼──────────┼──────────┤
# │ FP32 base    │ 93.5%    │    —     │
# │ MinMax       │ 88.2%    │  -5.3%   │
# │ Percentile   │ 92.1%    │  -1.4%   │
# │ MSE          │ 92.3%    │  -1.2%   │
# │ KL           │ 92.4%    │  -1.1%   │
# └──────────────┴──────────┴──────────┘
```

### 实验 3：逐层量化误差溯源（20 分钟）

```python
# 注册 forward hooks 到每一层，计算每层的输出在 8-bit 量化后的相对误差
# 找出量化误差最大的前 5 层
# 这 5 层就是 QAT 训练时需要重点关注的对象
```

---

## 检验标准

- [ ] **能徒手画** FP32 / FP16 / BF16 / INT8 的位布局图
- [ ] **能手推** FP32 → INT8 的量化/反量化公式，包括 scale 和 zero_point
- [ ] **能手写** 四种校准器的 Python 实现，并说出各自的适用场景和局限
- [ ] **能画出** 对称量化对 ReLU 激活值的"浪费"示意图，解释为什么非对称是必要的
- [ ] **能说出** per-tensor vs per-channel vs per-group 的存储开销和精度 trade-off
- [ ] **能解释** VNNI / DP4A / Tensor Core INT 的加速原理（不只是"更快"，而是"为什么更快"）
- [ ] **能画** PyTorch 量化类的继承树：ObserverBase → MinMaxObserver → MovingAverage → PerChannel → Histogram → FakeQuantize
- [ ] **能说清楚** QuantStub / DeQuantStub 的生命周期：用户创建 → prepare_qat → convert 各阶段它们是什么
- [ ] **能画出** fuse_modules 的完整流程图：路径解析 → 查找融合函数 → 数学折叠 / 封装 → hook 迁移 → 模块替换
- [ ] **能解释** prepare_qat 和 convert 分别对 Observer、FakeQuantize、Stubs、融合模块做了什么操作
- [ ] **能在** 一个 MLP 上完成从零开始的量化推理，并对比 FP32 和 INT8 的精度差异

---

> 💡 **学习建议**：Stage 0 的前七节建立"量化能工作"的直觉，后五节（PyTorch 源码深潜）是让你从 API 调用者变成能看懂框架源码的工程师。建议第一遍通读建立全局认知，第二遍跟着每个代码块在自己的环境里跑一遍，第三遍尝试不看文档默写关键类的继承关系和 forward 逻辑。
>
> Next: [Stage 1: PyTorch 原生 QAT 三种模式](./Stage1_PyTorch原生QAT三种模式.md)
