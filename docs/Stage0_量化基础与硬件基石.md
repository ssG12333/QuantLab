# Stage 0: 量化基础与硬件基石

> ⏱ 预计学习时间：10-15 小时 | 🎯 难度：⭐
>
> **目标**：从零建立量化直觉——知道浮点数怎么变成整数、为什么 8-bit 能替代 32-bit、量化误差从哪来、硬件上 INT8 矩阵乘法怎么跑的。**这个 Stage 不碰一行 PyTorch 源码**——Observer、FakeQuantize、QuantStub 等框架机制全部留给 Stage 1。这里只做一件事：让你从"浮点数"走到"量化后的整数推理"，每一步都亲手算过。

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
9. [7. 为什么量化能加速：硬件层的真相](#7-硬件层)
10. [8. 从零手写一个完整的 INT8 推理引擎](#8-从零手写)
11. [9. 动手实验](#9-动手实验)
12. [检验标准](#检验标准)

---

## 开篇：为什么深度学习量化能工作

在进入任何公式之前，先回答一个根本性问题：**一个 FP32 权重被四舍五入到 INT8 之后，模型为什么还能用？** FP32 可以表示约 42 亿个不同值，INT8 只有 256 个。从 42 亿降到 256——差了 7 个数量级。

答案有三个层次。

**第一层——冗余。** 一个 ResNet50 有 2500 万个 FP32 参数，但其中绝大多数参数对最终预测的贡献微乎其微。你用 256 个值去近似 42 亿个可能值——99% 的参数精度是"过剩"的。深度学习的常规操作就是用过度参数化换取训练稳定，再压缩换取推理效率。

**第二层——噪声免疫。** 网络在训练中已经经历过各种噪声：Dropout 随机丢弃 50% 的神经元、数据增强加随机扰动、BatchNorm 用 mini-batch 统计量近似总体分布。量化误差不过是另一种噪声——网络天然有容错能力。

**第三层——训练补偿（QAT 的关键）。** 如果你在训练中注入量化噪声，模型会把权重"推"到恰好落在量化网格上。不做训练的 PTQ 只能"忍受"误差，QAT 能"主动避开"误差。这就是 Stage 1.5 和 Stage 2 要讲的内容。

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

[7] 硬件加速原理 (内存带宽 / VNNI / DP4A / Tensor Core)

[8] 手写完整 INT8 推理引擎 → MNIST 验证
```

---

## 1. 浮点数在内存中长什么样

> 学这节时，把自己想象成一个正在设计量化芯片的硬件工程师。你要把 FP32 乘法器替换成 INT8 乘法器——在做这件事之前，必须精确知道"FP32 是什么"和"它浪费了什么"。

### 1.1 画三张位布局图（闭眼能默写）

FP32、FP16、BF16 是三个不同的 trade-off 点：

```
FP32 (Single Precision):  1 + 8 + 23 = 32 bits
┌─────┬──────────┬─────────────────────────────┐
│ sign│ exponent │         mantissa             │
│1 bit│  8 bits  │         23 bits              │
└─────┴──────────┴─────────────────────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
  范围: [~1.18e-38, ~3.40e38], 精度: ~7 位有效十进制

FP16 (Half Precision):  1 + 5 + 10 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  5 bits  │   10 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-15) × (1.mantissa)
  范围: [5.96e-8, 65504] ← exponent 只有 5 bits, 容易溢出/下溢

BF16 (Brain Float 16):  1 + 8 + 7 = 16 bits
┌─────┬──────────┬───────────────┐
│ sign│ exponent │   mantissa    │
│1 bit│  8 bits  │    7 bits     │
└─────┴──────────┴───────────────┘
  值 = (-1)^sign × 2^(exponent-127) × (1.mantissa)
  范围同 FP32（exponent 同宽！）, BF16↔FP32 互转只需截断低 16 位
```

**核心洞察**：
- exponent 的宽度决定了**动态范围**，mantissa 的宽度决定了**精度**
- BF16：宽 exponent，窄 mantissa——牺牲精度换范围，适合训练
- FP16：窄 exponent，中等 mantissa——牺牲范围换精度，适合推理
- **深度学习不需要 7 位有效数字**——3-4 位就够了。这就是量化能工作的根因

### 1.2 动手验证

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
        if b == '1': mantissa_val += 2**(-(i+1))
    return {
        'bits': bits, 'sign': sign,
        'exponent': exponent_raw - 127,
        'mantissa': mantissa_val,
        'value': (-1)**sign * mantissa_val * 2**(exponent_raw - 127),
    }

# 看 3.14159 的 FP32 内部
r = analyze_fp32(3.1415926)
print(f"值: {r['value']}")
print(f"32-bit: {r['bits']}")
print(f"Sign: {r['sign']} | Exp: {r['exponent']} | Mantissa: {r['mantissa']:.12f}")
print(f"相邻 FP32 间距 (magnitude≈1): {2**(-23):.2e}")
print(f"INT8 只有 FP32 的 {256/2**32*100:.8f}% 表示能力——但配合 scale 就够了")
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

不是所有误差都平等。量化误差来自两个独立过程：

1. **Round Error（舍入误差）**：`round()` 丢掉的零头。8-bit 下最大 `S/2 ≈ 0.014`。在大批量数据中正负误差倾向抵消，通常可控。

2. **Clip Error（截断误差）**：超出量程的值被硬夹住。这比 round error 严重得多——一个 outlier 被 clip 可能改变整层输出的统计特性。**校准算法的核心目标就是最小化 clip error。**

---

## 3. 对称量化 vs 非对称量化

### 3.1 选择的核心逻辑

| | 对称 (Symmetric) | 非对称 (Asymmetric) |
|---|-----------------|-------------------|
| 公式 | `q = round(r / S)` | `q = round(r / S + Z)` |
| Zero Point | **强制为 0** | **可以不为 0** |
| 典型场景 | **权重**（以 0 为中心对称） | **ReLU 激活**（全 ≥ 0，偏态） |
| Scale | `S = max(|min|, |max|) / 127` | `S = (max - min) / 255` |

### 3.2 可视化——理解的关键

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

一个具体例子：ReLU 后激活全在 [0, 3.0]。对称量化范用 [-3, 3]→[-128,127]，只用了 [0,127]（50% 浪费）。非对称把 [0,3]→[0,255]，全部用上。

### 3.3 代码验证

```python
import torch

acts = torch.relu(torch.randn(10000) * 2.0 + 0.5)   # ReLU 后全 ≥0
wts  = torch.randn(10000) * 0.3                       # 近似对称

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

假设 Conv2d weight 为 `[64, 3, 3, 3]`：

```
Per-Tensor:     所有权重共 1 个 (S,Z) → 存储 8B
                问题: 通道间范围差 64×，共用 scale 浪费精度

Per-Channel:    64 个 (S,Z)，每输出通道一个 → 存储 512B
                权重本身 6912B，scale 占 7.4%
                ★ 权重量化的默认选择

Per-Group(128): 每 128 元素一组 → 存储更多 scale，精度最高
                用于 LLM (GPTQ/AWQ 的 group_size 参数)
```

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
print(f"Per-Tensor MSE: {mse_t:.2e}  |  Per-Channel: {mse_c:.2e}  "
      f"|  改进 {mse_t/mse_c:.1f}x")
# 通常好 5-50x
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

        def smooth(p, eps=1e-10):
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

---

## 6. 舍入策略：四舍五入就够了吗

**Round-to-Nearest (RNN)**：99% 的场景用这个。大批量下正负误差倾向抵消。

**Stochastic Rounding**：低比特（≤3-bit）的救命稻草。用概率舍入保证期望值等于原始值：

```python
def stochastic_round(x):
    f = torch.floor(x)
    r = torch.rand_like(x)
    return torch.where(r < x - f, f + 1, f)

x = torch.tensor([2.7, 2.3, 2.5, 2.1])
rnn = torch.round(x).float()  # [3,2,3,2] 均值=2.5 ≠ 2.4
sr = sum(stochastic_round(x) for _ in range(10000))/10000  # ≈ [2.7,2.3,2.5,2.1] 无偏
```

---

## 7. 为什么量化能加速：硬件层的真相

加速不主要来自"INT8 乘法本身比 FP32 快"。真相是：**瓶颈是内存带宽，不是计算吞吐。**

```
推理流程: HBM → 计算单元(搬运权重) → 矩阵乘法 → 写回
                 ↑ 这一步通常比计算慢 3-5×（"带宽受限"）

FP32 权重: [1024,1024] → 4MB 搬运量
INT8 权重: [1024,1024] → 1MB 搬运量 → 4× 加速
```

**VNNI / DP4A 指令**：一条指令 `_mm256_dpbusd_epi32` 完成 4 次 INT8 乘 + INT32 累加。指令数降到 1/7。

**Tensor Core INT8** (A100)：FP16 312 TFLOPS → INT8 **624** TFLOPS（2× 吞吐）。

加速的三维度：**内存带宽（2-4x） × 指令效率（~7x） × 计算吞吐（2x）**。

---

## 8. 从零手写一个完整的 INT8 推理引擎

把 1-7 节串起来——手写一个量化 MLP，在 MNIST 上验证。

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
        self.S_w = w.abs().max() / (2**(n_bits-1) - 1)
        self.register_buffer("q_w", torch.round(w/self.S_w).clamp(-128,127).to(torch.int8))
        self.register_buffer("bias", fp_linear.bias.data.clone()
                             if fp_linear.bias is not None else torch.zeros(fp_linear.out_features))
        self.S_a = None

    def calibrate(self, x_sample):
        with torch.no_grad():
            out = F.linear(x_sample, self.q_w.float() * self.S_w, self.bias)
            self.S_a = out.abs().max() / 127.0

    def forward(self, x):
        q_a = torch.round(x / self.S_a).clamp(-128, 127).to(torch.int8)
        return F.linear(q_a.float(), self.q_w.float(), None) * (self.S_a * self.S_w) + self.bias

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
print(f"FP32: {fp_acc:.2f}%  →  INT8: {int8_acc:.2f}%  →  Δ = {fp_acc-int8_acc:.2f}%")
# 期望: Δ < 0.5%
```

---

## 9. 动手实验

| # | 实验 | 时间 | 产出 |
|---|------|:--:|------|
| 1 | 用 `QuantizedMLP` 完成 MNIST 量化推理 | 30min | 第一个从零完成的量化项目 |
| 2 | 四种校准器在 MLP 上的消融对比 | 20min | 精度差异表，理解哪种最好、为什么 |
| 3 | 计算 ResNet18 FP32 vs INT8 的显存搬运量 | 10min | 直观感受"瓶颈是内存带宽" |

---

## 检验标准

- [ ] 能徒手画 FP32 / FP16 / BF16 / INT8 的位布局图
- [ ] 能手推 `S = (rmax-rmin)/(qmax-qmin), Z = qmin - round(rmin/S)`
- [ ] 能手写 MinMax / Percentile / MSE / KL 四种校准器的 Python 实现
- [ ] 能解释为什么权重用对称、ReLU 激活用非对称
- [ ] 能说出 VNNI / DP4A / Tensor Core 的加速原理
- [ ] 能从零手写 QuantizedLinear + calibrate → 在 MNIST 上推理

---

> 💡 **学习建议**：这个 Stage 的核心目标是"形成量化直觉"。不碰任何 PyTorch 量化 API（Observer、FakeQuantize 等全部留给 Stage 1）。只做数学和代码的互相验证——公式旁有可运行的 Python，看完公式马上跑代码。这样进入 Stage 1 时，你已经有完整的底层认知，不会被 API 淹没。
>
> Next: [Stage 1: PyTorch 量化全景 — API + 源码深潜](./Stage1_PyTorch原生QAT三种模式.md)
