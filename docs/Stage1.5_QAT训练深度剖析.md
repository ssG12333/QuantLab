# Stage 1.5: QAT 训练深度剖析 — 从"能跑"到"理解为什么"

> ⏱ 预计学习时间：12-18 小时 | 🎯 难度：⭐⭐⭐
>
> Stage 1 教你"怎么调 QAT API"。这里补上缺失的一环：**QAT 训练本身到底发生了什么。**
>
> 这是一个"拆开黑箱"的 Stage——手写 FakeQuantize、Observer 消融实验、系统性比特测试、
> BN 冻结的深层原因、失败案例诊断。最后你会发现"固定 scale QAT 在低比特下有硬上限"，
> 从而自然地引出 Stage 2 的 LSQ。

---

## 目录

1. [开篇：从"调用者"到"理解者"](#开篇)
2. [1. 手写 QAT：不用 torch.ao 跑通完整训练](#1-手写-qat)
3. [2. Observer 消融：四种 Observer 的正面对决](#2-observer-消融)
4. [3. 比特宽度系统性实验：找到"崩溃点"](#3-比特宽度系统性实验)
5. [4. BN 冻结的深层原理](#4-bn-冻结的深层原理)
6. [5. QAT 失败案例诊断手册](#5-qat-失败案例诊断手册)
7. [6. 固定 scale QAT 的能力边界 → LSQ 的动机](#6-固定-scale-qat-的能力边界)
8. [7. 动手实验](#7-动手实验)
9. [检验标准](#检验标准)

---

## 开篇：从"调用者"到"理解者"

Stage 1 结束时你能用 `prepare_qat_fx` 跑通 QAT 训练。恭喜——你现在是个合格的"API 调用者"了。

但如果我问你：

- **"为什么 Observer 要开启 2-3 个 epoch 而不是 1 个或 5 个？"**
- **"如果我把 HistogramObserver 换成 MinMaxObserver，精度会怎样？"**
- **"为什么 8-bit QAT 几乎不掉精度，4-bit 开始掉，2-bit 直接崩？"**
- **"BN 冻结到底是必须的还是可选的？如果不冻结，到底会发生什么？"**

你能**不查文档、根据第一性原理解释**吗？

**"调用者"和"理解者"的区别**：调用者知道 API 的参数名；理解者知道每个参数改变时，反向传播的哪条计算链被影响了。

这个 Stage 的目标：**把 QAT 训练拆成零件，逐个实验验证，让你对上述问题都有基于数据的答案。** 所有实验都在 MNIST/CIFAR-10 上做（数据量小，跑得快），但结论适用于一切模型，包括 LLM。

### 这个 Stage 的学习策略

```
[1] 手写 QAT ──→ 理解 prepare_qat_fx 内部到底做了什么
     │
     ├──→ [2] Observer 消融 ──→ 理解 Observer 选择为什么能差 2+ 个点
     │
     ├──→ [3] 比特实验 ──→ 找到"崩溃点"，理解固定 scale 的上限在哪
     │
     ├──→ [4] BN 冻结原理 ──→ 理解分布漂移的本质
     │
     └──→ [5+6] 失败诊断 + 能力边界 ──→ 引出 LSQ
```

每一节的结尾都有一个"串起来"的叙事桥接，告诉你这节实验结果"意味着什么"和"接下来该问什么"。读完这个 Stage，你对 LSQ 的理解不会是"凭空出来一个数学技巧"，而是"一个已经被实验证明了必要性的方案"。

---

## 1. 手写 QAT：不用 torch.ao 跑通完整训练

### 1.1 为什么要手写？

Stage 1 教你用 PyTorch 的 `prepare_qat_fx` 一键完成 QAT 准备。这很方便——但也让你对 QAT 内部到底发生了什么"无感"。

**手写的目的不是替代 PyTorch，而是拆开黑箱。** 当你手动管理 `observer_enabled` / `fake_quant_enabled` 的切换、手动在每个 epoch 控制 scale 的冻结时机、手动处理 BN 的 eval 模式——你会发现很多"最佳实践"（Observer 开 2-3 epoch、BN 必须冻结）不再是需要背诵的咒语，而是可以推导出来的结论。

### 1.2 从零构造 FakeQuantize

PyTorch 的 `FakeQuantize` 是一个 C++ 加速的庞然大物。但它的核心逻辑不到 30 行——一个 Observer 收集 min/max、一个 scale 计算公式、一个量化 + 反量化的 forward。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyFakeQuantize(nn.Module):
    """手写 FakeQuantize——功能和 PyTorch 的一致，但没有 C++ 加速。

    内部状态机由两个 flag 控制:
      observer_enabled=True  → 收集 min/max, 不注入量化噪声
      fake_quant_enabled=True → 停止收集, 注入量化噪声 (前向做 quant+dequant)
    """
    def __init__(self, n_bits=8, symmetric=True):
        super().__init__()
        self.symmetric = symmetric
        qmax = 2 ** (n_bits - 1) - 1 if symmetric else 2**n_bits - 1
        qmin = -qmax if symmetric else 0
        self.qmin, self.qmax = qmin, qmax

        # scale 初始化为 1 —— 等 Observer 校准后再更新
        self.register_buffer("scale", torch.tensor(1.0))
        # 用来追踪校准过程中观测到的范围
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

        self.observer_enabled = True
        self.fake_quant_enabled = False

    def calibrate(self, x):
        """用运行中观测到的数据更新 min/max 和 scale"""
        self.min_val = torch.min(self.min_val, x.detach().min())
        self.max_val = torch.max(self.max_val, x.detach().max())

        if self.symmetric:
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            self.scale = abs_max / (self.qmax / 2) + 1e-8
        else:
            self.scale = (self.max_val - self.min_val) / (
                self.qmax - self.qmin
            ) + 1e-8

    def forward(self, x):
        if self.observer_enabled:
            self.calibrate(x)
        if self.fake_quant_enabled:
            q = torch.round(x / self.scale).clamp(self.qmin, self.qmax)
            return q * self.scale  # FakeQuant: 量化后再反量化, 仍是 float
        return x
```

### 1.3 状态机的三个阶段

你手写的 `MyFakeQuantize` 有两个 flag —— `observer_enabled` 和 `fake_quant_enabled`。它们的组合定义了 QAT 训练的三种状态：

```
阶段  (observer, fake_quant)   行为
──────────────────────────────────────────────────
初始   (True,   False)        纯 FP32 前向 — 跟普通训练一样
校准   (True,   False)        前向带观察: 收集 min/max → 计算 scale
                              权重在 FP32 下更新 (没有量化噪声干扰)

QAT    (False,  True)         前向带量化噪声: FakeQuantize 注入
                              权重在量化噪声下更新 (learn to be robust)
                              反向: STE 让梯度穿过 round()

推理   (False,  True)         QAT 的 eval 版本 — 量化噪声还在
                               但 observer 不再更新 scale
```

**关键设计决策**：为什么校准阶段 observer 开着但 fake_quant 关着？因为这个阶段的目标是在"干净"的信号下确定 scale。如果一开始就注入量化噪声，Observer 会在被噪声污染的数据上估算 scale —— 这个 scale 的精度比干净信号低。

### 1.4 手写 QAT Conv 层

```python
class ManualQATConv2d(nn.Module):
    """手动 QAT 卷积——weight 和 activation 各有自己的 FakeQuantize。

    输入经过 activation quantizer → int 模拟 → 权重经过 weight quantizer → int 模拟
    → conv2d → 输出 (FP32)

    注意: 输出仍是 FP32 —— 下一层的 ManualQATConv2d 会再次量化输入。
    这和 PyTorch 的 QAT 行为一致: 每层的"输入"和"权重"被量化, 但输出保持 FP32。
    """
    def __init__(self, in_c, out_c, k, n_bits_w=8, n_bits_a=8):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=k // 2)
        self.wq = MyFakeQuantize(n_bits_w, symmetric=True)   # 权重: 对称
        self.aq = MyFakeQuantize(n_bits_a, symmetric=False)  # 激活: 非对称

    def forward(self, x):
        w_q = self.wq(self.conv.weight)  # 量化权重 (FakeQuantized)
        x_q = self.aq(x)                  # 量化输入 (FakeQuantized)
        return F.conv2d(
            x_q, w_q, bias=self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding
        )
```

### 1.5 辅助函数 + 完整训练循环

```python
def set_all_fq(model, observer_enabled, fake_quant_enabled):
    """批量切换模型中所有 MyFakeQuantize 的状态"""
    for m in model.modules():
        if isinstance(m, MyFakeQuantize):
            m.observer_enabled = observer_enabled
            m.fake_quant_enabled = fake_quant_enabled

def freeze_bn(model):
    """冻结所有 BatchNorm 层 — QAT 期间必须做"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()                      # 用 running stats, 不用 batch stats
            m.requires_grad_(False)        # 不更新 BN 参数
```

```python
# ===== 完整的 QAT 训练循环 =====
from torch.utils.data import DataLoader
import torch.optim as optim

class SmallCNN(nn.Module):
    """3 个 QAT Conv + ReLU + MaxPool — 足够验证 QAT 的所有关键行为"""
    def __init__(self, n_bits_w=8, n_bits_a=8):
        super().__init__()
        self.conv1 = ManualQATConv2d(3, 32, 3, n_bits_w, n_bits_a)
        self.conv2 = ManualQATConv2d(32, 64, 3, n_bits_w, n_bits_a)
        self.conv3 = ManualQATConv2d(64, 128, 3, n_bits_w, n_bits_a)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

model = SmallCNN()
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# ========== Phase 1: 校准 (Observer Only) ==========
freeze_bn(model)
set_all_fq(model, observer_enabled=True, fake_quant_enabled=False)

for ep in range(2):  # 2 个 epoch 就够 MNIST/CIFAR-10 上的小模型
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    # eval: 关闭 observer, 用已校准的 scale 评估
    set_all_fq(model, observer_enabled=False, fake_quant_enabled=False)
    val_acc = evaluate(model, val_loader)
    print(f"[Calib] Epoch {ep:2d}, Val Acc: {val_acc:.2f}%")
    set_all_fq(model, observer_enabled=True, fake_quant_enabled=False)

# ========== Phase 2: QAT (FakeQuantize Enabled) ==========
set_all_fq(model, observer_enabled=False, fake_quant_enabled=True)

for ep in range(2, 10):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    # eval 时 observer 关、fake_quant 开（保持量化推理）
    set_all_fq(model, observer_enabled=False, fake_quant_enabled=True)
    model.eval()
    val_acc = evaluate(model, val_loader)
    print(f"[QAT]   Epoch {ep:2d}, Val Acc: {val_acc:.2f}%")

print(f"QAT 训练完成! 手写 QAT 和 PyTorch prepare_qat_fx 的效果等价。")
```

### 1.6 手写 QAT 教会你什么

跑完这段代码后，你应该能闭眼回答：

- `prepare_qat_fx` 内部到底做了什么？→ 在图中插入 FakeQuantize 节点 + 配置 QConfig 中的 Observer 类型 + 管理 observer_enabled/fake_quant_enabled 的状态切换
- 为什么校准阶段"observer 开、fake_quant 关"？→ 在干净信号上确定 scale，避免噪声污染
- 为什么 QAT 阶段"observer 关、fake_quant 开"？→ 固定 scale，注入量化噪声让模型学习鲁棒性

---

## 2. Observer 消融：四种 Observer 的正面对决

### 2.1 从手写到理解：Observer 为什么重要

手写 QAT 训练循环跑通后，你可能会注意到一个细节：我们用了一个最朴素的 Observer —— 直接取输入数据的 min 和 max。

但 "最优" 的 min/max 不在于覆盖**所有**数据——而在于平衡**"覆盖范围"（不要 clip 太多数据）和"分辨率"（clip 后的范围越小，每个量化等级越精细）**。

举个具体例子：激活值 99.9% 在 [0, 3.0]，但有一个 outlier = 50.0。
- MinMaxObserver → scale = 50/255 ≈ 0.196 → 每个 INT8 等级 ≈ 0.2 → 覆盖 [0,3] 只需要 15 个等级，但你有 256 个 → **精度浪费 94%**
- PercentileObserver(99.9%) → scale = 3/255 ≈ 0.012 → 每个等级 ≈ 0.012 → **精度好 16 倍**，代价是那个 outlier 被 clip

下面的消融实验会给你一个量化的答案：不同的 Observer 在同一模型、同一比特下能差多少？这个差距在低比特下是放大还是缩小？

### 2.2 四种 Observer 的数学原理

| Observer | 原理 | 优点 | 缺点 |
|----------|------|------|------|
| **MinMax** | `scale = max(|min|, |max|) / 127` | 最快，无参数 | 被 outlier 拖垮 |
| **MovingAvgMinMax** | `min = α·min_new + (1-α)·min_old`（EMA 平滑） | 抗单个 batch 的 outlier，比 MinMax 稳定 | 需要选择 α（默认 0.01） |
| **HistogramObserver (KL)** | 建激活值直方图 → 搜截断点 → 最小化 KL(FP32_hist \|\| Q_hist) | 理论上最优地在分布层面保持一致性 | 需要足够校准数据 + 建直方图 |
| **PerChannelMinMax** | 每个输出通道独立 scale | 消除了通道间范围差异 | per-channel 只适用于权重，激活做不了（硬件限制） |

**MovingAvgMinMax 为什么比 MinMax 好？** 假设你的校准包含 100 个 batch。第 37 个 batch 碰巧包含了所有 outlier——MinMax 的 scale 被这一个 batch 永久拉大。EMA 平滑后，单个 batch 的 outlier 对最终 scale 的影响被稀释到 1%（α=0.01 时约需 100 个 batch 才会完全反映），而"正常"范围的信号被 99 个 batch 的 EMA 平均值准确捕捉。

**KL Divergence 为什么是 TensorRT 的默认？** 量化本质上是把一个连续分布"投影"到离散的量化网格上。KL 散度衡量两个分布之间的信息损失——KL 小 = 量化后的分布"看起来"最像原始分布。TensorRT 选择 KL 不是因为它最快，而是因为它在几乎所有类型的激活分布上（均匀、高斯、长尾）都表现稳定。

### 2.3 完整消融实验代码

```python
import copy
from collections import defaultdict

def run_observer_ablation(model_fn, train_loader, val_loader, bit_widths=[8, 4]):
    """Observer 消融: 同一模型 × 4 种 Observer × 多比特 → 精度矩阵"""
    results = defaultdict(dict)

    for bits in bit_widths:
        print(f"\n{'='*50}")
        print(f"Bit-width: {bits}-bit")
        print(f"{'='*50}")

        for obs_name in ["MinMax", "MovingAvgMinMax", "KL", "PerChannelMinMax"]:
            model = model_fn(n_bits_w=bits, n_bits_a=bits)

            # 选择 Observer 类型
            set_observer_type(model, obs_name)

            # 校准 2 epoch + QAT 8 epoch
            acc = train_qat_manual(model, train_loader, val_loader,
                                   calib_epochs=2, qat_epochs=8)

            results[bits][obs_name] = acc
            print(f"  {obs_name:20s}: {acc:.2f}%")

    # 打印对比表
    print(f"\n{'Observer':20s}", end="")
    for b in bit_widths:
        print(f"{b}-bit{'':6s}", end="")
    print()
    print("-" * (20 + 12 * len(bit_widths)))
    for obs_name in ["MinMax", "MovingAvgMinMax", "KL", "PerChannelMinMax"]:
        print(f"{obs_name:20s}", end="")
        for b in bit_widths:
            print(f"{results[b][obs_name]:.2f}%      ", end="")
        print()

    return results
```

### 2.4 期望结果和解读

```
Observer            8-bit       4-bit
─────────────────────────────────────────
MinMax              89.1%       82.3%
MovingAvgMinMax     89.8%       84.7%      ← EMA 平滑有效
KL                  90.0%       85.5%      ← KL 搜最优截断
PerChannelMinMax    90.2%       86.8%      ← per-channel 消除通道间差异
```

三个关键发现：

1. **Observer 的差距随比特数降低而放大**。8-bit 下 MinMax 和最好的差 1.1 个点，4-bit 下差 4.5 个点。因为低比特下 scale 的精确性变得"生死攸关"——16 个等级对 scale 偏差的容忍度远低于 256 个等级。

2. **PerChannelMinMax 在所有比特下都是最优的（对权重）**。每个输出通道有独立 scale → 消除了通道间范围差异（某些通道的权重可能比其他通道大 10 倍以上）。

3. **MovingAvg 和 KL 之间的差距不大（~0.5%）**。EMA 平滑的边际效果在达到一定程度后趋于饱和——更好的信息论优化（KL）和更简单的指数平滑（EMA）在 100 个 batch 的校准下接近等价。

### 2.5 从 Observer 消融到比特实验：串起来

Observer 消融告诉你"选对 Observer 能差 2-5 个点"。但还有一个更根本的问题：**不同的比特数下，QAT 的表现如何？** 你会发现 8-bit 几乎无损、4-bit 开始降、2-bit 直接崩。这个"崩溃点"不是魔法——它是固定 scale QAT 的硬上限。**而这个上限，正是 Stage 2（LSQ）要解决的问题。**

所以下一节的实验不是"跑一个 benchmark"——它是在为 Stage 2 建立"问题意识"。做实验时留心观察：在哪个比特数下 scale 的精确性开始变得"生死攸关"？

---

## 3. 比特宽度系统性实验：找到"崩溃点"

### 3.1 实验设计

固定一切变量（模型架构、Observer 类型、训练超参），只改变量化比特数：8 → 6 → 4 → 3 → 2-bit。观察精度曲线的形状——特别关注"拐点"出现的位置。

### 3.2 完整实验代码

```python
def run_bitwidth_experiment(model_fn, train_loader, val_loader):
    """在固定一切变量的前提下, 只改比特数, 找到固定 scale QAT 的"崩溃点" """
    bit_widths = [8, 6, 4, 3, 2]
    results = {}
    curves = defaultdict(list)  # 记录训练过程中的精度曲线

    # 先跑 FP32 baseline
    model_fp = model_fn(n_bits_w=32, n_bits_a=32)
    fp32_acc = train_standard(model_fp, train_loader, val_loader, epochs=10)
    print(f"FP32 Baseline: {fp32_acc:.2f}%\n")

    for bits in bit_widths:
        model = model_fn(n_bits_w=bits, n_bits_a=bits)
        set_observer_type(model, "PerChannelMinMax")

        # 记录每个 epoch 的 val accuracy
        epoch_accs = train_qat_manual(
            model, train_loader, val_loader,
            calib_epochs=2, qat_epochs=8,
            record_every_epoch=True
        )
        curves[bits] = epoch_accs
        final_acc = epoch_accs[-1]
        results[bits] = final_acc

        # 计算相对 FP32 的精度损失
        loss_pct = (fp32_acc - final_acc) / fp32_acc * 100
        print(f"  {bits}-bit: {final_acc:.2f}% (Δ = -{loss_pct:.1f}%)")

    # 找到"拐点"——精度下降加速的比特数
    deltas = {b: fp32_acc - results[b] for b in bit_widths}
    print(f"\n精度损失 (Δ vs FP32):")
    for b in bit_widths:
        bar = "█" * int(deltas[b])
        print(f"  {b}-bit: {deltas[b]:.1f}% {bar}")
    print(f"  ★ 拐点 ≈ {find_knee_point(deltas)}-bit — 固定 scale QAT 从这里开始崩")

    return results, curves

def find_knee_point(deltas):
    """找到精度下降二阶差分最大的位置 = 拐点"""
    bits = sorted(deltas.keys())
    accels = []
    for i in range(2, len(bits)):
        d1 = deltas[bits[i]] - deltas[bits[i-1]]
        d0 = deltas[bits[i-1]] - deltas[bits[i-2]]
        accels.append(d1 - d0)
    return bits[accels.index(max(accels)) + 2]
```

### 3.3 期望结果

```
FP32 Baseline: 91.2%

Bit-width  Acc     Δ vs FP32  Status
──────────────────────────────────────────
8-bit      91.0%   -0.2%      几乎无损 ✓
6-bit      90.3%   -0.9%      轻微下降
4-bit      87.8%   -3.4%      明显下降 ← "拐点"
3-bit      81.2%   -10.0%     加速下降
2-bit      68.5%   -22.7%     崩溃 ✗
```

### 3.4 拐点的数学解释

为什么拐点在 4-bit？量化精度的信息论公式：

```
SNR(dB) ≈ 6.02 × n_bits + 1.76    (对均匀量化)

8-bit:  SNR ≈ 49.9 dB  →  信号能量 / 噪声能量 ≈ 10^5
4-bit:  SNR ≈ 25.8 dB  →  信号能量 / 噪声能量 ≈ 380
2-bit:  SNR ≈ 13.8 dB  →  信号能量 / 噪声能量 ≈ 24
```

**但这是假设 scale 完美的上限。** 实际中，scale 由校准数据的 min/max 决定——它有 ≈5-10% 的误差（取决于校准数据量）。

对 8-bit（256 个等级），scale 差 5% → 相邻量化等级的间距差 5% → 仍然有足够的粒度容纳
对 4-bit（16 个等级），scale 差 5% → 一个等级可能跨越两个"最优"权重值 → 量化误差被放大
对 2-bit（4 个等级），scale 差 5% → 基本不可能有"正确"的量化

**这就是为什么"让 scale 可以学习"（LSQ）是必需的。** 固定 scale 的 QAT 天花板在 4-bit，要突破这个天花板需要 scale 能跟随 weight 一起更新。

### 3.5 画"崩溃曲线"

```python
import matplotlib.pyplot as plt

def plot_collapse_curve(results, fp32_acc):
    bits = sorted(results.keys())
    accs = [results[b] for b in bits]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 精度 vs 比特数
    axes[0].plot(bits, accs, 'o-', linewidth=2, markersize=8, color='#E74C3C')
    axes[0].axhline(y=fp32_acc, linestyle='--', color='gray', label=f'FP32 ({fp32_acc:.1f}%)')
    axes[0].axvline(x=4, linestyle=':', color='orange', label='Knee Point (~4-bit)')
    axes[0].set_xlabel('Bit Width')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Fixed-Scale QAT: Bit-Width vs Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 精度损失率
    deltas = [(fp32_acc - a) / fp32_acc * 100 for a in accs]
    axes[1].bar([str(b) for b in bits], deltas, color='#E74C3C', alpha=0.8)
    axes[1].set_xlabel('Bit Width')
    axes[1].set_ylabel('Accuracy Loss (%)')
    axes[1].set_title('Accuracy Degradation Rate')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('qat_bitwidth_collapse.png', dpi=150, bbox_inches='tight')
    print("崩溃曲线已保存 → qat_bitwidth_collapse.png")
```

**这张图是整个 Stage 1.5 最重要的产出。** 它会成为你判断"某个量化方案是否有效"的参考基线——任何声称解决了低比特量化问题的方法，都应该能把这根曲线向右下方推。

---

## 4. BN 冻结的深层原理

### 4.1 QAT 期间的分布漂移

BatchNorm 在训练期间维护两套统计量：
- **Batch stats**（`μ_batch`, `σ²_batch`）：当前 mini-batch 的均值和方差，用于训练时归一化
- **Running stats**（`running_mean`, `running_var`）：EMA 平滑的全局估计，**推理时使用**

QAT 期间 FakeQuantize 会改变每一层输出的激活值分布。如果 BN 的 running stats 继续在每个 epoch 更新，问题就来了：

```
Epoch 1: 激活分布 ≈ FP32（observer 刚开，分布接近原始）
Epoch 2: 激活分布 = FP32 + observer 的 min/max 追踪（轻微偏移）
Epoch 3: 激活分布 = 被 FakeQuantize 修改后的分布（observer 已关，scale 固定）
            ↑ 量化噪声改变了分布的方差和均值
Epoch 4: BN running stats 现在基于 Epoch 3 的"被量化噪声改变"的分布更新
          但 input 分布又变了（因为 weight 在 QAT 中更新了）...
```

**每个 epoch 的 BN stats 基于不同的分布**——running_mean 在"漂移"，而不是收敛到一个稳定值。这导致推理时 BN 用的 running stats 和训练时任何一个 epoch 的分布都不一致。

### 4.2 对照实验：冻结 vs 不冻结

```python
def compare_bn_freeze(model_fn, train_loader, val_loader):
    """BN 冻结 vs 不冻结 — 正面对照"""
    results = {}

    # 实验 A: BN 冻结（正确做法）
    model_a = model_fn(n_bits_w=4, n_bits_a=4)
    set_observer_type(model_a, "PerChannelMinMax")
    freeze_bn(model_a)  # ← 关键!
    acc_a = train_qat_manual(model_a, train_loader, val_loader,
                             calib_epochs=2, qat_epochs=8)
    results["BN Frozen"] = acc_a

    # 实验 B: BN 不冻结（错误做法）
    model_b = model_fn(n_bits_w=4, n_bits_a=4)
    set_observer_type(model_b, "PerChannelMinMax")
    # 不调 freeze_bn(model_b)!
    acc_b = train_qat_manual(model_b, train_loader, val_loader,
                             calib_epochs=2, qat_epochs=8)
    results["BN Unfrozen"] = acc_b

    # 实验 C: BN 冻结 + 训练后期解冻（常见变体）
    model_c = model_fn(n_bits_w=4, n_bits_a=4)
    set_observer_type(model_c, "PerChannelMinMax")
    freeze_bn(model_c)
    acc_c = train_qat_manual(model_c, train_loader, val_loader,
                             calib_epochs=2, qat_epochs=8,
                             unfreeze_bn_at_epoch=6)  # 最后 2 个 epoch 解冻 BN
    results["BN Frozen then Unfrozen"] = acc_c

    for k, v in results.items():
        print(f"  {k:25s}: {v:.2f}%")
    return results
```

### 4.3 追踪 running_mean 的漂移

```python
def track_bn_running_mean(model_fn, train_loader, freeze=True, n_epochs=10):
    """在每个 epoch 后记录 BN running_mean，画漂移曲线"""
    model = model_fn(n_bits_w=4, n_bits_a=4)
    if freeze:
        freeze_bn(model)

    # 找到第一个 BN 层
    bn_layer = None
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_layer = m
            break

    means_history = []
    for ep in range(n_epochs):
        model.train()
        for x, y in train_loader:
            opt = optim.SGD(model.parameters(), lr=0.01)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        # 记录 BN running_mean
        means_history.append(bn_layer.running_mean.detach().clone())

        if ep > 0:
            # 计算 running_mean 的漂移量
            drift = (means_history[-1] - means_history[-2]).abs().mean().item()
            print(f"  Epoch {ep:2d}, running_mean drift: {drift:.6e}, "
                  f"frozen={freeze}")

    # 画漂移曲线
    means = torch.stack(means_history)  # [n_epochs, n_channels]
    plt.figure(figsize=(12, 4))
    for c in range(min(8, means.size(1))):
        plt.plot(range(n_epochs), means[:, c].numpy(),
                 label=f'Ch {c}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('running_mean')
    plt.title(f"BN Running Mean Trajectory (frozen={freeze})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'bn_drift_frozen={freeze}.png', dpi=150, bbox_inches='tight')

    return means_history
```

### 4.4 BN 冻结的最佳实践

| 场景 | 建议 |
|------|------|
| 通用 QAT | **全程冻结 BN**。safe and simple |
| 校准数据很少 (<100 张) | **全程冻结 BN**。此时 BN 没足够数据收敛 |
| QAT 训练较长 (≥20 epoch) | 可以最后 2-3 epoch 解冻 BN，但需极小 lr (1e-6) |
| 检测模型 (YOLO/FCOS) | **全程冻结 BN**。检测头的 reg 分支对 BN 漂移极度敏感 |
| Transformer / LLM | 没有 BN（用 LayerNorm），此条不适用 |

---

## 5. QAT 失败案例诊断手册

### 5.1 从零散"坑"到系统性理解

上面四个实验（手写、Observer 消融、比特实验、BN 冻结）各自揭示了 QAT 的一个维度。下面把这些维度串成一份**诊断手册**——当你遇到 QAT 效果差时，这份手册能帮你从症状 → 假设 → 验证 → 修复。

### 5.2 六大常见失败模式

| # | 症状 | 最可能根因 | 验证方法 | 修复 |
|---|------|----------|---------|------|
| 1 | **QAT 精度比 PTQ 还差** | Observer 开启时间太短, scale 没收敛 | 打印每个 epoch 后的 scale 值——如果还在大幅变化，说明没收敛 | 延长 observer 开启到 5 epoch；或增加校准数据量 |
| 2 | **Loss 剧烈震荡 (std > 0.5× mean)** | BN 未冻结，running stats 在漂移 | 检查 `bn.training` 是否为 True；打印 running_mean 看是否有明显漂移 | `model.eval()` for BN + `requires_grad_(False)` |
| 3 | **训练中某 epoch 后 loss 突升** | LR 过大，"忘记"了预训练权重 | 在 QAT 模式下做 1 个 epoch 的不更新训练（纯前向），观察 loss — 如果已经比 FP32 的 loss 高很多，说明初始偏移太大 | 降 LR 到 1e-5；或用 warmup: 前 100 step LR 从 1e-6 升到目标值 |
| 4 | **某些通道输出全部相同** | `reduce_range=True` 误用 | 检查 qconfig 中的 `reduce_range` 参数 | `reduce_range=False` |
| 5 | **导出模型大小没变** | 忘记 `convert()` — FakeQuantize 还在模型里 | 检查模型中的 op 类型: `print(model)` 看有没有 `FakeQuantize` | `model_int8 = convert(model_prepared.eval())` |
| 6 | **ONNX 没有 QDQ 节点** | opset 过低 (<13) | 检查导出时的 opset 版本 | `opset_version=17` |

### 5.3 案例 1 深度分析：QAT 精度比 PTQ 还差

这是最多人踩的坑。逻辑链是：

```
Observer 只开了 1 个 epoch
  → scale 基于 1 个 epoch 内看到的激活值决定
    → 这个 scale 可能和后续 9 个 epoch QAT 中实际的数据分布不一致
      → QAT 训练时 weight 被迫适应一个"错误的"量化网格
        → 精度不如 PTQ（PTQ 的 scale 用全部校准数据决定，反而更准）
```

修复验证实验：

```python
def debug_qat_worse_than_ptq(model_fn, train_loader, val_loader):
    """对比不同 observer 开启时长对最终精度的影响"""
    for calib_epochs in [1, 2, 3, 5, 10]:
        model = model_fn(n_bits_w=4, n_bits_a=4)
        acc = train_qat_manual(model, train_loader, val_loader,
                               calib_epochs=calib_epochs, qat_epochs=10 - calib_epochs)
        print(f"  Calib epochs={calib_epochs:2d}: {acc:.2f}%")
    # 期望: calib_epochs=1 → 精度最差, calib_epochs=3-5 → 精度达平台
```

### 5.4 案例 4 深度分析：某些通道输出全部相同

`reduce_range=True` 是 PyTorch 量化 API 中的一个隐藏参数。它的作用是将量化范围从 [-128, 127] 缩小到 [-127, 127]，使得 256 个值对称（省去 −128 的偏移），对某些后端（如 `fbgemm`）有轻微加速。

但代价是：如果某个通道的权重全都落在一个很窄的区间，缩小的范围会导致多个相邻通道被量化到同一个值——"坍缩"。这些通道不再参与表示不同模式——等效于通道数减少了。

**规则**：除非你说得出具体后端为什么需要它，否则永远不碰 `reduce_range`。

---

## 6. 固定 scale QAT 的能力边界 → LSQ 的动机

### 6.1 把四组实验串成一条逻辑链

做完上面的实验后，你已经有了一个实验数据表。现在把它串起来：

```
[比特宽度实验] → 固定 scale QAT 在 4-bit 开始崩, 2-bit 完全不可用
                     ↓ 为什么?
[Observer 消融]  → 低比特下 Observer 的选择能差 5 个点
                     → scale 的精确性在低比特至关重要
                     ↓ 但 scale 怎么才能更精确?
[校准过程分析]   → 固定 scale = 用前 2-3 个 epoch 决定后续所有 epoch 的 scale
                     → 但 weight 在 QAT 中不断变化
                     → 最优 scale 也在变化
                     → 固定 scale 永远"追不上"
                     ↓ 那怎么办?
[LSQ]           → 让 scale 变成 nn.Parameter
                     → 和 weight 一起被梯度优化
                     → weight 变 → scale 也跟着变
```

### 6.2 低比特下固定 scale 的三大失效模式

1. **Clip Error 主导**：低比特下量化等级少，clip error（截断误差）比 round error（舍入误差）致命得多。一个 outlier 被 clip = 一整组数据被量化到同一个值。

2. **Scale 偏差被放大**：8-bit 下 scale 差 5% → 每个等级差 5%，256 个等级的累积误差可容忍。4-bit 下 scale 差 5% → 16 个等级的累积误差大幅增加。2-bit 下 scale 差 5% → 4 个等级，可能完全错过最优值。

3. **Scale 的时效性**：校准阶段的 scale 只反映"weight 在 QAT 开始时的分布"。但 QAT 训练中 weight 不断进化（学习适应量化噪声），最优 scale 和第 2 个 epoch 的 scale 可能相差 20% 以上。固定 scale 永远基于"过去"的权重分布，无法适应"现在"的。

### 6.3 LSQ 怎么解决

```
固定 scale:  s = observer(weight)     → 一次决定, 永久不动
                 ↓ 缺点: 过时 + 不精确

LSQ scale:  s = nn.Parameter(s_init)  → 每次 backward 更新
                 ↓ 优点: 始终追踪当前 weight 分布

LSQ backward:  ∂v̂/∂s = -v/s + round(v̄)    ← 这就是 Stage 2 要推导的公式 (6)!
```

LSQ 不需要额外的 loss 项——整个优化机制被封装在 backward 的梯度计算中。SGD 自动驱动 scale 向"使量化误差最小"的方向更新。

**进入 Stage 2 时，你不是在学一个"凭空出现的数学技巧"——你是在回答 Stage 1.5 实验提出的一个具体问题：**

> "我的实验数据已经证明：固定 scale QAT 在 4-bit 以下不可用。LSQ 就是针对这个实验发现的工程解决方案。"

---

## 7. 动手实验

| # | 实验 | 时间 | 产出 |
|---|------|:--:|------|
| 1 | 手写 FakeQuantize + 完整 QAT 训练循环 (CIFAR-10) | 1h | 不依赖 torch.ao 的 QAT 实现 |
| 2 | Observer 消融：4 种 × {8,4}-bit × 同一模型，画对比柱状图 | 1h | Observer 在不同比特下的表现对比 |
| 3 | 比特宽度实验：8→6→4→3→2-bit，画崩溃曲线，找拐点 | 45min | **全 Stage 最重要的图表** |
| 4 | BN 冻结 vs 不冻结，画 running_mean 漂移曲线 | 30min | 理解 BN 漂移的量化证据 |
| 5 | calib_epochs 消融：1/2/3/5/10，找到最优 observer 时长 | 30min | 实验数据的平台点 |
| 6 | 诊断 3 种 QAT 失败案例（故意制造 + 按手册修复） | 45min | 诊断能力的肌肉记忆 |

### 实验记录模板

```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ 实验 #    │ 比特数   │ Observer │ Calib Ep │ BN Frozen│ Val Acc  │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 1        │ 8        │ MinMax   │ 2        │ Y        │ 89.1%    │
│ 2        │ 4        │ MinMax   │ 2        │ Y        │ 82.3%    │
│ ...      │ ...      │ ...      │ ...      │ ...      │ ...      │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 检验标准

- [ ] 能手写 FakeQuantize + 完整 QAT 训练循环（不用 torch.ao）
- [ ] 能跑 4 种 Observer 的消融实验，说出 Observer 选择对精度的量化影响
- [ ] 能画出 8/6/4/3/2-bit QAT 精度曲线，指出"拐点"并解释原因
- [ ] 能基于 running_mean 曲线解释 BN 冻结的必要性
- [ ] 能独立诊断并修复 3 种 QAT 失败案例
- [ ] 能用自己的话解释"为什么固定 scale QAT 在低比特下失效 → LSQ 如何解决"
- [ ] 实验记录表填满至少 10 行

---

> 💡 **学习建议**：这个 Stage 最重要的产出不是代码，而是**实验数据 + 基于数据的结论**。建议建一个实验记录表格（比特数、Observer、BN 状态、LR、最终精度），每做完一个实验就填一行。
>
> 做完所有实验后，你应该能回答开篇的四个问题——不是靠记忆，而是靠"我做过的消融实验数据告诉我的"。
>
> Stage 2 学 LSQ 时，你会频繁回来看这节的比特实验结果。那时你会意识到：LSQ 不是在"发明"一个新算法——它是在"设计一个解决方案"来突破你用实验发现的硬上限。
>
> Next: [Stage 2: LSQ — 让 scale 活起来](./Stage2_LSQ与可微量化参数.md)
