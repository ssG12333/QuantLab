# Stage 1.5: QAT 训练深度剖析 — 从"能跑"到"理解为什么"

> ⏱ 预计学习时间：12-18 小时 | 🎯 难度：⭐⭐⭐
>
> Stage 1 教你"怎么调 QAT API"。这里补上缺失的一环：**QAT 训练本身到底发生了什么。**
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

Stage 1 结束时你能用 `prepare_qat_fx` 跑通 QAT 训练。但如果我问你：

- "为什么 Observer 要开启 2-3 个 epoch 而不是 1 个或 5 个？"
- "如果我把 HistogramObserver 换成 MinMaxObserver，精度会怎样？"
- "为什么 8-bit QAT 几乎不掉精度，4-bit 开始掉，2-bit 直接崩？"

你能**不查文档、根据第一性原理解释**吗？

这个 Stage 的目标就是把 QAT 训练拆成零件，逐个实验验证，让你对上述问题都有基于数据的答案。所有实验都在 MNIST/CIFAR-10 上做（数据量小，跑得快），但结论适用于一切模型。

---

## 1. 手写 QAT：不用 torch.ao 跑通完整训练

目标：脱离 PyTorch 量化 API，从零构造一个 QAT 训练循环。这会让你彻底理解 `prepare_qat_fx` 帮你做了什么。

### 1.1 从零构造 FakeQuantize

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MyFakeQuantize(nn.Module):
    """手写 FakeQuantize——功能和 PyTorch 的一致，但没有 C++ 加速"""
    def __init__(self, n_bits=8, symmetric=True):
        super().__init__()
        self.symmetric = symmetric
        qmax = 2**(n_bits-1)-1 if symmetric else 2**n_bits-1
        qmin = -qmax if symmetric else 0
        self.qmin, self.qmax = qmin, qmax
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.observer_enabled = True
        self.fake_quant_enabled = False

    def calibrate(self, x):
        self.min_val = torch.min(self.min_val, x.detach().min())
        self.max_val = torch.max(self.max_val, x.detach().max())
        abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
        self.scale = abs_max / (self.qmax / 2) + 1e-8

    def forward(self, x):
        if self.observer_enabled:
            self.calibrate(x)
        if self.fake_quant_enabled:
            q = torch.round(x / self.scale).clamp(self.qmin, self.qmax)
            return q * self.scale  # FakeQuant: 量化后反量化, 还是 float
        return x
```

### 1.2 手写 QAT 训练循环

```python
class ManualQATConv2d(nn.Module):
    """手动 QAT 卷积——weight 和 activation 各有自己的 FakeQuantize"""
    def __init__(self, in_c, out_c, k, n_bits_w=8, n_bits_a=8):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=k//2)
        self.wq = MyFakeQuantize(n_bits_w, symmetric=True)
        self.aq = MyFakeQuantize(n_bits_a, symmetric=False)

    def forward(self, x):
        w_q = self.wq(self.conv.weight)
        x_q = self.aq(x)
        return F.conv2d(x_q, w_q, bias=self.conv.bias,
                        stride=self.conv.stride, padding=self.conv.padding)

# ===== 手写 QAT 训练 =====
model = SmallCNN()  # 3 个 ManualQATConv2d + ReLU + Pool
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Phase 1 (Calibration): 只观察, 不做量化
set_all_fq(model, observer=True, fq=False)
for ep in range(2):
    for x, y in train_loader:
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()

# Phase 2 (QAT): 固定 scale, 做量化训练
set_all_fq(model, observer=False, fq=True)
for ep in range(2, 10):
    for x, y in train_loader:
        out = model(x); loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
```

跑完后你应该能回答：**`prepare_qat_fx` 内部到底做了什么？** 答案：它在图中插入了 FakeQuantize 节点，配置了 QConfig 中的 Observer 类型，并管理了 observer_enabled/fake_quant_enabled 的状态切换——和你手写的逻辑完全一样。

---

## 2. Observer 消融：四种 Observer 的正面对决

同一个模型、同一个比特（4-bit）、同样的训练条件，换不同类型的 Observer。记录 test accuracy。

```python
observers = {
    "MinMax":             MyFakeQuantize(observer_type="minmax"),
    "MovingAvgMinMax":    MyFakeQuantize(observer_type="moving_avg"),
    "Histogram(KL)":      MyFakeQuantize(observer_type="histogram_kl"),
    "PerChannelMinMax":   MyFakeQuantize(observer_type="per_channel"),
}

results = {}
for name, obs in observers.items():
    acc = train_and_eval_with_observer(obs, bits=4, epochs=10)
    results[name] = acc

# 期望结果（CIFAR-10, ResNet-20, 4-bit）:
# MinMax:             85.2%  ← 基础
# MovingAvgMinMax:    86.8%  ← EMA 平滑有效
# Histogram(KL):      87.3%  ← KL 搜索最优
# PerChannelMinMax:   87.5%  ← 每通道独立 scale
```

**核心发现**：不同的 Observer 可以差 2+ 个点。选 Observer 不是"背答案"——是理解数据分布后的选择。

---

## 3. 比特宽度系统性实验：找到"崩溃点"

在 CIFAR-10 上用同一个模型跑 8/6/4/3/2-bit QAT：

```python
for bits in [8, 6, 4, 3, 2]:
    acc = train_fixed_qat(bits, epochs=10)
    results[bits] = acc

# 期望结果:
# FP32: 91.2%
# 8-bit: 91.0%  ← 几乎无损
# 6-bit: 90.3%  ← 轻微下降
# 4-bit: 87.8%  ← 开始明显下降 ← "拐点"
# 3-bit: 81.2%  ← 加速下降
# 2-bit: 68.5%  ← 崩溃
```

**画"比特数 vs 精度"曲线**。你会发现一个明显的"拐点"在 4-bit 和 3-bit 之间——这就是固定 scale QAT 的能力边界。

**为什么会有这个拐点？** 8-bit 的 256 个量化等级足够细，即使 scale 不太精确也能凑合。但 4-bit（16 个等级）和 3-bit（8 个等级）下，scale 的微小偏差就导致整层量化网格"失准"。到 2-bit（4 个等级），scale 几乎不可能被正确设置——因为"正确"的 scale 在训练过程中随时在变化，而固定 scale 无法跟随。

**这个发现直接导向 LSQ**：如果 scale 能跟随 weight 一起学习、一起更新，低比特的拐点就可以被推到更低的比特数。

---

## 4. BN 冻结的深层原理

做一组对照实验：

```python
# A: BN 冻结（正确）
freeze_bn(model)
qat_train(model, epochs=10)  # 精度稳定

# B: BN 不冻结（错误）
qat_train(model, epochs=10)  # 精度在 epoch 3+ 突然崩掉
```

**为什么会崩？** FakeQuantize 改变了每层输出的激活值分布。如果 BN 继续用 batch stats 更新 running stats：

1. Epoch 1: 激活分布 = 正常 FP32（observer 刚开）
2. Epoch 3: 激活分布 = 被 FakeQuantize 修改后的分布（observer 已关）
3. 每个 epoch 的 BN stats 基于不同的分布，running_mean 在"漂移"

可以画 running_mean 的轨迹来验证：

```python
# 记录每个 epoch 后的 BN running_mean
means_history = []
for ep in range(10):
    train_one_epoch(model)
    means_history.append(model.bn1.running_mean.clone())
# 画出 means_history——如果没冻结，能看到明显的漂移曲线
```

---

## 5. QAT 失败案例诊断手册

| 症状 | 最可能根因 | 验证方法 | 修复 |
|------|----------|---------|------|
| QAT 精度比 PTQ 还差 | Observer 开启时间太短, scale 没收敛 | 延长 observer epoch 数到 5 | disable_observer_epoch = 5 |
| Loss 剧烈震荡 | BN 未冻结 | 检查 `bn.training` 是否为 True | 对所有 BN: `.eval()` + `requires_grad_(False)` |
| 训练中某 epoch 后 loss 突升 | LR 过大,"忘记"预训练权重 | 降到 1e-5 | `lr = 1e-5` |
| 某些通道输出全部相同 | `reduce_range=True` 误用 | 检查 qconfig | `reduce_range=False` |
| 导出模型大小没变 | 忘记 `convert()` | 检查模型中的 op 类型 | `model_int8 = convert(model_prepared)` |
| ONNX 没有 QDQ 节点 | opset 过低 | 检查 opset | `opset_version = 17` |

**每个案例的推理链**：症状 → 假设 → 设计实验验证假设 → 修复 → 验证修复有效。这个过程比"记住每个修复"重要 100 倍。

---

## 6. 固定 scale QAT 的能力边界 → LSQ 的动机

做完上面的实验后，坐下来想一个问题：

**"固定 scale QAT 在什么条件下会失效？"**

答案是：**当量化比特数很低（≤4-bit）时，scale 的精确性变得至关重要，而固定 scale（只靠几个 epoch 的 Observer 决定）的精度不够。**

在低比特下：
- Clip error 主导（量化范围太小或太大都致命）
- Scale 的微小偏差（10%）→ 量化误差急剧放大
- Weight 在 QAT 训练中不断变化 → 最优 scale 也在变化 → 固定 scale 永远"追不上"

**解决方案**：让 scale 变成一个可学习的参数，和 weight 一起通过梯度下降来优化。weight 变 → scale 也跟着变，始终保持在"当前 weight 分布下的最优值"。

**这就是 LSQ 要解决的问题。** 进入 Stage 2 时，你不是在学一个"凭空出现的数学技巧"，而是在回答 Stage 1.5 实验提出的一个具体问题。

---

## 7. 动手实验

| # | 实验 | 时间 |
|---|------|:--:|
| 1 | 手写 FakeQuantize + 手写 QAT 训练循环 (CIFAR-10) | 1h |
| 2 | Observer 消融：4 种 × 4-bit × 同一模型 | 1h |
| 3 | 比特宽度实验：8→6→4→3→2-bit, 画崩溃曲线 | 45min |
| 4 | BN 冻结 vs 不冻结对比，画 running_mean 曲线 | 30min |
| 5 | 诊断 4 种 QAT 失败案例（故意制造 + 修复） | 45min |

---

## 检验标准

- [ ] 能手写 FakeQuantize + 完整 QAT 训练循环（不用 torch.ao）
- [ ] 能画出 8/6/4/3/2-bit QAT 精度曲线，指出"拐点"
- [ ] 能解释 4 种 Observer 的区别和使用场景
- [ ] 能基于 running_mean 曲线解释 BN 冻结的必要性
- [ ] 能独立诊断并修复 4 种 QAT 失败案例
- [ ] 能用自己的话解释"为什么固定 scale QAT 在低比特下失效 → LSQ 如何解决"

---

> 💡 **学习建议**：这个 Stage 最重要的产出不是代码，而是**实验数据 + 基于数据的结论**。建议建一个实验记录表格（比特数、Observer、BN 状态、LR、最终精度），每做完一个实验就填一行。Stage 2 学 LSQ 时，你会频繁回来看这些数据。
>
> Next: [Stage 2: LSQ — 让 scale 活起来](./Stage2_LSQ与可微量化参数.md)
