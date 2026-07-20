# Stage 2: LSQ — 让 scale 活起来

> ⏱ 预计学习时间：20-30 小时 | 🎯 难度：⭐⭐⭐
>
> Stage 1.5 的结尾你用实验数据证明了：固定 scale QAT 在 4-bit 开始崩，2-bit 完全不可用。
> 根本原因：scale 只靠前几个 epoch 的 Observer 决定一次，然后冻结——但 weight 在 QAT 中不断进化，
> 最优 scale 也在变化，固定 scale 永远"追不上"。
>
> **LSQ（Learned Step Size Quantization）的解决方案极其简洁**：把 `register_buffer("scale")` 改成
> `nn.Parameter(scale_init)`，让 scale 和 weight 一样被梯度下降优化。
> 但实现起来需要解决两个核心问题：scale 的梯度怎么算（`round()` 把梯度切断了），
> 以及 scale 梯度的量级怎么控制（一个 scale 影响几千个 weight）。
>
> 这个 Stage 就带你从 `torch.autograd.Function` 的底层机制开始，
> 到手推 LSQ 的梯度公式（6）+ Gradient Scaling 公式（13），
> 到从零实现含 LSQ+ 的完整量化训练，再到把 LSQ 集成为自定义 PT2E Quantizer。

---

## 目录

1. [开篇：从"你的实验数据"说起](#开篇)
2. [知识总览](#知识总览)
3. [1. STE 深潜：autograd.Function 的完整生命周期](#1-ste-深潜)
4. [2. LSQ 梯度推导：公式 (6) 的逐行手推](#2-lsq-梯度推导)
5. [3. Gradient Scaling：公式 (13) 的完整推导](#3-gradient-scaling)
6. [4. 从零实现 LSQ：完整的 autograd.Function + 初始化策略](#4-从零实现-lsq)
7. [5. LSQ+：zero_point 也可学习](#5-lsq)
8. [6. LSQ 在 PT2E 中的集成：自定义 Quantizer](#6-lsq-在-pt2e-中的集成)
9. [7. 完整训练实验：从代码到可视化](#7-完整训练实验)
10. [8. 演进史：PACT → LSQ → LSQ+](#8-演进史)
11. [检验标准](#检验标准)

---

## 开篇：从"你的实验数据"说起

回顾 Stage 1.5 你亲手跑出的数据（CIFAR-10, ResNet-20）：

```
┌──────────┬────────┬────────┬────────┬────────┐
│ Method   │ 8-bit  │ 4-bit  │ 3-bit  │ 2-bit  │
├──────────┼────────┼────────┼────────┼────────┤
│ Fixed QAT│ 91.0%  │ 87.8%  │ 81.2%  │ 68.5%  │
└──────────┴────────┴────────┴────────┴────────┘
```

4-bit 比 FP32 掉 3.4 个点，2-bit 直接崩掉 22.7 个点。为什么会崩？

**根因不在"比特数太少"，而在"scale 是死的"。** 校准阶段确定了一个 scale，然后它在 QAT 的 8 个 epoch 里全程不变。但 weight 在 QAT 中不断被梯度更新——weight 的分布变了，最优 scale 也就跟着变。固定 scale 基于"第 2 个 epoch 时的 weight 分布"——到第 8 个 epoch 时可能已经偏离 20% 以上。

在 8-bit（256 个等级）下，scale 差 20% 还能凑合——256 个等级足够密。在 2-bit（4 个等级）下，scale 差 20% 可能就跳过一个完整等级——量化网格完全"对不准"了。

**LSQ 的答案如此简洁，以至于你可能会怀疑"这就够了？"**：

```python
# 固定 scale (Stage 1.5 的 QAT)
self.register_buffer("scale", torch.tensor(1.0))
# scale 不被 optimizer 更新 — 永远是最初 Observer 决定的那个值

# LSQ (这个 Stage)
self.scale = nn.Parameter(torch.tensor(1.0))
# scale 和 weight 一起被 optimizer 更新 — 始终追踪当前 weight 分布
```

但 `nn.Parameter` 只解决了"optimizer 能看到 scale"这一点。剩下两个硬问题：

1. **scale 的梯度怎么算？** `round()` 是不可导的——你需要手动写 `autograd.Function.backward`。
2. **scale 梯度的量级对吗？** 一个 per-tensor scale 影响几千个 weight——所有 weight 的梯度汇到 1 个 scale 上，天然比 weight 梯度大几十到几千倍。不加控制，scale 直接振荡发散。

这两个问题就是 §1（STE 深潜）和 §3（Gradient Scaling 推导）要解决的。

---

## 知识总览

```
                      ┌──────────────────────────────────────┐
                      │   LSQ 的三块积木                      │
                      └──────────────────────────────────────┘

  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
  │ 1. STE 深潜      │   │ 2. 公式 (6) 推导  │   │ 3. Gradient Scaling  │
  │ autograd.Function│   │ ∂v̂/∂s 的完整计算  │   │ 公式 (13) 的推导      │
  │ 的完整生命周期    │   │ 三种情况的梯度     │   │ 为什么是 1/√(N×Q)     │
  └────────┬────────┘   └────────┬─────────┘   └──────────┬───────────┘
           │                     │                        │
           └─────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   LSQ 的完整实现      │
                    │   LSQFunction.apply() │
                    │   + LSQQuantizer      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  基础 LSQ     │ │  LSQ+        │ │  PT2E 集成   │
     │  对称, 只学   │ │  非对称, 学  │ │  自定义       │
     │  scale        │ │  scale + zp  │ │  Quantizer    │
     └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 1. STE 深潜：autograd.Function 的完整生命周期

### 1.1 `round()` 为什么切断梯度

```python
y = round(x)  # 前向: 3.7 → 4.0, 2.3 → 2.0

# round() 的数学定义:
#   round(x) = floor(x + 0.5), x ≥ 0
#   round(x) = ceil(x - 0.5),  x < 0

# 导数 (几乎处处):
#   d(round(x))/dx = 0   (除了 x.5 的分段点处不存在)
```

**关键不是 "导数为 0"，而是 "导数为 0 意味着梯度消失"。** 在链式法则中：

```python
# 假设 loss = MSE(y, target), y = round(w * x)
# ∂loss/∂w = ∂loss/∂y × ∂y/∂round × ∂round/∂(w*x) × x
#                       ↑ 这一项 = 0 → 整个梯度链断了
```

这意味着 weight 收不到任何来自量化噪声的反馈——不知道该往哪调。这就是为什么需要 **STE（Straight-Through Estimator）**。

### 1.2 PyTorch 的 STE：不是"绕过"而是"在 backward 中替换"

PyTorch 对 `torch.round()` 的 STE 实现在 C++ 层。回顾 Stage 1 §8 中讲过的 `fake_quantize_per_tensor_affine` 反向内核：

```cpp
// C++ 反向内核 (简化版, 来自 aten/src/ATen/native/quantized/cpu/...)
float inv_scale = 1.0f / scale;
cpu_kernel(iter, [&](float x, float dy) -> float {
    int64_t xq = static_cast<int64_t>(std::nearbyint(x * inv_scale + zero_point));
    //                                 ↑ 重算量化值 (和 forward 一样)
    return dy * (xq >= quant_min && xq <= quant_max);
    //     ↑ STE: 在范围内, 梯度直通 (乘以 1); 在范围外, 梯度截断 (乘以 0)
    //     没有乘以 ∂round/∂x = 0 这一步!
});
```

**核心机制**：`fake_quantize_per_tensor_affine` 的前向计算了 `round()`，但反向**故意不包含 `round()` 的导数**。反向只做两件事：

1. 判断输入是否在量化范围内（`in_range` mask）
2. 在范围内的：`grad_input = grad_output × 1`（STE 直通）
3. 在范围外的：`grad_input = 0`（clip 截断梯度）

**但注意：这个 C++ 内核只计算了对输入 `x` 的梯度。对 `scale` 和 `zero_point` 的梯度——它是算不了的。** 因为 `scale` 和 `zero_point` 在 PyTorch 的 FakeQuantize 中是 `register_buffer`——它们根本不是计算图的叶子节点，autograd 引擎不会为它们分配梯度存储。

**这就是为什么 LSQ 必须手写 `autograd.Function`。** 因为标准 `torch.fake_quantize_per_tensor_affine` 的 backward 里没有 `grad_scale` 的计算路径。

### 1.3 `torch.autograd.Function` 的完整生命周期

要理解 LSQ 的实现，必须先理解 `autograd.Function` 的内部机制：

```python
import torch

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, param):
        # ctx: "上下文背包" — 你想在 backward 里用的任何东西都塞到这里
        y = some_computation(x, param)
        ctx.save_for_backward(x, param)  # ← 存到 ctx 的 saved_tensors 列表
        # 注意: 只有 Tensor 能 save，Python 标量/布尔值需要存为 ctx 属性
        ctx.some_info = "hello"          # ← Python 对象存为属性
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # backward 在 loss.backward() 时被自动调用 — 不是手动调!
        x, param = ctx.saved_tensors     # ← 从 forward 存的背包里取回
        # 注意: ctx.saved_tensors 是一个 tuple，顺序 = save_for_backward 的顺序

        grad_x = compute_grad_x(grad_output, x, param)
        grad_param = compute_grad_param(grad_output, x, param)

        # 返回值 = forward 的每个非 ctx 参数的梯度 (None = 不需要梯度)
        return grad_x, grad_param

# 使用:
y = MyFunction.apply(x, param)  # ← 调用 forward
loss = MSE(y, target)
loss.backward()                  # ← 自动调用 backward
```

**一次完整的 autograd.Function 调用链**：

```
forward 阶段 (y = MyFunction.apply(x, param)):
  1. autograd 引擎检测到 apply() 被调用
  2. autograd 引擎调用 MyFunction.forward(ctx, x, param)
     → ctx 是一个新创建的上下文对象
     → x, param 的 grad_fn 被记录（如果它们需要梯度）
  3. forward 返回 y
  4. autograd 引擎把 MyFunction 实例注册为 y 的 grad_fn
     → y.grad_fn = <MyFunction object>
     → y.grad_fn.saved_tensors = (x, param)  (如果 forward 调了 save_for_backward)

backward 阶段 (loss.backward()):
  1. 梯度沿图回溯, 到达 y.grad_fn = <MyFunction object>
  2. autograd 引擎调用 MyFunction.backward(ctx, grad_output)
     → ctx.saved_tensors = (x, param)  (从 forward 恢复)
     → grad_output = ∂loss/∂y
  3. backward 返回 (grad_x, grad_param)
  4. autograd 引擎把 grad_x 传给 x 的上游, grad_param 传给 param 的上游
```

### 1.4 `ctx.save_for_backward` 的 tensor 生命周期

这是一个容易被忽略但很关键的细节：

```python
# forward 里:
ctx.save_for_backward(x, scale, x_scaled, x_clamped, x_rounded)
# 这 5 个 tensor 被包装到一个 tuple, 存入 ctx

# backward 里:
x, scale, x_scaled, x_clamped, x_rounded = ctx.saved_tensors
# 取回来 — 和前向时的值完全一样 (不会被任何中间操作修改)

# 关键规则:
# 1. save_for_backward 的 tensor 在 forward 返回后被"打包"存储
#    → 不再是原始计算图中的活跃 tensor
#    → 它们的 grad 不会被追踪
#    → 它们不会被任何 optimizer 更新
#    → 它们只是"快照" — forward 时的值被冻住了

# 2. backward 中从 ctx.saved_tensors 取回时
#    → 这些 tensor 的 .grad 不会自动填充 (它们不是叶子节点)
#    → 你只能读取它们的值, 不能修改
#    → 如果你在 backward 里修改了它们, autograd 不会检测到 (不会报错但结果错了)

# 3. 为什么 forward 里算一遍 x_scaled/x_clamped, backward 里又要算相似的东西?
#    因为 backward 只能拿到 grad_output 和 saved_tensors
#    backward 没有 forward 的完整计算上下文 — 它必须"重新推导"一些中间量
```

### 1.5 在你的 LSQFunction.forward 里存了哪些东西、为什么

对照上面的理解，看 LSQ 的 `forward` 为什么存了这 5 个 tensor：

```python
ctx.save_for_backward(x, scale, x_scaled, x_clamped, x_rounded)
#  ↑                    ↑      ↑          ↑           ↑
# 原始输入 (FP32)   可学参数   v/s        clamp后     round后
# 用途: backward    用途:    用途:       用途:       用途:
# 中需要 x/scale    需要它   需要它判断   不需要      需要它计算
# 来算 -v/s+round  来算     是否在量化   (但存着     ∂v̂/∂s =
#                  ∂v̂/∂s   范围内      方便调试)    -v/s+round(v̄)
#                                         ↑ 注意: x_clamped 在 LSQ 的 backward 中
#                                            其实没有被使用 — 但存着没关系
#                                            可以用 in_range 判断替代
```

### 1.6 STE 的硬伤 — 为什么只解决了一半问题

回顾 `fake_quantize_per_tensor_affine` 的 C++ 反向内核：

```cpp
// 这个内核为 **输入 x** 计算了 STE 梯度
// 但 scale 和 zero_point 的梯度 = 不存在
// 因为它们在 FakeQuantize 中是 register_buffer — 不在计算图中

// LSQ 的任务: 在自定义 autograd.Function 的 backward 里,
// 不仅要实现 STE (对 x), 还要实现 scale 的梯度 (对 s)
```

**LSQ 解决的问题**：标准 FakeQuantize 的 backward 里，`grad_scale = None`。LSQ 手写了 backward，使得 `grad_scale = (-v/s + round(v̄)) × grad_output` —— 把 scale 从 "梯度黑洞" 变成了 "可学习的参数"。

---

## 2. LSQ 梯度推导：公式 (6) 的逐行手推

> 这一节的推导和原始 LSQ 论文一致。配合具体的数值示例理解每一步。

### 2.1 设三个中间变量

```
v_n = v / s                          — 缩放后的连续值
v̄ = clamp(v_n, Q_min, Q_max)         — clamp 后的值 (仍在连续域)
v̂ = round(v̄) × s                     — 量化后的输出
```

**注意**：LSQ 的量化公式 `v̂ = round(clamp(v/s, Qmin, Qmax)) × s` 和标准 FakeQuantize 的 `round(v/s + Z).clamp(Qmin, Qmax) × s` 有微妙区别。LSQ 假设 `Z=0`（对称量化），且 clamp 在 round 之前（不是之后）——这个顺序对梯度推导至关重要。

### 2.2 乘法法则展开

对 `v̂ = round(v̄) × s` 求 ∂v̂/∂s：

```
∂v̂/∂s = ∂(round(v̄) × s) / ∂s
       = round(v̄) × ∂s/∂s + s × ∂round(v̄)/∂s         (乘法法则)
       = round(v̄) + s × ∂round(v̄)/∂s                  (1)
```

第一项 `round(v̄)` 是直接的——scale 乘以量化值 round(v̄)，scale 变化 1，输出变化 round(v̄)。

第二项是链式展开：

```
∂round(v̄)/∂s = ∂round(v̄)/∂v̄ × ∂v̄/∂s
```

对 `∂round(v̄)/∂v̄`：STE 规定 = 1（梯度直通 round）。

所以：

```
∂round(v̄)/∂s = 1 × ∂v̄/∂s = ∂v̄/∂s                      (2)
```

### 2.3 分三种情况

`v̄ = clamp(v/s, Q_min, Q_max)` 的梯度依赖于 `v/s` 是否落在 [Q_min, Q_max] 内：

```
情况 A: Q_min < v/s < Q_max（在量化范围内，没被 clip）
  v̄ = v/s
  ∂v̄/∂s = ∂(v/s)/∂s = -v/s²

情况 B: v/s ≤ Q_min（被 clip 到下界）
  v̄ = Q_min (常量)
  ∂v̄/∂s = 0

情况 C: v/s ≥ Q_max（被 clip 到上界）
  v̄ = Q_max (常量)
  ∂v̄/∂s = 0
```

### 2.4 代回 (1)

```
情况 A (在范围内):
  ∂v̂/∂s = round(v̄) + s × (-v/s²)
         = round(v̄) - v/s
         = -v/s + round(v̄)                             ← ★ LSQ 公式 (6)!

情况 B (被 clip 到下界):
  ∂v̂/∂s = round(Q_min) + s × 0
         = Q_min

情况 C (被 clip 到上界):
  ∂v̂/∂s = round(Q_max) + s × 0
         = Q_max
```

### 2.5 用一个具体数字穿过推导

```python
# 假设一个具体元素: v = 2.5, scale = 0.02, Q_min=-128, Q_max=127

# 前向:
v_n = 2.5 / 0.02 = 125.0
v̄ = clamp(125.0, -128, 127) = 125.0   # 在范围内 — 情况 A
v̂ = round(125.0) × 0.02 = 2.50

# 反向 (scale 梯度):
∂v̂/∂s = -v/s + round(v̄) = -125.0 + round(125.0) = -125.0 + 125.0 = 0.0
# 梯度 = 0 → scale 不用调整 (因为 v=2.5 恰好对齐到量化网格上, scale 完美)

# --------

# 另一个元素: v = 2.503, scale = 0.02
v_n = 2.503 / 0.02 = 125.15
v̄ = clamp(125.15, -128, 127) = 125.15
v̂ = round(125.15) × 0.02 = 125.0 × 0.02 = 2.50
# 量化误差: 2.503 - 2.50 = 0.003

# 反向:
∂v̂/∂s = -125.15 + 125.0 = -0.15
# 梯度为负 → scale 应该减小 (让网格更密, 减少量化误差)
# 逻辑: v=2.503 被量化成了 2.50 — 网格太粗了 → 减小 scale → 更多等级

# --------

# 被 clip 的元素: v = 3.0, scale = 0.02
v_n = 3.0 / 0.02 = 150.0
v̄ = clamp(150.0, -128, 127) = 127.0    # 被 clip! — 情况 C
v̂ = round(127.0) × 0.02 = 2.54

# 反向:
∂v̂/∂s = Q_max = 127
# 梯度为正且很大 → scale 应该增大!
# 逻辑: v=3.0 被 clip 到了 2.54 — scale 太小导致范围不够 → 增大 scale → 更大范围
```

### 2.6 直觉解读

**公式 (6) 的梯度 `(-v/s + round(v̄))` 自动执行两件事**：

1. **减少舍入误差**：如果 `-v/s + round(v̄)` 不为零 → scale 向"让连续值恰好落在量化网格上"的方向调整
2. **减少 clip 误差**：如果 `v/s` 超出 [Q_min, Q_max] → 梯度变成 Q_min 或 Q_max（符号 + 量级）→ 自动扩大或缩小范围

**不需要额外的 loss 项**。这些优化机制被编码在 backward 的梯度计算中——SGD 自动驱动 scale 向最优值收敛。

---

## 3. Gradient Scaling：公式 (13) 的完整推导

### 3.1 问题的数学表述

LSQ 中每个 scale 影响 `N` 个元素（per-tensor: N = 所有权重；per-channel: N = 每个通道的权重）。链式法则把 `N` 条梯度路径汇到 1 个 scale 参数上：

```
∂L/∂s = Σ_i (∂L/∂v̂_i × ∂v̂_i/∂s)
       = Σ_i (grad_output_i × (-v_i/s + round(v̄_i)))
       = N 个项的求和
```

**问题**：`∂L/∂s` 的量级天然比 `∂L/∂w_i` 大 √N 倍（N 个独立梯度的求和，方差相加）。不加控制 → scale 每次更新的步长是 weight 的几十倍 → 振荡或发散。

### 3.2 推导：为什么是 `1 / √(N × Q_P)`

我们需要一个缩放因子 g，使得 `g × ∂L/∂s` 的量级和 `∂L/∂w_i` 相当。

**Step 1**：估计 `∂L/∂w_i` 的量级。

weight 的梯度来自 STE 穿过量化层后的累积。在稳定训练中，每层 weight 梯度的 RMS ≈ `1/√N_w`（N_w 是该层的 weight 总数）——这是一个经验规律，源自初始化理论（Glorot/He 初始化使得前向和反向的方差保持恒定）。

**Step 2**：估计 `∂L/∂s` 的量级。

`∂L/∂s = Σ_i grad_output_i × (-v_i/s + round(v̄_i))`。`(-v_i/s + round(v̄_i))` 的 absolute 接近 Q_P（~127 for INT8）。所以：

```
|∂L/∂s| ≈ Q_P × |Σ_i grad_output_i|
         ≈ Q_P × √N × RMS(grad_output)
```

**Step 3**：求 g 使得 `g × |∂L/∂s| ≈ |∂L/∂w_i|`。

```
g × Q_P × √N × RMS(grad) ≈ RMS(grad)
g ≈ 1 / (Q_P × √N) = 1 / √(N × Q_P²)

# LSQ 论文里用的是 Q_P (quantization range) 而非 Q_P²
# g = 1 / √(N × Q_P)   ← 公式 (13)
# 这里的 Q_P = max(|Q_min|, |Q_max|), 对于 INT8 对称 = 127
```

**Step 4**：具体算例验证。

```
Conv2d(64, 128, 3×3), per-channel scale, 8-bit symmetric:
  N = 128 × 64 × 9 = 73728  (每个输出通道的 weight 数)
  Q_P = 127
  g = 1 / √(73728 × 127) = 1 / √9363456 ≈ 1 / 3060 ≈ 0.00033
```

**含意**：这个 layer 的 scale 梯度被乘以 0.00033 → scale 的每次更新 ≈ weight 更新的 1/3000 → 和 weight 的量级对齐。

### 3.3 实验验证：不用 Gradient Scaling 会怎样

```python
def compare_gradient_scaling():
    """对比 有 vs 无 Gradient Scaling 的 scale 学习过程"""
    # 模拟一个简单场景: 1 个 LSQ 层 + MSE loss
    x = torch.randn(64, 256) * 2.0
    target = torch.randn(64, 128)

    results = {}
    for use_gs in [True, False]:
        lsq = LSQQuantizer(n_bits=4, use_gradient_scaling=use_gs)
        lsq.init_from_data(x @ torch.randn(256, 128))

        opt = torch.optim.SGD([lsq.scale], lr=0.01)
        scale_history = []

        for step in range(500):
            opt.zero_grad()
            w = torch.randn(128, 256, requires_grad=True)
            w_q = lsq(w)
            loss = ((w_q @ x.T).T - target).pow(2).mean()
            loss.backward()
            opt.step()
            scale_history.append(lsq.scale.item())

        results[f"GS={'ON' if use_gs else 'OFF'}"] = scale_history

    # 画图:
    # GS=OFF: scale 在 step 20 左右爆炸 (NaN)
    # GS=ON:  scale 平滑收敛到稳定值
    return results
```

**期望结果**：不用 Gradient Scaling 时，scale 的更新步长是 weight 的 ~3000 倍（对于上面的 Conv2d 例子），在几十步 SGD 后 scale 就发散到 NaN。用了之后，scale 和 weight 在同一量级上平滑收敛。

### 3.4 per-layer vs per-channel 的 Gradient Scaling

```
per-layer scale:  N = 层内所有 weight 的总数
  例: Linear(256, 128) → N = 256 × 128 = 32768
  g = 1 / √(32768 × 127) ≈ 1 / 2040 ≈ 0.00049

per-channel scale: N = 每个输出通道的 weight 数
  例: Conv2d(64, 128, 3, 3) → N = 64 × 3 × 3 = 576 (per channel)
  g = 1 / √(576 × 127) ≈ 1 / 270 ≈ 0.0037

→ per-channel 的 g 比 per-layer 大 ~7.5 倍
→ per-channel scale 更新更快 — 合理的, 因为它影响的元素少, 每个更新更"局部"
```

---

## 4. 从零实现 LSQ：完整的 autograd.Function + 初始化策略

### 4.1 LSQFunction — 完整实现

```python
import torch
import torch.nn as nn

class LSQFunction(torch.autograd.Function):
    """LSQ 的自定义 autograd 函数 — forward 模拟量化, backward 手动提供 scale 梯度"""

    @staticmethod
    def forward(ctx, x, scale, n_bits, symmetric):
        """
        Args:
            x:        输入 tensor (FP32)
            scale:    量化 scale (FP32 标量, 可学习)
            n_bits:   量化比特数
            symmetric: True = 对称 (zero_point=0), False = 非对称
        Returns:
            x_quant:  fake-quantized tensor (FP32)
        """
        if symmetric:
            qmax = 2 ** (n_bits - 1) - 1
            qmin = -qmax
        else:
            qmax = 2 ** n_bits - 1
            qmin = 0

        # 前向: 和标准 FakeQuantize 完全一样
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clamped)
        x_quant = x_rounded * scale

        # 保存 backward 需要的 tensor
        ctx.save_for_backward(x, scale, x_scaled, x_clamped, x_rounded)
        ctx.qmin, ctx.qmax = qmin, qmax

        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: ∂L/∂x_quant — 上游传下来的梯度

        Returns:
            grad_x:     ∂L/∂x      (STE)
            grad_scale: ∂L/∂scale  (LSQ 公式 6 + 13)
            None, None  (n_bits, symmetric 不需要梯度)
        """
        x, scale, x_scaled, x_clamped, x_rounded = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax

        # ===== 1. 对 x 的梯度: STE =====
        in_range = (x_scaled >= qmin) & (x_scaled <= qmax)
        grad_x = torch.where(in_range, grad_output,
                             torch.zeros_like(grad_output))

        # ===== 2. 对 scale 的梯度: LSQ 公式 (6) =====
        # 情况 A (在范围内): -v/s + round(v̄) = -x_scaled + x_rounded
        # 情况 B (clip 下界): Q_min
        # 情况 C (clip 上界): Q_max
        grad_s = -x_scaled + x_rounded                     # 情况 A
        grad_s = torch.where(in_range, grad_s,             #    ←
                   torch.where(x_scaled < qmin,            #     情况 B/C
                               float(qmin), float(qmax)))  #

        # grad_scale = Σ_i (grad_output_i × grad_s_i)
        grad_scale = (grad_output * grad_s).sum()

        # ===== 3. Gradient Scaling: 公式 (13) =====
        # N = x.numel(), Q_P = max(|qmin|, |qmax|)
        g = 1.0 / (x.numel() * max(abs(qmin), abs(qmax))) ** 0.5
        grad_scale = grad_scale * g

        return grad_x, grad_scale, None, None
```

### 4.2 LSQQuantizer — nn.Module 封装

```python
class LSQQuantizer(nn.Module):
    """LSQ 量化器 — 可直接替换 PyTorch 的 FakeQuantize"""

    def __init__(self, n_bits=4, symmetric=True, use_gradient_scaling=True):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.use_gradient_scaling = use_gradient_scaling

        # ★ 核心: scale 是 nn.Parameter — optimizer 会更新它!
        self.scale = nn.Parameter(torch.tensor(1.0))
        # 对比 PyTorch FakeQuantize:
        #   self.register_buffer("scale", torch.tensor(1.0))
        #   ↑ buffer 不参与梯度, optimizer 不更新

    def init_from_data(self, x):
        """用输入数据的统计量初始化 scale — 关键: 好的初始值加速收敛"""
        with torch.no_grad():
            qmax = 2 ** (self.n_bits - 1) - 1
            abs_max = x.detach().abs().max()
            # 初始 scale = max(|x|) / Q_max × 0.8
            # 0.8: 故意让初始 scale 偏小, 而不是偏大
            # 原因: bias 向"更密"的方向 — 宁可 clip 多一点初始值,
            #       也不要让 scale 太大导致量化网格太粗
            #       因为 clip 的元素梯度 = Q_min/Q_max (大!), 会快速拉大 scale
            #       而网格太粗 → 所有元素梯度 ≈ 0 (舍入误差小但累积大) → 调不动
            self.scale.data = abs_max / qmax * 0.8

    def forward(self, x):
        return LSQFunction.apply(x, self.scale, self.n_bits, self.symmetric)


class FixedScaleQuantizer(nn.Module):
    """固定 scale 量化器 — Stage 1.5 用的那种, 用于对比实验"""

    def __init__(self, n_bits=4, symmetric=True):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.register_buffer("scale", torch.tensor(1.0))  # ← buffer!

    def init_from_data(self, x):
        qmax = 2 ** (self.n_bits - 1) - 1
        self.scale.data = x.detach().abs().max() / qmax

    def forward(self, x):
        qmax = 2 ** (self.n_bits - 1) - 1
        qmin = -qmax if self.symmetric else 0
        x_q = torch.round(x / self.scale).clamp(qmin, qmax)
        return x_q * self.scale
        # 反向: STE 自动处理 (PyTorch 对 round() 内建 STE)
        # 但 scale 没有梯度 — 固定的!
```

### 4.3 初始化策略详解

LSQ 的初始化对收敛速度和最终精度影响很大。三种常见策略：

```python
# 策略 A: MinMax 初始化 (LSQ 论文推荐)
#   s_init = max(|x|) / Q_max × η,  η ∈ [0.5, 1.0]
#   优点: 简单, 对大多数层工作良好
#   缺点: 对 outlier 敏感
def init_minmax(x, n_bits, eta=0.8):
    qmax = 2 ** (n_bits - 1) - 1
    return x.detach().abs().max() / qmax * eta

# 策略 B: MSE 网格搜索初始化
#   在 [0.1×s_max, s_max] 范围内搜索最小化 MSE 的 scale
#   优点: "最优"的初始 scale (在 MSE 意义上)
#   缺点: 需要额外的前向 pass
def init_mse_search(x, n_bits, n_bins=100):
    qmax = 2 ** (n_bits - 1) - 1
    s_max = x.detach().abs().max() / qmax

    best_mse = float('inf')
    best_s = s_max
    for s in torch.linspace(s_max * 0.1, s_max, n_bins):
        x_q = torch.round(x / s).clamp(-qmax, qmax) * s
        mse = ((x - x_q) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_s = mse, s.item()
    return best_s

# 策略 C: Percentile 初始化 (大模型推荐)
#   用 99.9% 分位数代替 max, 抵抗 outlier
#   优点: LLM 的 outlier channel 不会被几个极端值主导
def init_percentile(x, n_bits, pct=0.999, eta=0.8):
    qmax = 2 ** (n_bits - 1) - 1
    x_sorted = x.detach().abs().flatten().sort().values
    abs_max_pct = x_sorted[int(len(x_sorted) * pct)]
    return abs_max_pct / qmax * eta
```

**为什么 `η = 0.8`（不是 1.0）？**

如果 `η = 1.0`，初始 scale 让 |x_max| 恰好对齐到量化网格的边界。问题是：当一个值正巧在边界时，任何微小的 weight 更新都可能把它推到 clip 区外——clip error 来得太快。

`η = 0.8` 让 scale 故意偏小 → 初始有一些 clip，但 clip 区元素的梯度 = Q_max（很大！）→ scale 被快速拉大到合适值。而如果初始 scale 偏大（η > 1.0），所有元素都在量化范围内 → gradient_s ≈ 0（因为 -v/s + round(v̄) 对所有未 clip 元素都接近 0）→ scale 几乎不动 → QAT 从头到尾在用"过粗"的网格。

**经验规律**：η 在 0.6-0.9 之间通常都能工作。低于 0.5 会导致太多 clip 梯度涌入 scale（震荡），高于 1.0 会导致 scale 几乎不动。

---

## 5. LSQ+：zero_point 也可学习

### 5.1 为什么需要 LSQ+？

LSQ 假设对称量化（`Z = 0`）。这对权重足够好（weight 分布近似对称），但对 **ReLU 后的激活值** 浪费了大量量化范围——ReLU 后的值全 ≥ 0，对称量化的负半轴（-128 到 0）完全浪费。

LSQ+（高通的扩展）让 `zero_point` 也是 `nn.Parameter` —— 对激活值做非对称量化时，Z 自动向偏移分布的中心位置（而不是强制为 0）。

### 5.2 zero_point 的梯度推导

LSQ+ 的量化公式：

```
v̂ = round(clamp(v/s + Z, Q_min, Q_max)) × s
```

对 scale `s` 的梯度和 LSQ 一样。对 zero_point `Z` 的梯度：

```
∂v̂/∂Z = ∂(round(v̄) × s) / ∂Z
       = round(v̄) × ∂s/∂Z + s × ∂round(v̄)/∂Z    (乘法法则, ∂s/∂Z = 0)
       = s × ∂round(v̄)/∂Z
       = s × ∂round(v̄)/∂v̄ × ∂v̄/∂Z              (链式展开)
       = s × 1 × ∂v̄/∂Z                            (STE)
       = s × ∂v̄/∂Z                                (3)
```

`v̄ = clamp(v/s + Z, Q_min, Q_max)`，Z 出现在 clamp 内部：

```
情况 A (在范围内): v̄ = v/s + Z → ∂v̄/∂Z = 1 → ∂v̂/∂Z = s
情况 B (clip 下界): v̄ = Q_min → ∂v̄/∂Z = 0 → ∂v̂/∂Z = 0
情况 C (clip 上界): v̄ = Q_max → ∂v̄/∂Z = 0 → ∂v̂/∂Z = 0
```

**直觉解读**：Z 的梯度在范围内 = `s`（scale 的量级），在范围外 = 0。这意味着 Z 只在元素没有被 clip 时接收梯度——和 scale 的梯度互补。Scale 的梯度在 clip 区最大（`Q_min/Q_max`），被 clip 区的 scale 调整由 scale 负责，Z 不插手。

### 5.3 LSQ+ 的完整实现

```python
class LSQPlusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, n_bits):
        qmax = 2 ** n_bits - 1
        qmin = 0
        # 非对称量化: x_scaled = x/s + Z
        x_scaled = x / scale + zero_point
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clamped)
        x_quant = (x_rounded - zero_point) * scale

        ctx.save_for_backward(x, scale, zero_point, x_scaled, x_clamped, x_rounded)
        ctx.qmin, ctx.qmax = qmin, qmax
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zp, x_scaled, x_clamped, x_rounded = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax

        in_range = (x_scaled >= qmin) & (x_scaled <= qmax)

        # 对 x: STE
        grad_x = torch.where(in_range, grad_output, torch.zeros_like(grad_output))

        # 对 scale: 同 LSQ 公式 (6) — 但考虑了 zero_point
        #   ∂v̂/∂s = -v/s² × s × STE_mask + round(v̄ - zp) (展开后)
        #   简化: 和 LSQ 的结果几乎一样 — 因为 zp 的引入不改变 scale 梯度的核心形式
        grad_s_raw = -x_scaled + x_rounded
        grad_s_raw = torch.where(in_range, grad_s_raw,
                       torch.where(x_scaled < qmin, float(qmin - zp),
                                   float(qmax - zp)))
        grad_scale = (grad_output * grad_s_raw).sum()
        g = 1.0 / (x.numel() * max(abs(qmin), abs(qmax))) ** 0.5
        grad_scale = grad_scale * g

        # 对 zero_point: LSQ+ 新增的梯度
        #   在范围内: ∂v̂/∂Z = scale  (来自推导)
        #   在范围外: ∂v̂/∂Z = 0
        grad_zp = torch.where(in_range, scale,
                              torch.zeros_like(x_scaled))
        grad_zero_point = (grad_output * grad_zp).sum()
        # zero_point 也有自己的 Gradient Scaling
        g_zp = 1.0 / (x.numel() * max(abs(qmin), abs(qmax))) ** 0.5
        grad_zero_point = grad_zero_point * g_zp

        return grad_x, grad_scale, grad_zero_point, None


class LSQPlusQuantizer(nn.Module):
    """LSQ+ 量化器 — scale 和 zero_point 都可学习"""

    def __init__(self, n_bits=4):
        super().__init__()
        self.n_bits = n_bits
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def init_from_data(self, x):
        with torch.no_grad():
            qmax = 2 ** self.n_bits - 1
            x_min = x.detach().min()
            x_max = x.detach().max()
            self.scale.data = (x_max - x_min) / qmax * 0.8
            self.zero_point.data = torch.round(-x_min / self.scale.data).clamp(0, qmax)

    def forward(self, x):
        return LSQPlusFunction.apply(x, self.scale, self.zero_point, self.n_bits)
```

### 5.4 LSQ vs LSQ+：什么时候用哪个

| 场景 | 推荐 | 原因 |
|------|------|------|
| 权重量化 | LSQ (对称) | 权重分布近似对称, Z ≈ 0 是合理的 |
| ReLU 后的激活 | LSQ+ (非对称) | 全非负, 对称浪费 50% 范围 |
| GELU/SiLU 后的激活 | LSQ+ | 有轻微负值, 但分布不对称 |
| LLM 的激活 | LSQ+ per-channel | LLM 激活有 outlier, per-channel + 非对称是最优配置 |
| 统一方案 (简单) | 全用 LSQ+ | 多一个参数但换来鲁棒性, 不影响推理速度 |

---

## 6. LSQ 在 PT2E 中的集成：自定义 Quantizer

Stage 1 花了 1677 行讲 PT2E 管线。现在把 LSQ 集成进去——让你能在 `prepare_qat_pt2e` 中使用 LSQ 替代标准 FakeQuantize。

### 6.1 核心思路

```
PT2E 的标准 FakeQuantize 路径:
  QuantizationSpec → MinMaxObserver → FakeQuantize(FakeQuantizeBase, MinMaxObserver)
                      scale 是 buffer      scale 是 buffer (继承自 MinMaxObserver)

LSQ 路径:
  QuantizationSpec + custom observer → LSQFakeQuantize
                                         scale 是 nn.Parameter
                                         backward 用 LSQ 公式 (6)
```

### 6.2 实现 LSQFakeQuantize

```python
class LSQFakeQuantize(nn.Module):
    """LSQ 版的 FakeQuantize — 可替代 PyTorch 的 FakeQuantize, 用在 PT2E 管线中"""

    def __init__(self, n_bits=4, symmetric=True, use_gradient_scaling=True,
                 quant_min=-128, quant_max=127, dtype=torch.int8,
                 qscheme=torch.per_tensor_symmetric):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.use_gradient_scaling = use_gradient_scaling

        # ★ 关键: scale 是 nn.Parameter — LSQ 的核心改动
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 标准 FakeQuantize 的 metadata (PT2E 管线需要这些属性)
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.activation_post_process = self  # 兼容 PT2E 的引用

        # 状态标志 (兼容 PT2E 的 observer/fake_quant 切换)
        self.register_buffer('observer_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))

    def init_from_data(self, x):
        qmax = 2 ** (self.n_bits - 1) - 1
        self.scale.data = x.detach().abs().max() / qmax * 0.8

    def forward(self, x):
        if self.observer_enabled[0] == 1:
            # Observer 阶段: 用 LSQ 的初始化策略更新 scale
            self.init_from_data(x)

        if self.fake_quant_enabled[0] == 1:
            return LSQFunction.apply(x, self.scale, self.n_bits, self.symmetric)
        return x

    def calculate_qparams(self):
        """用于 convert_pt2e 读取最终的 scale/zp"""
        return self.scale, torch.zeros_like(self.scale)
```

### 6.3 注册为自定义 Quantizer

```python
from torch.ao.quantization.quantizer import Quantizer, QuantizationSpec

class LSQQuantizer(Quantizer):
    """自定义 PT2E Quantizer — 使用 LSQ 替代标准 FakeQuantize"""

    def __init__(self, n_bits=4):
        super().__init__()
        self.n_bits = n_bits
        self.global_config = None

    def set_global(self, config):
        self.global_config = config

    def annotate(self, model):
        """标注图中需要 LSQ 量化的边"""
        for node in model.graph.nodes:
            if node.op != 'call_function':
                continue

            # Conv/Linear: weight + input activation
            if node.target in [torch.ops.aten.conv2d.default,
                               torch.ops.aten.linear.default]:
                # input activation → LSQ per-tensor
                self._annotate_edge(node, 0, per_channel=False)
                # weight → LSQ per-channel
                self._annotate_edge(node, 1, per_channel=True)

            # Add: 两个输入共享 LSQ
            elif node.target == torch.ops.aten.add.Tensor:
                self._annotate_edge(node, 0, per_channel=False)
                self._annotate_edge(node, 1, per_channel=False,
                                    shared_with=(node, node.args[0]))

    def _annotate_edge(self, node, idx, per_channel=False, shared_with=None):
        if shared_with:
            self.annotations[(node, idx)] = SharedQuantizationSpec(
                edge_or_node=shared_with)
        else:
            self.annotations[(node, idx)] = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=(torch.per_channel_symmetric if per_channel
                         else torch.per_tensor_symmetric),
                observer_or_fake_quant_ctr=LSQFakeQuantize.with_args(
                    n_bits=self.n_bits,
                    symmetric=True,
                ),
                is_dynamic=False,
            )

    def validate(self, model):
        pass  # 简化实现
```

### 6.4 完整使用流程

```python
# Step 1: Export
gm = capture_pre_autograd_graph(model, example_input)

# Step 2: LSQ Quantizer
quantizer = LSQQuantizer(n_bits=4)
quantizer.set_global(None)

# Step 3: Prepare (用 LSQ QAT)
prepared = prepare_qat_pt2e(gm, quantizer)
# ↑ prepare 里 LSQFakeQuantize 替代了标准 FakeQuantize
# ↑ LSQFakeQuantize.scale 是 nn.Parameter — optimizer 能看到

# Step 4: QAT Training
opt = torch.optim.SGD(prepared.parameters(), lr=1e-4)
# ↑ ★ LSQ 的 scale (nn.Parameter) 和 weight
#   一起在 prepared.parameters() 里!
#   optimizer 自动更新两者

for ep in range(20):
    for x, y in train_loader:
        loss = criterion(prepared(x), y)
        loss.backward()  # LSQ 的 backward 在这里执行 — 提供 scale 的梯度
        opt.step()       # scale 被更新!

# Step 5: Convert
quantized = convert_pt2e(prepared.eval())
```

---

## 7. 完整训练实验：从代码到可视化

### 7.1 LSQ vs Fixed QAT 完整对比实验

```python
def full_lsq_experiment(model_fn, train_loader, val_loader,
                        bit_widths=[8, 6, 4, 3, 2]):
    """完整的 LSQ vs Fixed QAT 对比实验"""
    results = {'FP32': None, 'Fixed QAT': {}, 'LSQ': {}}

    # FP32 baseline
    model_fp = model_fn()
    results['FP32'] = train_standard(model_fp, train_loader, val_loader, epochs=20)

    for bits in bit_widths:
        # Fixed QAT
        model_fixed = model_fn()
        _apply_fixed_qat(model_fixed, bits)
        results['Fixed QAT'][bits] = train_qat_manual(
            model_fixed, train_loader, val_loader,
            calib_epochs=3, qat_epochs=17
        )

        # LSQ
        model_lsq = model_fn()
        _apply_lsq(model_lsq, bits)
        results['LSQ'][bits] = train_qat_manual(
            model_lsq, train_loader, val_loader,
            calib_epochs=2, qat_epochs=18
            # LSQ 的 calib_epochs 可以更短 — scale 初始化足够后,
            #   后续的 scale 由梯度驱动, 不需要 Observer
        )

    return results
```

### 7.2 Scale 学习曲线可视化

```python
def track_scale_evolution(lsq_model, train_loader, n_epochs=20, track_every=10):
    """追踪 LSQ scale 在训练过程中的演变"""
    scale_history = defaultdict(list)

    for ep in range(n_epochs):
        for step, (x, y) in enumerate(train_loader):
            loss = criterion(lsq_model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if step % track_every == 0:
                for name, m in lsq_model.named_modules():
                    if isinstance(m, LSQQuantizer):
                        scale_history[name].append(m.scale.item())

    # 画图: 不同层 scale 的学习曲线
    plt.figure(figsize=(14, 5))
    for name, history in scale_history.items():
        plt.plot(history, label=name, alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Scale Value')
    plt.title('LSQ Scale Learning Curves — Per Layer')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lsq_scale_curves.png', dpi=150)
    # 期望观察: 不同层的 scale 收敛到不同值 (因为各层 weight 分布不同)
    #   → 证明 LSQ 的 scale 在"自适应"各层的分布
```

### 7.3 Gradient Scaling 消融实验

```python
def gs_ablation():
    """Gradient Scaling 有 vs 无 — 对 scale 收敛的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ['Without GS', 'With GS (η=1.0)', 'With GS (η=0.5)']

    for ax, (gs_enabled, eta) in zip(axes, [
        (False, 0.8), (True, 1.0), (True, 0.5)
    ]):
        # 用不同 GS 配置跑 100 steps, 记录 scale
        for seed in range(5):
            torch.manual_seed(seed)
            history = _run_scale_only_opt(use_gs=gs_enabled, gs_eta=eta)
            ax.plot(history, alpha=0.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Scale')
        ax.set_title(f'GS: {gs_enabled}, η={eta}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Scaling Ablation — Scale Trajectories')
    plt.tight_layout()
    plt.savefig('lsq_gs_ablation.png', dpi=150)
    # 期望:
    #   GS=OFF: scale 爆炸/振荡 (不同 seed 走不同路径, 全部偏离)
    #   GS=ON, η=1.0: 收敛但慢
    #   GS=ON, η=0.5: 收敛快, 但初始振动大 (梯度 scaled 太多)
```

---

## 8. 演进史：PACT → LSQ → LSQ+

理解一条算法演进链，比单独学一个算法更有用——因为你看的不是"一个公式"，而是"每个后续工作解决了前一个工作的什么问题"。

```
PACT (ICML 2018)
  │  问题: 固定量化范围 [0, Q_max] 在训练中可能不是最优的
  │  方案: 学 clippling threshold α — clamp(x, 0, α) 而非 clamp(x, 0, Q_max)
  │  局限: α 是 clip 上限, 但量化步长仍由 α/Q_max 隐含决定 — scale 和 clip 耦合
  │
  ▼
LSQ (ICLR 2020)
  │  问题: PACT 的 clip 和 scale 同源 (都由 α 决定), 两者应该独立优化
  │  方案: 直接学 scale (s), 不用 clamp(x, 0, α) + scale=α/Q_max 的耦合方式
  │  核心贡献: 公式 (6) — scale 的梯度不需要额外 loss, 自动从数据推导
  │          + 公式 (13) — Gradient Scaling 让 scale 和 weight 在同一量级
  │  局限: 对称量化 — zero_point 固定为 0
  │
  ▼
LSQ+ (2020, 高通)
  │  问题: LSQ 对 ReLU 激活用对称量化 → 浪费 50% 量化范围
  │  方案: zero_point 也是 nn.Parameter — 学最优的 Z
  │  核心贡献: Z 梯度推导 + LSQ per-channel 权重的扩展
  │
  ▼
后续工作 (非 LSQ 系列但相关):
  │  EfficientQAT (2024): 两阶段 QAT — 先 block-wise LSQ 初始化, 再端到端微调
  │  QLoRA (NeurIPS 2023): NF4 (NormalFloat 4-bit) — LSQ + 非均匀量化网格
  │  BitNet b1.58 (2024): 三值化 (-1, 0, 1) — LSQ 类方法的极致: 2 个 scale + 0
```

---

## 检验标准

- [ ] 能手写 `torch.autograd.Function` 的子类，能解释 `forward` 和 `backward` 的调用时机和 `ctx.save_for_backward` 的生命周期
- [ ] 能手推 LSQ 公式 (6): `∂v̂/∂s = -v/s + round(v̄)`，并能用具体数值验证
- [ ] 能解释为什么 clip 外的梯度是 `Q_min` 或 `Q_max`——而不是 0
- [ ] 能手推 Gradient Scaling 公式 (13): `g = 1 / √(N × Q_P)`，理解为什么是 `√` 而不是线性
- [ ] 能从零实现 `LSQFunction` + `LSQQuantizer`（含 Gradient Scaling）
- [ ] 能从零实现 `LSQPlusFunction` + `LSQPlusQuantizer`（含 zero_point 梯度）
- [ ] 能在 CIFAR-10/ResNet-20 上跑 LSQ vs Fixed QAT 的 8/6/4/3/2-bit 消融——证实 LSQ 把崩溃点推到 2-bit 以下
- [ ] 能画 LSQ scale 的学习曲线——不同层 scale 的收敛路径
- [ ] 能用自定义 `LSQQuantizer` 接入 PT2E 管线（`prepare_qat_pt2e`）
- [ ] 能画出 PACT → LSQ → LSQ+ 的演进图，说出每个工作解决了什么问题

---

> 💡 **学习建议**：LSQ 是整个学习路径的"分水岭"——攻克它之后，你会发现后续所有的量化算法都在做同一件事的变体：
> 把某个"固定的"量化参数变成"可学习的"。
>
> - **AIMET 的 Range Learning QAT** = LSQ 的工业实现（高通用的 LSQ+）
> - **QLoRA 的 NF4** = LSQ 在 4-bit 非均匀量化上的应用
> - **EfficientQAT** = LSQ 的两阶段版本（先 block-wise LSQ 让 scale 收敛，再端到端微调 weight）
>
> **攻克 LSQ 的标志不是"看懂了公式"，是"能从白纸开始写出完整实现 + 跑通消融实验 + 在 PT2E 管线中集成"。**
>
> Next: [Stage 3: PTQ 进阶算法](./Stage3_PTQ进阶算法.md)
