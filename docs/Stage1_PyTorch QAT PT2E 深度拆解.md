# Stage 1: PyTorch QAT 全流程深度拆解 — 以 PT2E 为准

> ⏱ 预计学习时间：25-35 小时 | 🎯 难度：⭐⭐
>
> **目标**：以 PT2E（PyTorch 2 Export）为锚点，一次性吃透"从裸模型到 QAT-ready 模型"的完整数据流——不满足于"这个 API 怎么调用"，而是知道**每一步对 tensor 做了什么、为不同算子插了什么 FQ、为什么要这样设计**。读完这章，你看 `prepare_qat_pt2e` 的源码能逐行解释它为什么这样写。

---

## 目录

1. [开篇：从 Stage 0 到 Stage 1——从数学到工程](#开篇)
2. [1. PT2E 完整管线：从裸模型到 QAT-ready 的五步走](#1-pt2e-完整管线)
3. [2. Step 1 深入：capture_pre_autograd_graph — nn.Module → ATen 图](#2-step-1-深入)
4. [3. Step 2 深入：Quantizer 标注 — 哪些边需要量化、哪些不需要](#3-step-2-深入)
5. [4. Step 3 深入：prepare_qat_pt2e — 在图中插入 FQ 节点](#4-step-3-深入)
6. [5. 算子级分派：不同 op 怎么插 FQ](#5-算子级分派)
7. [6. 完整 tensor 数据流追踪：一个 batch 穿过 QAT 图的全过程](#6-完整-tensor-数据流追踪)
8. [7. 四种 QuantizationSpec 完整拆解 🆕](#7-四种-quantizationspec-完整拆解)
9. [8. Tensor 级操作：从 FQ 节点到 C++ 内核 🆕](#8-tensor-级操作从-fq-节点到-c-内核)
10. [9. Observer / FakeQuantize 核心模块](#9-observer--fakequantize-核心模块)
11. [10. fuse_modules：Conv+BN 融合](#10-fuse_modulesconvbn-融合)
12. [11. QConfigMapping 与 Quantizer 体系](#11-qconfigmapping-与-quantizer-体系)
13. [12. 三种模式对比：Eager / FX / PT2E](#12-三种模式对比eager--fx--pt2e)
14. [13. QAT 训练实战技巧](#13-qat-训练实战技巧)
15. [14. 动手实验](#14-动手实验)
16. [检验标准](#检验标准)

---

## 开篇：从 Stage 0 到 Stage 1——从数学到工程

Stage 0 结束时的你：能手写一个 `QuantizedMLP`，理解 `q = round(r/S)` 和 STE，知道四种校准器的区别。但那是 **100 行的玩具代码**——一个 `QuantizedLinear` 类里硬编码了所有逻辑。

PyTorch 的量化模块把同样的事情拆成了上万行代码——不是因为它喜欢复杂，而是因为：

1. **不同的硬件后端要求不同的量化策略**。fbgemm（x86）和 qnnpack（ARM）的 INT8 Conv 内部实现不同，要求 scale/zero_point 的排列不同，支持的 op 也不同。
2. **不同的 op 类型需要不同的 FQ 插入逻辑**。Conv 的权重要做 per-channel 量化，Add 的两个输入要共享同一个 scale（否则加法没意义），ReLU 不需要量化（passthrough）。
3. **图级别的变换需要精确的依赖追踪**。一个 tensor 被三个下游 op 消费——插一个 FQ 还是三个？答案取决于这三个 op 的类型和共享策略。

这一章就带你逐层拆解——以 PT2E 为锚点（PyTorch 2.x 的推荐路径），从 capture → quantize → prepare → train → convert，每一步都看到 tensor 级别的变化。

---

## 1. PT2E 完整管线：从裸模型到 QAT-ready 的五步走

### 1.1 一张图看清全部

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    PT2E QAT Pipeline (5 Stages)                      │
  │                                                                      │
  │  Stage 1          Stage 2          Stage 3          Stage 4    Stage5│
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐  ┌──────┐│
  │  │Raw       │    │Quantizer │    │Prepare   │    │QAT    │  │Convert││
  │  │nn.Module ├───→│annotate  ├───→│insert FQ ├───→│Train  ├─→│INT8   ││
  │  │          │    │graph     │    │nodes     │    │Loop   │  │Model  ││
  │  └──────────┘    └──────────┘    └──────────┘    └───────┘  └──────┘│
  │                                                                      │
  │  capture_       XNNPACKQuantizer  prepare_qat_pt2e  standard  convert│
  │  pre_autograd   .set_global()                      train loop _pt2e  │
  │  _graph()       .set_module()                                       │
  │                                                                      │
  │  产出:           产出:              产出:            产出:     产出:  │
  │  ATen FX Graph   标注后的图         带 FQ 的图       QAT 权重  INT8 图│
  └─────────────────────────────────────────────────────────────────────┘
```

每一阶段有**一个明确的输入和一个明确的输出**。下面五节逐阶段拆解。

### 1.2 用一个 3-op 迷你模型贯穿全文

为了让你看清楚每一步对 tensor 做了什么，整章用一个具体的迷你模型：

```python
import torch
import torch.nn as nn

class MiniModel(nn.Module):
    """一个最简单的、但包含所有关键 op 类型的 QAT 测试模型"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)  # 有 weight 的 op
        self.bn = nn.BatchNorm2d(16)                  # 需要融合的 op
        self.relu = nn.ReLU()                          # passthrough op

    def forward(self, x):
        x = self.conv(x)   # 输入 x, weight W — 两个量化点
        x = self.bn(x)     # BN 输出 — 一个量化点（融合前）
        x = self.relu(x)   # 无损 passthrough — 无量化点
        return x

# 整个分析围绕这个模型的 PT2E 量化展开
model = MiniModel().train()
example_input = (torch.randn(1, 3, 32, 32),)
```

**在进入每一节之前，先问自己："现在图中的每个节点代表什么？tensor 经过它时发生了什么？"**

---

## 2. Step 1 深入：capture_pre_autograd_graph — nn.Module → ATen 图

### 2.1 你调用的是这个

```python
from torch._export import capture_pre_autograd_graph

# PyTorch 2.5+: 改用 torch.export.export_for_training
gm = capture_pre_autograd_graph(model, example_input)
```

`gm` 是一个 `torch.fx.GraphModule`——不是原来的 `nn.Module` 了。

### 2.2 到底发生了什么变化？

**原来的 `model.forward`**：调用 Python 对象的方法——`self.conv(x)` 创建一个 `Conv2d.__call__` 的执行上下文。

**capture 后的图**：`forward` 被重写为一串 `call_function` 节点——每个节点对应一个 **ATen 操作**（PyTorch C++ 后端的底层 op）。

```python
# 打印捕获后的图
print(gm.code)
```

你看到的会类似这样（简化版）：

```python
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, x):
    # arg0_1 = conv.weight, arg1_1 = conv.bias,
    # arg2_1 = bn.weight, arg3_1 = bn.bias,
    # arg4_1 = bn.running_mean, arg5_1 = bn.running_var

    # ★ 注意: 不再是 self.conv(x) 了 —— conv 被展开为 ATen conv2d
    _conv2d = torch.ops.aten.conv2d.default(x, arg0_1, arg1_1, [1,1], [1,1])
    #           ↑ ATen 函数调用, 不是 nn.Module!

    _bn = torch.ops.aten._native_batch_norm_legit_functional.default(
        _conv2d, arg4_1, arg5_1, arg2_1, arg3_1, True, 0.1, 1e-5
    )
    # ↑ BN 也被展开为 ATen 函数，weight/bias/running_mean/running_var 都是输入

    _relu = torch.ops.aten.relu.default(_bn[0])  # ReLU 也是一个 ATen 函数
    return _relu
```

### 2.3 关键变化：nn.Module 变成了参数 + ATen op

这是 PT2E 区别于 Eager Mode 和 FX Mode 的根本特征：

```
Eager Mode: 模型里是 nn.Module 对象，FakeQuantize 被插在 Module.forward 内部
FX Mode:    symbolic_trace 保留了 call_module 节点，但 FQ 作为独立 call_module 插入
PT2E:      全部展开为 ATen call_function 节点，FQ 作为 call_function 插入
```

**这意味着什么？** 在 PT2E 中，你操作的是**纯函数级的计算图**。每个节点是一个 ATen op + 它的输入/输出边。要插入 FQ，你就在某条边上插一个 `call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default, ...)` 节点。

### 2.4 capture 后的图结构（MiniModel 的完整节点列表）

```
Node Name                Op Type         Target                  Inputs
──────────────────────────────────────────────────────────────────────────
x                        placeholder     x                       (none)
arg0_1                   placeholder     conv.weight             (none)
arg1_1                   placeholder     conv.bias               (none)
arg2_1                   placeholder     bn.weight               (none)
arg3_1                   placeholder     bn.bias                 (none)
arg4_1                   placeholder     bn.running_mean         (none)
arg5_1                   placeholder     bn.running_var          (none)
_conv2d                  call_function   aten.conv2d.default     (x, arg0_1, arg1_1, ...)
_bn                      call_function   aten._native_bn_...     (_conv2d, arg4_1, ...)
_getitem                 call_function   operator.getitem        (_bn, 0)
_relu                    call_function   aten.relu.default       (_getitem,)
output                   output          output                  (_relu,)
```

**这是 Step 1 的产出**——一张纯 ATen 计算图，没有量化节点。所有参数（weight、bias、running_mean 等）都是图的 placeholder 输入，而不再藏在 Module 内部。

---

## 3. Step 2 深入：Quantizer 标注 — 哪些边需要量化、哪些不需要

### 3.1 Quantizer 的作用

有了计算图后，下一步是决定**在哪些边上插入 FQ 节点**。这项决策由 `Quantizer` 对象负责。

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer()
quantizer.set_global(
    get_symmetric_quantization_config(is_qat=True)
)
```

调用 `quantizer.set_global(config)` 只是注册了一个规则。真正的标注发生在 `prepare_qat_pt2e(gm, quantizer)` 调用时——Quantizer 遍历图中的每个节点，根据**节点类型**决定要不要标注、标注什么。

### 3.2 标注的决策树：每种 op 怎么判

Quantizer 内部对每种 ATen op 有一个分派表。以 XNNPACKQuantizer 为例：

```
对于每个 call_function 节点:

  ┌─ torch.ops.aten.conv2d.default     → 标注 input[0] (输入激活) → 需要一个 FQ
  │                                     标注 input[1] (weight)     → 需要 per-channel FQ
  │
  ├─ torch.ops.aten.linear.default     → 同上
  │
  ├─ torch.ops.aten.addmm.default      → 同上 (Linear 的底层 ATen 形式)
  │
  ├─ torch.ops.aten.add.Tensor         → 标注 input[0] + input[1] → 需要 **共享** FQ
  │                                     (两个输入必须用同一个 scale!)
  │
  ├─ torch.ops.aten.cat.default        → 标注所有输入 → 需要共享 FQ
  │
  ├─ torch.ops.aten.relu.default       → ★ 不标注 (passthrough)
  ├─ torch.ops.aten.gelu.default       → ★ 不标注
  ├─ torch.ops.aten.max_pool2d.default → ★ 不标注
  │
  ├─ torch.ops.aten.softmax.default    → ★ 不标注 (FP32 精度)
  ├─ torch.ops.aten.layer_norm.default → ★ 不标注 (FP32 精度)
  │
  └─ 其他未注册 op                     → ★ 不标注 (unknown → 保守)
```

### 3.3 标注的内部表示：一个 annotation map

Quantizer 维护一个内部映射表——**每条边 → 一个量化配置**：

```python
# Quantizer 内部的 annotation map (概念上):
annotations = {
    # key = (node_name, input_index) — 哪条边需要量化
    ("_conv2d", 0): QuantizationSpec(    # Conv 的输入激活
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,  # 激活: per-tensor
        is_qat=True,
    ),
    ("_conv2d", 1): QuantizationSpec(    # Conv 的 weight
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,  # 权重: per-channel!
        is_qat=True,
    ),
    ("_bn", 0): QuantizationSpec(        # BN 输入 (来自 Conv 输出)
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        is_qat=True,
    ),
    # relu 不标注 → 不下量化
}
```

**注意三个关键设计决策**：

1. **Weight 用 per-channel symmetric，激活用 per-tensor affine**。这是硬件决定的——INT8 Conv 指令支持 per-channel weight scale，但输入/输出 scale 必须是 per-tensor。
2. **ReLU、Pool、Softmax 不标注**。这些 op 不引入新的量化需求——它们只是对现有 tensor 做变换。
3. **Add/Concat 的多个输入要共享 FQ**。如果 `add(a, b)` 的两个输入被不同的 FQ 量化——a 的 scale=0.1, b 的 scale=0.01——加法后的值就不可比较了（不同"单位制"）。

---

## 4. Step 3 深入：prepare_qat_pt2e — 在图中插入 FQ 节点

### 4.1 从 annotation 到 FQ 节点

这是整个管线最核心的变换。`prepare_qat_pt2e` 读取 Quantizer 的标注，在图的每条被标注的边上**插入一个 FQ 节点**（或共享已有的）。

```python
prepared_model = prepare_qat_pt2e(gm, quantizer)
```

### 4.2 插入的四种情形

```
情形 A: 单消费者 — 在被标注 op 前插入 FQ
──────────────────────────────────────────
  标注: ("_conv2d", 0)  — Conv 的输入需要量化

  插入前:  x ──────────→ aten.conv2d
  插入后:  x → FQ_node → aten.conv2d

  新 FQ_node 做: x_fq = fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)
  Conv 收到的 x_fq 是"看起来像 INT8"的 FP32 值


情形 B: 多消费者共享 FQ — Add 的两个输入用同一个 FQ
──────────────────────────────────────────────────
  标注: ("_add", 0) + ("_add", 1) — 两个输入都需要量化，但要共享

  插入前:  a ─┐
               ├→ aten.add.Tensor → output
           b ─┘

  插入后:  a ─┐
               ├→ [共享 FQ_node] → a_fq ─┐
           b ─┘                          ├→ aten.add.Tensor → output
                                         │
               ├→ [共享 FQ_node] → b_fq ─┘
              (同一个 FQ 模块!)

  关键设计: a 和 b 通过同一个 FQ 节点 = 同一个 scale = 同一"单位制" → 加法有意义


情形 C: Weight FQ — 插入在 weight 参数路径上
─────────────────────────────────────────
  标注: ("_conv2d", 1) — weight 需要 per-channel 量化

  插入前:  arg0_1 (conv.weight) ──→ aten.conv2d
  插入后:  arg0_1 → FQ_weight_node → aten.conv2d

  FQ_weight_node 做: w_fq = fake_quantize_per_channel_affine(w, scale_per_ch, zp, -128, 127)
  ★ 注意: weight FQ 是 per-channel — 64 个输出通道各有自己的 scale


情形 D: 输出 FQ — 确保下层的输入已经是量化的
─────────────────────────────────────────
  有时在一层的输出后插入 FQ，确保下一层收到的输入是量化版本。
  这在 eager mode 中更常见 (activation_post_process)，PT2E 中通常在消费侧插入。
```

### 4.3 完整的图变换：MiniModel 在 prepare 前后

```
=============== prepare_qat_pt2e 之前 ===============

  x ──────→ conv2d ──────→ _bn ──────→ relu ──────→ output
              ↑                ↑
         conv.weight      bn.params

=============== prepare_qat_pt2e 之后 ===============

  x ──→ [FQ_act_0] ──→ conv2d ──→ [FQ_act_1] ──→ _bn ──→ relu ──→ output
         (per-tensor,      ↑                (per-tensor,
          activation)  [FQ_wt_0]             activation)
                      (per-channel,
                       conv.weight)

  ★ 注意: relu 前后都没有 FQ — 它是 passthrough
  ★ 注意: conv2d 的 weight 输入上有一个 per-channel FQ
  ★ 注意: 输入 x 经过 FQ_act_0 变成 fake-quantized 值后才进入 conv2d
```

### 4.4 prepare 不改变的：图的拓扑结构

```
prepare 前:  8 个节点 (4 placeholders + 3 call_functions + 1 output)
prepare 后: 11 个节点 (原来的 8 个 + 3 个新增的 FQ call_function 节点)

图的结构不变: conv → bn → relu 的拓扑顺序不变
     变的: 在边上插入 FQ 节点 = 数据流过 FQ 后再到原目标
```

---

## 5. 算子级分派：不同 op 怎么插 FQ

这是 PT2E 量化最核心的设计——不是"所有 op 都一样处理"，而是**根据 op 的数学性质分派**。

### 5.1 有 weight 的 op（Conv, Linear, ConvTranspose, LSTM）

**需要两个 FQ**：一个给输入激活（per-tensor），一个给 weight（per-channel）。

```python
# Conv2d 的完整量化前向:
# input  [N, C_in, H, W]   → FQ_act  → [N, C_in, H, W]_fq
# weight [C_out, C_in, K, K] → FQ_wt → [C_out, C_in, K, K]_fq
# output = conv2d(input_fq, weight_fq, bias, ...)
#                                                      ↑ bias 不量化 — 保持 FP32
```

**为什么 bias 不量化？** Bias 的值通常比 weight 小 2-3 个数量级。如果用 INT8 量化 bias → 大部分 bias 值被量化到同一个值或 0 → 丢失所有信息。INT8 Conv 指令设计上也是 "INT8 input × INT8 weight → INT32 accumulation → + FP32 bias → FP32 output"。bias 全程 FP32。

```python
# 在图中看到的效果:
# arg0_1 (conv.weight) → [FQ_wt_0] → w_fq ─┐
# x → [FQ_act_0] → x_fq ────────────────────┤
#                                             ├→ aten.conv2d(x_fq, w_fq, arg1_1, ...)
# arg1_1 (conv.bias) ─────────────────────────┘    ↑ bias 直接进入, 无 FQ
```

### 5.2 Element-wise 二元 op（Add, Sub, Mul）

**需要 FQ，且两个输入必须共享同一个 FQ 实例。**

为什么必须共享？Add 的数学：`a + b`。如果 a 的 scale=0.1, b 的 scale=0.01：
- a=10 → 代表实际值 1.0
- b=50 → 代表实际值 0.5
- 加法: 10 + 50 = 60 — 但这不是任何有意义的"量化加法"结果

**共享 FQ 保证 a 和 b 有同一个 scale → 它们可以直接在 INT 域相加。**

```python
# 图中 Add 的量化:
# a_fp ─→ [FQ_shared] ─→ a_q ─┐
# b_fp ─→ [FQ_shared] ─→ b_q ─┤
#                              ├→ add(a_q, b_q) → output
#                              │
#      同一个 FQ 模块 ← 同一个 scale ← Add 的前提
```

**实现细节**：Quantizer 在标注 `("_add", 0)` 和 `("_add", 1)` 时，给它们分配同一个 `QuantizationSpec.shared` token。`prepare_qat_pt2e` 看到共享 token 时，为第一个 input 创建 FQ 节点，第二个 input 复用同一个。

### 5.3 Concat / Stack

**需要 FQ，且所有被拼接的输入必须共享同一个 FQ。**

和 Add 同样的理由——拼接在一起的 tensor 必须来自同一个"量化域"。

```python
# Concat 的量化:
# a_fp ─→ [FQ_shared] ─→ a_q ─┐
# b_fp ─→ [FQ_shared] ─→ b_q ─┤
# c_fp ─→ [FQ_shared] ─→ c_q ─┤
#                              ├→ cat([a_q, b_q, c_q], dim=1)
```

### 5.4 Passthrough op（ReLU, GELU, Sigmoid, MaxPool, AvgPool, Upsample, Pad）

**不插 FQ。** 这些 op 不改变量化域——输入和输出在同一个"scale 空间"内。

```
设计理由: 如果 ReLU 前后各插一个 FQ:
  x_fq → [FQ] → x_q → relu → y → [FQ] → y_q → 下一层

  两次量化引入双重量化误差, 且 ReLU(max(0,x)) 本身就 trivial——中间插 FQ 没有信息增益。

正确做法: 让 FQ 在 ReLU 之前（或之后）出现一次就够了
  x → [FQ] → x_q → relu → y_q → 下一层
       ↑
    x_q 是 fake-quantized 的, ReLU 后的输出自然也是 fake-quantized 的
    因为 ReLU(max(0, x_q)) 不能"恢复"被截断的精度
```

### 5.5 Softmax / LayerNorm / BatchNorm（训练期间）

**不插 FQ。** 这些 op 的精度敏感，尤其 Softmax（注意力权重）和 LayerNorm（归一化）——量化它们通常导致严重精度损失。保持 FP32。

### 5.6 MatMul（两个动态输入）

**两个输入都需要 FQ，且不需要共享（各自独立 scale）。**

和 Conv 类比：MatMul 的两个输入类似于"激活"和"权重"——各自独立量化。MatMul 本身可以处理不同 scale 的输入（输出 = 两个 scale 的乘积）。

### 5.7 总结：一张分派表

```
Op 类别              示例                      输入 FQ         Weight FQ      FQ 共享?
──────────────────────────────────────────────────────────────────────────────────
Weighted              Conv, Linear, ConvTrans   per-tensor     per-channel    不共享
                      LSTM                      per-tensor     per-channel    不共享
Element-wise Binary   Add, Sub, Mul             per-tensor     N/A            必须共享
Concat                cat, stack                per-tensor     N/A            必须共享
Passthrough           ReLU, GELU, MaxPool       N/A            N/A            N/A
                      AvgPool, Upsample         N/A            N/A            N/A
Precision-sensitive   Softmax, LayerNorm        N/A            N/A            N/A
                      BatchNorm                 N/A            N/A            N/A
MatMul                两个动态矩阵输入            per-tensor × 2  N/A           各自独立
```

---

## 6. 完整 tensor 数据流追踪：一个 batch 穿过 QAT 图的全过程

### 6.1 追踪一个具体的 tensor

以 MiniModel + batch_size=1 为例。假设输入 `x` 是一个形状 `[1, 3, 32, 32]` 的 FP32 tensor。我们追踪它穿过 prepare 后的 QAT 图。

```
Step 1: 输入 x [1,3,32,32] (FP32)
  │
  ▼
Step 2: FQ_act_0 (per-tensor, 8-bit symmetric)
  │   observer_enabled=True → 更新 min_val/max_val → 计算 scale_act_0
  │   fake_quant_enabled=True → x_fq = round(x/scale_act_0).clamp(-128,127) * scale_act_0
  │   输出: [1,3,32,32] (FP32, 但值域被限制在 256 个离散值)
  │
  ▼
Step 3: aten.conv2d
  │   输入 A: x_fq [1,3,32,32]
  │   输入 B: w_fq [16,3,3,3] ← 经过 FQ_wt_0 量化后的 weight
  │   输入 C: conv.bias [16] (FP32, 不量化)
  │
  │   内部: INT8 matmul (模拟) = (INT8)A × (INT8)B → INT32 → + FP32 bias → FP32
  │   输出: y_conv [1,16,32,32] (FP32)
  │
  ▼
Step 4: FQ_act_1 (per-tensor, 8-bit affine — 非对称, 因为 ReLU 后 ≥0)
  │   observer_enabled=True → 更新 min_val/max_val → 计算 scale_act_1, zp_act_1
  │   fake_quant_enabled=True → y_fq = round(y_conv/scale_act_1+zp).clamp(0,255) * scale_act_1
  │   输出: [1,16,32,32] (FP32, 离散值)
  │
  ▼
Step 5: aten._native_batch_norm_legit_functional
  │   输入: y_fq [1,16,32,32] (已量化)
  │   参数: bn.weight, bn.bias, running_mean, running_var
  │   BN 用 running stats (model 在 train 模式但 BN 被冻结 → 用 running, 不用 batch)
  │   输出: y_bn [1,16,32,32] (FP32)
  │
  ▼
Step 6: aten.relu
  │   输入: y_bn [1,16,32,32]
  │   输出: [1,16,32,32] (全 ≥ 0)
  │   ★ 没有 FQ 插入在这里 — ReLU 是 passthrough
  │
  ▼
Step 7: output → 传给下一层（或 loss 函数）
```

### 6.2 在每个 Step 观察 tensor 的 dtype 和值域

```
Step    Tensor          dtype   值域                         离散值数量
───────────────────────────────────────────────────────────────────
1       x               FP32    ~[-2.0, 2.0] (归一化后)      连续
2       x_fq            FP32    ~[-2.0, 2.0] 但只有 256 个值 256
3       y_conv          FP32    ~(任意范围)                    连续(≈)
4       y_fq            FP32    [0, 某个正数] 256 个值        256
5       y_bn            FP32    ~归一化后范围                 连续(≈)
6       y_relu          FP32    全 ≥ 0                       (同输入)
```

**核心洞察**：FQ 节点的输入是 FP32 连续值，输出是 FP32 离散值（只有 256 个可能取值）。下游 op 只能"看到"这 256 个值——这就是"量化噪声注入"的机制。反向传播时，STE 让梯度穿过 `round()`，weight 被推向"适合这 256 个值"的方向。

### 6.3 反向传播时发生了什么

```
loss.backward() 时梯度沿每条边逆流:

output grad
  → relu grad (STE: relu 的梯度 = grad × (input > 0))
    → _bn grad — BN 参数被更新 (但如果 BN 冻结, weight/bias 不更新)
      → FQ_act_1 grad
          输入侧: x_grad = STE(grad) — 梯度穿过 round() 不变
          参数侧: scale 的梯度 = None! ← 这是固定 scale QAT 的关键
        → conv2d grad
            weight grad = x_fq ⊗ grad (STE)  → 更新 weight
            bias grad = grad.sum(dims)  → 更新 bias
          → FQ_wt_0 grad
              输入侧: w_grad = STE(grad)
              参数侧: scale 的梯度 = None! ← 固定 scale 无法学习
            → conv.weight 参数: 收到 w_grad, 被 optimizer 更新
          → FQ_act_0 grad
              输入侧: x_grad = STE(grad)
            → x (输入) — 不需要更新
```

**关键观察**：在固定 scale QAT 中，**scale 的梯度始终为 None**。这不是 bug——这是设计。`register_buffer` 的 scale 不参与梯度计算。只有 weight 和 bias 在更新。这就是为什么 Stage 1.5 会发现在低比特下"固定 scale 不够用"——因为 scale 不学习，而 weight 在学习 → scale 逐渐偏离最优值 → Stage 2 用 LSQ 解决。

### 6.4 如果模型有 skip connection

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x                          # ← skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual                  # ← Add: 两个输入需要共享 FQ!
        return self.relu(out)
```

**Prepare 后的图**：

```
x ──→ [FQ_act_0] ──→ conv1 ──→ [FQ_act_1] ──→ bn1 ──→ relu
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
                  [FQ_act_2] ──→ conv2 ──→ bn2 ──→ [FQ_shared] ──→ out_q ─┐
                                                                           │
x ──→ [FQ_act_0] ──→ x_q ──→ [FQ_shared] ──→ residual_q ─────────────────┤
                                                                           │
                                                                   aten.add(out_q, residual_q)
                                                                           │
                                                                          relu → output
```

**注意**：residual 路径和主路径在 Add 前通过同一个 `FQ_shared` 量化——保证它们有相同的 scale。

---

## 7. 四种 QuantizationSpec 完整拆解

前文 §3-4 讲 Quantizer 标注和 prepare 插入 FQ 时，反复出现了 `QuantizationSpec`、`SharedQuantizationSpec` 这些概念。这一节把它们彻底拆开——每种 Spec 的语义、在 prepare 中如何被消费、在图中生成什么节点。

### 7.1 Spec 体系全景

```python
from torch.ao.quantization.quantizer import (
    QuantizationSpec,           # "新建 FQ 节点"
    SharedQuantizationSpec,     # "复用别人的 FQ 节点"
    DerivedQuantizationSpec,    # "从别人的 scale/zp 计算出我的"
    FixedQParamsQuantizationSpec, # "我的 scale/zp 是写死的，不需要 Observer"
)
```

`prepare()` 函数（`pt2e/prepare.py`）的核心逻辑：**遍历 Quantizer 标注的 `node.meta["quantization_annotation"]`，对每条被标注的边，根据 Spec 类型决定——创建新 FQ、复用已有的、还是推导。**

```
prepare() 消费 annotation 的决策:

  annotation.input_qspec_map = {
    (node, input_0): QuantizationSpec(...),       → 创建新 FQ
    (node, input_1): SharedQuantizationSpec(edge),→ 复用 edge 对应的 FQ
    (node, input_2): DerivedQuantizationSpec(...),→ 创建推导式 FQ
    (node, input_3): FixedQParamsQuantizationSpec(→ 创建固定参数 FQ
        scale=1/255, zp=0),
  }
```

### 7.2 QuantizationSpec — 新建 FQ

**最常用的 Spec**。语义："在这条边上创建一个全新的 Observer/FakeQuantize 节点。"

```python
QuantizationSpec(
    dtype=torch.int8,
    qscheme=torch.per_tensor_affine,
    quant_min=-128,
    quant_max=127,
    observer_or_fake_quant_ctr=MinMaxObserver.with_args(),
    # ↑ 用哪个 Observer 类来创建 FQ
    # QAT 模式: 这个就会被升级为 FakeQuantize（继承自 Observer，额外有 fake_quant 逻辑）
    # PTQ 模式: 保持纯 Observer（只收集 min/max，不做 fake_quant）
)
```

**prepare() 如何消费它**：

```python
# pt2e/prepare.py 中的简化逻辑
def _create_obs_or_fq_from_qspec(qspec, edge, is_qat):
    if isinstance(qspec, QuantizationSpec):
        # Step 1: 创建 Observer/FakeQuantize 实例
        obs_or_fq = qspec.observer_or_fake_quant_ctr(
            dtype=qspec.dtype,
            qscheme=qspec.qscheme,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
        )
        if is_qat:
            # 升级为 FakeQuantize（继承自 Observer，多了 fake_quant_enabled）
            obs_or_fq = _upgrade_to_fake_quantize(obs_or_fq)

        # Step 2: 把实例注册为 GraphModule 的子模块
        #   → 给它一个唯一名字: "activation_post_process_0", "weight_fake_quant_1" 等
        graph_module.add_submodule(fq_name, obs_or_fq)

        # Step 3: 在图中创建 call_module 节点
        #   → 在 edge 上插入: prev_node → [call_module(fq_name)] → cur_node
        with graph.inserting_before(cur_node):
            fq_node = graph.create_node('call_module', fq_name, args=(prev_node,))

        # Step 4: 重新连线
        #   原: prev_node → cur_node (cur_node.args = (prev_node,))
        #   新: prev_node → fq_node → cur_node
        cur_node.replace_input_with(prev_node, fq_node)

        return obs_or_fq, fq_node
```

**使用者**：Conv/Lienar 的输入激活、Conv/Lienar 的 weight、单独需要量化的 tensor。

### 7.3 SharedQuantizationSpec — 复用别人的 FQ

**语义**："我不创建新 FQ，我用那一条边上的 FQ。"

```python
SharedQuantizationSpec(
    edge_or_node=(add_node, add_node.args[0])
    # ↑ "我和 (add_node, add_node.args[0]) 这条边共享同一个 FQ"
)
```

**为什么需要共享？** 最典型的是 `aten.add(a, b)`：a 和 b 必须在同一个量化"单位制"下才能做加法（相同的 scale，相同的 zero_point）。如果 a 的 scale=0.1、b 的 scale=0.01，`a_q × 0.1 + b_q × 0.01` 的结果没有物理意义。

**prepare() 如何消费它**：

```python
def _create_obs_or_fq_from_qspec(qspec, edge, is_qat, obs_or_fq_map):
    if isinstance(qspec, SharedQuantizationSpec):
        # Step 1: 找到共享源边
        shared_edge = qspec.edge_or_node

        # Step 2: 从 map 中取已有的 FQ 实例
        #   obs_or_fq_map 是在前面处理 QuantizationSpec 时建立的
        shared_obs_or_fq = obs_or_fq_map[shared_edge]
        #   ★ 如果 shared_edge 还没处理过（共享链），抛异常

        # Step 3: 把这条边也映射到同一个 FQ 实例
        obs_or_fq_map[edge] = shared_obs_or_fq

        # Step 4: 在图中创建 call_module 节点
        #   ★ 不同的 call_module 节点, 但 target 指向同一个子模块!
        fq_name = _get_fq_name(shared_obs_or_fq)
        with graph.inserting_before(cur_node):
            fq_node = graph.create_node('call_module', fq_name, args=(prev_node,))
        #   注意: fq_name 和共享源的 fq_name 相同 → 两个 call_module 调用同一模块

        cur_node.replace_input_with(prev_node, fq_node)
        return shared_obs_or_fq, fq_node
```

**关键结果**：图中有两个 `call_module` 节点（一个处理 a，一个处理 b），但它们的 `target` 指向**同一个子模块** `FQ_shared`。这个子模块只有一个 `scale` buffer——a 和 b 穿过同一个 FQ 实例 → 同一个 scale → 同一"单位制" → Add 有意义。

**在 Add 的图结构中的表现**：

```
a_fp ──→ [call_module("FQ_shared")] ──→ a_q ─┐
                                               ├─→ add(a_q, b_q)
b_fp ──→ [call_module("FQ_shared")] ──→ b_q ─┘
          (同一个 target!)
```

**使用者**：`aten.add` 的 input[1]，`aten.sub` 的 input[1]，`aten.mul` 的 input[1]，`aten.cat` 的 input[1:] 全部。

**共享的传递性**：如果 `b` 和 `a` 共享，`c` 和 `b` 共享 → `c` 和 `a` 也共享同一个 FQ。`obs_or_fq_map` 保证了解引用时找到的是同一个底层实例。

### 7.4 DerivedQuantizationSpec — 推导 scale/zp

**语义**："我不需要自己的 Observer。我的 scale 和 zero_point 是从别的 FQ 的 scale/zp 计算出来的。"

最典型的场景：**量化 Conv 的 bias**。

Bias 的值太小（通常比 weight 小 2-3 个数量级），直接量化 bias → 大部分值被量化到同一个值 → 丢失信息。但 INT8 Conv 指令 `INT8 input × INT8 weight → INT32 acc → +??? bias` 需要 bias 也有 scale。

**解决方案**：bias 的 scale 由 "输入 scale × 权重 scale" 推导出来。不需要 Observer 观察 bias 的数据——只靠已有的两个 FQ 就能算出 bias 的 scale。

```python
def derive_bias_qparams(obs_or_fqs):
    """
    obs_or_fqs[0] = 输入激活的 FQ
    obs_or_fqs[1] = 权重的 FQ
    """
    act_scale, act_zp = obs_or_fqs[0].calculate_qparams()
    weight_scale, weight_zp = obs_or_fqs[1].calculate_qparams()

    # bias_scale = act_scale × wt_scale
    # 原因: INT8 Conv 内部计算: (q_a - z_a) × (q_w - z_w)
    # = q_a × q_w - z_a × q_w - q_a × z_w + z_a × z_w
    # 反量化时 × act_scale × wt_scale → bias 的 scale 也是 act_scale × wt_scale
    return act_scale * weight_scale, torch.zeros(1, dtype=torch.int32)

bias_spec = DerivedQuantizationSpec(
    derived_from=[
        (conv_node, conv_node.args[0]),  # 源 1: 输入激活的 FQ
        (conv_node, conv_node.args[1]),  # 源 2: 权重的 FQ
    ],
    derive_qparams_fn=derive_bias_qparams,
    dtype=torch.int32,
    quant_min=-(2**31),
    quant_max=2**31 - 1,
    qscheme=torch.per_tensor_symmetric,
)
```

**prepare() 如何消费它**：

```python
def _create_obs_or_fq_from_qspec(qspec, edge, is_qat, obs_or_fq_map):
    if isinstance(qspec, DerivedQuantizationSpec):
        # Step 1: 找到源 FQ
        src_obs_or_fqs = []
        for src_edge in qspec.derived_from:
            src_obs = obs_or_fq_map[src_edge]  # 从 map 中取已有的
            src_obs_or_fqs.append(src_obs)

        # Step 2: 创建 DerivedObserverOrFakeQuantize 包装器
        #   (这是一个特殊的 FQ——没有自己的 Observer，scale 来自 derive_qparams_fn)
        derived_obs = DerivedObserverOrFakeQuantize(
            derive_qparams_fn=qspec.derive_qparams_fn,
            src_obs_or_fqs=src_obs_or_fqs,
            dtype=qspec.dtype,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
            qscheme=qspec.qscheme,
        )

        # Step 3: 注册 + 插入节点（同 QuantizationSpec）
        graph_module.add_submodule(fq_name, derived_obs)
        # ...创建 call_module 节点...

        return derived_obs, fq_node
```

**运行时行为**：

```
每次 forward:
  1. 源 FQ（输入激活）更新 scale_act
  2. 源 FQ（权重）更新 scale_wt
  3. Derived FQ 调用 derive_qparams_fn([FQ_act, FQ_wt])
     → scale_bias = scale_act × scale_wt
     → 用 scale_bias 做 bias 的 fake_quantize
  4. aten.conv2d 收到 (x_fq, w_fq, bias_fq)
```

**关键设计**：Derived FQ 不维护自己的 Observer——它没有 min_val/max_val buffer。每次 forward 时 scale 被从源头重新计算。这保证了 bias_scale 始终等于 `act_scale × wt_scale`——即使 act_scale 和 wt_scale 在 QAT 训练中发生变化。

### 7.5 FixedQParamsQuantizationSpec — 写死的 scale/zp

**语义**："我的 scale 和 zero_point 是常量——不需要 Observer 收集数据。"

最典型的场景：**Sigmoid 的输出**。Sigmoid 输出永远在 [0, 1] 范围内，无论输入是什么。可以直接写死 `scale = 1/255, zp = 0`——校准对它没有意义，因为范围是已知且固定的。

```python
FixedQParamsQuantizationSpec(
    dtype=torch.uint8,
    scale=1.0 / 255.0,     # ← 写死
    zero_point=0,           # ← 写死
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
)
```

**prepare() 如何消费它**：

```python
def _create_obs_or_fq_from_qspec(qspec, edge, is_qat, obs_or_fq_map):
    if isinstance(qspec, FixedQParamsQuantizationSpec):
        # Step 1: 创建 FixedQParamsFakeQuantize（特殊 FQ——没有 Observer）
        obs = FixedQParamsFakeQuantize(
            scale=qspec.scale,         # ← 直接作为属性存储
            zero_point=qspec.zero_point,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
            dtype=qspec.dtype,
            qscheme=qspec.qscheme,
        )

        # Step 2: 注册 + 插入节点
        graph_module.add_submodule(fq_name, obs)
        with graph.inserting_before(cur_node):
            fq_node = graph.create_node('call_module', fq_name, args=(prev_node,))

        cur_node.replace_input_with(prev_node, fq_node)
        return obs, fq_node
```

**运行时行为**：

```python
# FixedQParamsFakeQuantize.forward(X):
#   不管 X 的数据分布如何，始终用写死的 scale/zp:
#   X_q = round(X / 1/255 + 0).clamp(0, 255) * 1/255
#   = round(X * 255).clamp(0, 255) / 255

# 等价于:
def forward(self, X):
    X_q = torch.round(X * 255.0).clamp(0, 255) / 255.0
    return X_q
```

**使用者**：Sigmoid 输出、Tanh 输出（范围永远 [-1, 1]）、某些后端的预定义 lookup table。

### 7.6 四种 Spec 的决策树

Quantizer 的作者在标注每条边时做以下判断：

```
对图中的每条边 edge = (consumer_node, input_arg):

Q1: 这条边需要量化吗？
  ├── NO  → 不标注（ReLU、Pool、Softmax 等）
  └── YES → Q2

Q2: 这个 tensor 的 scale/zp 是已知的常量吗？
  ├── YES → FixedQParamsQuantizationSpec  (Sigmoid输出永远[0,1])
  └── NO  → Q3

Q3: 这个 tensor 的 scale/zp 应该和别人共享, 还是从别人推导, 还是独立?
  ├── 和另一条边共享 → Q3a
  ├── 从其他 FQ 推导 → Q3b
  └── 独立           → Q3c

Q3a → SharedQuantizationSpec(edge_or_node=共享源)
      场景: Add input[1], Cat input[1:], Stack input[1:]

Q3b → DerivedQuantizationSpec(derived_from=[源边列表], derive_qparams_fn=函数)
      场景: Conv bias (scale = act_scale × wt_scale)

Q3c → QuantizationSpec(dtype=..., qscheme=..., observer_or_fq_ctr=...)
      场景: Conv input activation, Conv weight, Linear weight
```

### 7.7 完整例子：Conv+Bias+Add 的 Spec 标注全景

```python
# 图:
#   x → conv2d(x, weight, bias) → out
#   out + skip → add_out

# Quantizer 标注:
class ExampleQuantizer:
    def annotate(self, gm):
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.conv2d.default:
                # 输入 x → 新建 FQ (activation)
                annotate_edge(node, 0,
                    QuantizationSpec(dtype=torch.int8, qscheme=per_tensor, ...))

                # weight → 新建 FQ (per-channel weight)
                annotate_edge(node, 1,
                    QuantizationSpec(dtype=torch.int8, qscheme=per_channel, ...))

                # bias → 推导 (从 input scale × weight scale)
                annotate_edge(node, 2,
                    DerivedQuantizationSpec(
                        derived_from=[(node, node.args[0]), (node, node.args[1])],
                        derive_qparams_fn=derive_bias_qparams,
                        dtype=torch.int32, ...
                    ))

            elif node.target == torch.ops.aten.add.Tensor:
                # input[0] (out) → 新建 FQ
                annotate_edge(node, 0,
                    QuantizationSpec(dtype=torch.int8, ...))

                # input[1] (skip) → 共享 input[0] 的 FQ
                annotate_edge(node, 1,
                    SharedQuantizationSpec(edge_or_node=(node, node.args[0])))
```

**prepare 后的图结构**：

```
x_fp ──→ [FQ_act] ──→ x_q ─┐
weight ──→ [FQ_wt] ──→ w_q ─┤
                              ├─→ conv2d(x_q, w_q) ──→ out_fp
bias ──→ [FQ_bias_derived] ──→ b_q ─────────────────────┘
              ↑ 没有 Observer!
              ↑ scale = FQ_act.scale × FQ_wt.scale (每次 forward 重新计算)
              ↑
         derive_qparams_fn = lambda [FQ_act, FQ_wt]: (act_s*wt_s, 0)

out_fp ──→ [FQ_add] ──→ out_q ─┐
skip_fp ─→ [FQ_add] ──→ sk_q  ─┤  ← 同一个 FQ_add!
                                 ├─→ add(out_q, sk_q) → result
```

---

## 8. Tensor 级操作：从 FQ 节点到 C++ 内核

前面讲的是"图中多了哪些节点"，这一节下沉到"一个 tensor 穿过 FQ 节点时，C++ 内核做了什么"。

### 8.1 `torch.fake_quantize_per_tensor_affine` 的完整拆解

这是整个 QAT 系统中最关键的 tensor 级操作。每种 FQ 节点的 forward 最终都调用它。

**函数签名**：

```python
torch.fake_quantize_per_tensor_affine(
    input,          # Tensor: 输入 FP32 tensor, 任意 shape
    scale,          # float: 量化步长
    zero_point,     # int: 零点偏移
    quant_min,      # int: 量化范围下界 (如 -128)
    quant_max,      # int: 量化范围上界 (如 127)
) -> Tensor        # 输出 FP32 tensor, 同 shape, 但只有 256 个离散值
```

**C++ 前向内核（CPU 路径）** 位于 `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp`，核心逻辑逐元素执行：

```cpp
// C++ 伪代码等价
float inv_scale = 1.0f / scale;

for each element x in input:
    // Step 1: 缩放 + 零点偏移
    float x_scaled = x * inv_scale + zero_point;

    // Step 2: 舍入 (round-half-to-even — banker's rounding)
    int64_t x_q = std::nearbyint(x_scaled);
    // ↑ nearbyint 而非 round: 使用当前 FPU 舍入模式 (默认 round-half-to-even)
    //   这比 std::round 更快, 且和硬件 INT8 转换行为一致

    // Step 3: 夹紧
    x_q = std::min(std::max(x_q, quant_min), quant_max);

    // Step 4: 反量化 (保持 FP32)
    output[i] = (x_q - zero_point) * scale;
```

**tensor 层面发生了什么，用具体数字**：

```python
# 具体例子: 输入 x = [0.3, -1.5, 2.8, -3.0], scale=0.02, zp=0, qmin=-128, qmax=127

# Step 1: x_scaled = x / 0.02 = [15, -75, 140, -150]
# Step 2: x_q = nearbyint(x_scaled) = [15, -75, 140, -150]
# Step 3: x_q = clamp(x_q, -128, 127) = [15, -75, 127, -128]
#          ↑ 2.8→140 被截断到 127 (clip error!)
#          ↑ -3.0→-150 被截断到 -128 (clip error!)
# Step 4: output = x_q * 0.02 = [0.30, -1.50, 2.54, -2.56]
#          ↑ 原本 2.8 → 变成 2.54 (误差 0.26, clip error)
#          ↑ 原本 -3.0 → 变成 -2.56 (误差 0.44, clip error)
#          ↑ 原本 0.3 → 不变 (舍入无损失)
```

### 8.2 STE 反向内核

**C++ 反向内核** 位于同一文件，核心逻辑：

```cpp
// C++ 伪代码等价
float inv_scale = 1.0f / scale;

for each element i:
    float x = saved_input[i];     // 从 saved_tensors 恢复前向输入
    float dy = grad_output[i];    // 上游传下来的梯度

    // Step 1: 计算 x 的量化值 (和前向一样的计算)
    int64_t x_q = std::nearbyint(x * inv_scale + zero_point);

    // Step 2: 判断 x 是否在量化范围内
    bool in_range = (x_q >= quant_min && x_q <= quant_max);

    // Step 3: STE — 在范围内: 梯度直通; 在范围外: 梯度截断为 0
    grad_input[i] = dy * (in_range ? 1.0f : 0.0f);

// 注意: scale 和 zero_point 的梯度永远是 0!
// 因为它们是 buffer (register_buffer), 不参与梯度计算
// LSQ (Stage 2) 正是为了修复这个问题 → scale 变成 nn.Parameter
```

**tensor 层面的梯度流**：

```python
# 具体例子 (继续上面的):
# x = [0.3, -1.5, 2.8, -3.0], output = [0.30, -1.50, 2.54, -2.56]
# grad_output = [0.1, 0.2, 0.3, 0.4] (来自上层)

# 反向:
# x_q = [15, -75, 140, -150]
# in_range = [True, True, False, False]
#           ↑ 15 在[-128,127]内   ↑ 140 超出 127  → 梯度 = 0!
#                                     ↑ -150 超出 -128 → 梯度 = 0!

# grad_input = [0.1×1, 0.2×1, 0.3×0, 0.4×0] = [0.1, 0.2, 0, 0]
#                                                                ↑ 被 clip 的元素得不到梯度
#                                                                ↑ 模型无法"知道"应该把 2.8 往 -128 靠
#                                                                ↑ 这就是 clip error 比 round error 致命的原因!
```

**关键洞察**：被 clip 的元素在反向中得到零梯度——模型无法从它们身上学习。这就是为什么校准阶段要好好选 scale——如果 scale 太小（clip 太多数据），大量元素得不到梯度 → QAT 训练效率大幅降低。

### 8.3 `torch.fake_quantize_per_channel_affine` — per-channel 版本

当 FQ 用 `qscheme=per_channel_symmetric` 时（权重量化的默认配置），调用这个版本。

**和 per-tensor 版本的核心区别**：

```python
# per-tensor: 一个 scale 管全部元素
scale = 0.02  # 标量
x_q = nearbyint(x / 0.02).clamp(-128, 127) * 0.02

# per-channel: 每个输出通道有自己的 scale
scale = [0.02, 0.05, 0.01, ..., 0.03]  # shape: [C_out], 共 64 个值
# 对第 i 个输出通道:
x_q[i] = nearbyint(x[i] / scale[i]).clamp(-128, 127) * scale[i]
```

**C++ 内核**（`fake_quantize_per_channel_affine_cpu`）：

```cpp
// 简化伪代码
for each channel c:
    float inv_scale_c = 1.0f / scale[c];
    for each element in channel c:
        x_scaled = x * inv_scale_c + zero_point[c];
        x_q = nearbyint(x_scaled);
        x_q = clamp(x_q, quant_min, quant_max);
        output = (x_q - zero_point[c]) * scale[c];
```

### 8.4 FQ 节点在图中的实际创建过程

前面讲的是"C++ 内核对 tensor 做了什么"，这里补上"FX 图是怎么创建这些节点的"。

```
prepare() 遍历 annotation map 时，对每条被标注的边执行:

┌─ Step 1: _create_obs_or_fq_from_qspec()
│   根据 Spec 类型创建 FQ 模块实例 (如 §7 所述)
│
├─ Step 2: graph_module.add_submodule("fq_0", fq_instance)
│   把 FQ 实例注册为 GraphModule 的属性
│   → graph_module.fq_0 = fq_instance
│   → 图节点可以通过 call_module("fq_0") 引用它
│
├─ Step 3: graph.inserting_before(target_node)
│   进入"在 target_node 之前插入"的上下文
│
├─ Step 4: graph.create_node('call_module', 'fq_0', args=(prev_node,))
│   创建新节点:
│     op = 'call_module'    → 表示"调用一个子模块"
│     target = 'fq_0'       → 子模块名
│     args = (prev_node,)   → 输入是 prev_node 的输出
│   返回: fq_node (新的 FX Node 对象)
│
├─ Step 5: target_node.replace_input_with(prev_node, fq_node)
│   修改边:
│     改前: prev_node ──→ target_node   (target_node.args[i] = prev_node)
│     改后: prev_node → fq_node → target_node
│     fq_node 自动成为 prev_node 的新 user
│
└─ Step 6: fq_node.meta 拷贝
　　 从 prev_node 和目标 op 继承 metadata (dtype, 量化信息等)
```

**具体例子：给 conv2d 的输入插入 FQ**：

```
# 改前:
#   %x : [1,3,32,32] (placeholder)
#   %conv2d : call_function[aten.conv2d](%x, %weight, %bias, ...)

# prepare() 找到 annotation: ("_conv2d", 0) → QuantizationSpec
# 插入:
#   %fq_0 : call_module[fq_0](%x)
#   %conv2d : call_function[aten.conv2d](%fq_0, %weight, %bias, ...)
#                                        ↑ args[0] 从 %x 改为 %fq_0

# 改后的 tensor 流:
#   x (FP32, 连续值) → [call_module fq_0] → x_fq (FP32, 256个离散值) → conv2d
```

### 8.5 convert_pt2e：FQ → 真量化节点的转换

QAT 训练完成后，`convert_pt2e` 把 FQ 节点替换为真正的量化/反量化操作。

```
convert_pt2e 的五步:

Step 1: _convert_to_reference_decomposed_fx(model)
  遍历所有 FQ 节点:
    读取 FQ 的 scale 和 zero_point (从 buffer 中取出最终固定值)
    把 FQ 替换为: quantize → dequantize 操作对

  FQ 节点 (QAT):
    x → [call_module FQ_0] → x_fq
    前向: fake_quantize (FP32 → INT8模拟 → FP32)

  Q/DQ 节点 (转换后):
    x → [call_function quantize_per_tensor(x, scale, zp, ...)] → x_int8
         → [call_function dequantize_per_tensor(x_int8, scale, zp, ...)] → x_fp32

Step 2: DuplicateDQPass — 去重 dequantize 节点
  如果同一个 tensor 被多个 op 消费 → 合并重复的 dequantize

Step 3: PortNodeMetaForQDQ — 搬运元数据
  把 FQ 节点上的 scale/zp/dtype 元数据搬运到新 Q/DQ 节点

Step 4: constant_fold — 常量折叠
  对于: weight(get_attr) → quantize → dequantize → conv2d
  折叠为: weight_quantized(get_attr) → dequantize → conv2d
  (预计算 quantize(weight) — 因为 weight 在推理时不变)

Step 5 (可选): use_reference_representation
  将所有 Q/DQ 对转换为后端相关的 INT8 计算模式
  例如: dequantize → fp32_conv → quantize → quantized_int8_conv
```

**转换前后的 tensor 流对比**：

```
========== 转换前 (QAT 训练完的模型) ==========
x (FP32, 连续值)
  → [FQ_act]  → x_fq (FP32, 256离散值, STE反向)
    → conv2d(x_fq, w_fq) → y_fp32
      → [FQ_out] → y_fq (FP32, 256离散值)
        → next_layer

========== 转换后 (INT8 推理模型) ==========
x (FP32, 连续值)
  → [quantize]   → x_i8 (torch.int8, [-128, 127])
    → [dequantize] → x_fp32 (FP32)
      → conv2d(x_fp32, w_fp32) → y_fp32  ← 还是 FP32 matmul, 但输入来自 dequantize
        → [quantize]   → y_i8 (torch.int8)
          → [dequantize] → y_fp32
            → next_layer
```

### 8.6 tensor 的生命周期：从 FP32 输入到 INT8 推理

把一切串起来——一个具体的 tensor 从用户输入到最终 INT8 推理的完整旅程：

```
Phase 1: capture (export)
  x [1,3,224,224] FP32, 连续值, 无量化
  图: x → conv2d → bn → relu → output
  所有计算是 FP32

Phase 2: prepare_qat_pt2e
  x → [FQ_act_0, observer=ON, fq=ON]
       │
       │  forward: x_fq = fake_quantize(x, scale_act_0, zp_act_0, -128, 127)
       │    → x_fq (FP32, 看起来像是 INT8 的 256 个离散值)
       │  backward: grad = STE(grad_output) × in_range_mask
       │    → scale_act_0 梯度 = None (buffer, 不学习)
       │
       └→ conv2d(x_fq, w_fq, bias)
            w_fq 来自 [FQ_wt_0](weight)
            bias 来自 [FQ_bias_derived](bias)

Phase 3: QAT training (10 epochs)
  每个 batch:
    前向: FP32 → FQ → 离散化 → conv → FQ → 离散化 → ...
    反向: loss.backward() → STE 穿过 FQ → weight 更新
    scale 不更新 (固定 scale QAT)  《— Stage 1.5 发现这是硬上限

  epoch 3+: observer 关, scale 冻结

Phase 4: convert_pt2e
  FQ 节点被替换为 Q/DQ 操作对
  scale 从 FQ buffer 中取出 → 写入 Q 节点属性
  weight 的 Q + DQ 被常量折叠 → pre-quantized weight

Phase 5: INT8 推理
  x (FP32) → quantize(scale=0.02, zp=0) → x_i8 [-128,127]
           → dequantize → x_fp32
           → INT8 Conv (后端指令: VNNI/DP4A)
           → y_fp32 → quantize → ... → output
```

---

## 9. Observer / FakeQuantize 核心模块

> 源码位置: `torch/ao/quantization/observer.py` + `fake_quantize.py`

上面讲的是"图级别的变换"和"tensor 级别的 C++ 操作"。现在下沉到**模块级别**——Observer 和 FakeQuantize 的内部实现。

### 9.1 Observer 继承树

```
ObserverBase (nn.Module)
  └── _ObserverBase  ← 添加 eps、dtype、qscheme、quant_min/quant_max
        ├── MinMaxObserver         — 直接取 min/max
        │     ├── MovingAverageMinMaxObserver  — EMA 平滑
        │     │     └── MovingAveragePerChannelMinMaxObserver
        │     └── PerChannelMinMaxObserver     — per-channel
        └── HistogramObserver      — 直方图 KL/MSE 搜索
```

### 9.2 MinMaxObserver 的完整实现

```python
import torch
import torch.nn as nn

class MinMaxObserver(nn.Module):
    """
    核心状态: min_val / max_val (两个 buffer)
    核心逻辑: forward 时更新 min/max, 返回原始输入 (不修改数据)

    设计决策:
    1. min_val/max_val 用 register_buffer — 跟随 device 迁移，保存到 state_dict
    2. forward 不修改数据 — 返回原始输入，纯"观察"
    3. calculate_qparams() 在需要时才调用 — 不是每次 forward 都算
    """
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=0, quant_max=255, eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps

        # buffer: 不是 nn.Parameter → 不参与梯度
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # ← detach: min/max 更新不进入计算图

        if self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            self.min_val = torch.min(self.min_val, x.min())
            self.max_val = torch.max(self.max_val, x.max())

        elif self.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            # channel = dim 0, 沿其余 dim 取 min/max
            x_flat = x.reshape(x.shape[0], -1)
            self.min_val = torch.min(self.min_val, x_flat.min(dim=1).values)
            self.max_val = torch.max(self.max_val, x_flat.max(dim=1).values)

        return x_orig  # ← 关键: 返回原始输入! 下游不知道 Observer 的存在

    def calculate_qparams(self):
        """在 forward 之外调用 — 通常在 eval 或手动触发"""
        if self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric):
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            scale = abs_max / float(self.quant_max - self.quant_min) * 2
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.max_val - self.min_val) / float(self.quant_max - self.quant_min)
            scale = torch.clamp(scale, min=self.eps)  # 防除零
            zero_point = self.quant_min - torch.round(self.min_val / scale)
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return scale, zero_point
```

### 9.3 MovingAverageMinMaxObserver：EMA 平滑

```python
class MovingAverageMinMaxObserver(MinMaxObserver):
    """
    为什么 EMA？直接 min/max 被一个异常 batch 污染 → 所有后续量化的 scale 都偏大。

    EMA 公式:
      new_val = old_val + α × (observed_val - old_val)

    当 α = 0.01: 最近 100 个 batch 占 ~63% 权重
    """
    def __init__(self, averaging_constant=0.01, **kwargs):
        super().__init__(**kwargs)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()

        if self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            min_cur = x.min()
            max_cur = x.max()
            # 初始化: 如果 min_val 是 inf → 直接赋值
            if self.min_val == float("inf"):
                self.min_val = min_cur
                self.max_val = max_cur
            else:
                self.min_val += self.averaging_constant * (min_cur - self.min_val)
                self.max_val += self.averaging_constant * (max_cur - self.max_val)

        return x_orig
```

### 9.4 FakeQuantize 的双开关状态机

FakeQuantize 是 Observer 的升级版——继承了 Observer 的 min/max 收集能力，额外加了**量化模拟**功能。

```python
class FakeQuantizeBase(nn.Module):
    """两个独立开关：为什么不用一个 mode 枚举？

    因为真实场景可能需要:
    - 某层永远不做量化 (两者都关)
    - 某层同时观察 + 量化 (两者都开, 某些高级校准场景)
    - 校准阶段 (observer 开, fake_quant 开) — 在量化噪声下收集统计
    - QAT 阶段 (observer 关, fake_quant 开) — 固定 scale 训练
    """
    def __init__(self):
        super().__init__()
        # tensor([1]) 而非 True: TorchScript 兼容
        self.register_buffer('observer_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))

    def enable_observer(self):    self.observer_enabled[0] = 1
    def disable_observer(self):   self.observer_enabled[0] = 0
    def enable_fake_quant(self):  self.fake_quant_enabled[0] = 1
    def disable_fake_quant(self): self.fake_quant_enabled[0] = 0


class FakeQuantize(FakeQuantizeBase, MovingAverageMinMaxObserver):
    """FakeQuantize — 多重继承 Observer 的统计能力 + FakeQuantizeBase 的状态管理"""

    def forward(self, X):
        # Phase 1: 观察 — 更新 min/max → 重新计算 scale
        if self.observer_enabled[0] == 1:
            # 调用 Observer.forward() — 更新 min_val/max_val
            self.activation_post_process(X)  # 等于 MovingAverageMinMaxObserver.forward(X)

        # Phase 2: 量化-反量化 — 模拟 INT8 推理
        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.calculate_qparams()

            if self.qscheme == torch.per_tensor_affine:
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(scale), int(zero_point),
                    self.quant_min, self.quant_max)
            elif self.qscheme == torch.per_tensor_symmetric:
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(scale), int(zero_point),
                    self.quant_min, self.quant_max)
            elif self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch.fake_quantize_per_channel_affine(
                    X, scale, zero_point, 0,  # channel dim = 0
                    self.quant_min, self.quant_max)

        return X
```

**`torch.fake_quantize_per_tensor_affine` 做了什么？** 这是 PyTorch C++ 核心函数。前向：`round(x/s + z) → clamp → (x - z) × s`。反向：STE——梯度直接穿过 `round()`。

### 9.5 QAT 三阶段的标志位切换

```
Phase 1 (前 2-3 epoch):   Calibration
  observer_enabled = 1     → Observer 正在收集 min/max, scale 每个 batch 更新
  fake_quant_enabled = 1   → 量化噪声已注入, weight 暴露在高散度信号中

Phase 2 (后续 epoch):     QAT Training
  observer_enabled = 0     → scale 冻结, 不再更新
  fake_quant_enabled = 1   → 量化噪声继续注入

Phase 3 (eval):           Inference/Eval
  observer_enabled = 0     → scale 不变
  fake_quant_enabled = 1   → 量化推理
  model.eval()             → BN 用 running stats
```

---

## 10. fuse_modules：Conv+BN 融合

> 源码位置: `torch/ao/quantization/fuse_modules.py`

### 10.1 为什么要融合？

训练时 Conv 和 BN 是两个独立的 op。推理时它们的数学等价于一个 Conv——所以先融合成纯 Conv，再对它插入量化节点。**减少一个 op = 减少一个 FQ 节点 = 减少一层量化误差。**

### 10.2 融合的数学

```
Conv:  y_conv = W @ x + b
BN:    y_bn = γ × (y_conv - μ) / √(σ² + ε) + β

融合后:
  y = W' @ x + b'
  其中:
    W' = W × γ / √(σ² + ε)       — 新的 weight
    b' = (b - μ) × γ / √(σ² + ε) + β  — 新的 bias
```

### 10.3 fuse_modules 六步流程

```
fuse_modules(model, [['conv1', 'bn1', 'relu1']])

Step 1: 解析路径
  _get_module(model, 'conv1') → Conv2d
  _get_module(model, 'bn1') → BatchNorm2d
  _get_module(model, 'relu1') → ReLU

Step 2: 查找融合函数
  (Conv2d, BatchNorm2d, ReLU) → fuse_conv_bn_relu

Step 3: 执行融合
  W' = W × γ / √(σ²+ε), b' = (b - μ) × γ / √(σ²+ε) + β
  返回 ConvReLU2d(W', b')  ← 纯 Conv + ReLU (BN 已吸收)

Step 4: 迁移 hooks
  原 conv1, relu1 的 hooks → 融合模块

Step 5: 替换模块树
  _set_module(model, 'conv1', ConvReLU2d)
  _set_module(model, 'bn1', nn.Identity())
  _set_module(model, 'relu1', nn.Identity())

Step 6: Post-processing
  验证 bn1 和 relu1 不再被 forward 调用
```

### 10.4 PT2E 中融合的处理

PT2E 中融合**不是必须的**——因为 `capture_pre_autograd_graph` 可能自动内联 BN 到 Conv 中（取决于 tracing 时的状态）。但手动融合仍然是好习惯——确保 BN 被正确吸收。

```python
# 在 capture 前融合
from torch.ao.quantization import fuse_modules

model = fuse_modules(model, [
    ['conv', 'bn', 'relu'],
])

# 然后再 capture
gm = capture_pre_autograd_graph(model, example_input)
```

---

## 11. QConfigMapping 与 Quantizer 体系

### 11.1 QConfigMapping（FX Mode 使用）

精细化控制：哪些层用哪种 Observer、哪些层不量化。

```python
from torch.ao.quantization import QConfigMapping, QConfig
from torch.ao.quantization.observer import (
    HistogramObserver, PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
)

qmap = QConfigMapping()
# 全局默认
qmap.set_global(QConfig(
    activation=HistogramObserver.with_args(dtype=torch.quint8),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
          qscheme=torch.per_channel_symmetric),
))
# 第一层不量化（保持 FP32 输入）
qmap.set_module_name("conv1", None)
# 最后一层不量化（保持 FP32 logits）
qmap.set_module_name("fc", None)
# 特定类型用特定配置
qmap.set_object_type(nn.Linear, QConfig(
    activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
          qscheme=torch.per_channel_symmetric),
))
# 正则匹配
qmap.set_module_name_regex("layer[34].*conv.*", QConfig(
    activation=HistogramObserver.with_args(dtype=torch.quint8),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
          qscheme=torch.per_channel_symmetric),
))

# 优先级: 精确名 > 正则 > 类型 > 全局
```

### 11.2 Quantizer（PT2E 使用）

PT2E 用 `Quantizer` 替代 `QConfigMapping`。换后端 = 换 Quantizer。

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer()

# 全局配置
quantizer.set_global(
    get_symmetric_quantization_config(is_qat=True)
)

# 按 op 类型配置
from torch.ao.quantization.quantizer import Quantizer
quantizer.set_object_type(
    torch.nn.Conv2d,
    get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
)

# 按模块名配置
quantizer.set_module_name(
    "conv1",
    get_symmetric_quantization_config(is_qat=True)
)
```

---

## 12. 三种模式对比：Eager / FX / PT2E

| 维度 | Eager Mode | FX Graph Mode | PT2E |
|------|-----------|--------------|------|
| **图捕获方式** | 无图 — 直接修改 `Module.forward` | `symbolic_trace`（跑一遍记录） | `capture_pre_autograd_graph`（ATen 展开） |
| **FQ 插入位置** | 在 `Module.forward` 内部 wrap conv/linear 调用 | 作为独立 `call_module` 节点插入图 | 作为独立 `call_function` 节点插入图 |
| **图可见性** | ✗ 看不见 FQ — 藏在 Module 内部 | ✓ 可 `print_tabular()` 查看所有节点 | ✓ 可打印 ATen 级图 |
| **动态控制流** | ✓ 支持 | ✗ 不支持 | ✓ 支持（`torch.export` 级别分析） |
| **后端适配** | `qconfig` 的 Observer 选择 | `qconfig_mapping` | 显式 `Quantizer` 对象 |
| **自定义 op** | 手动注册 | 手动注册量化 pattern | 在 Quantizer 中标注 |
| **跨后端可移植** | 差（qconfig 和 backend 强绑定） | 一般（QConfigMapping 可换） | 好（换 Quantizer = 换后端） |
| **当前状态** | Deprecated (PyTorch 2.x) | Stable, 生产可用 | PyTorch 2.5+ 推荐 |
| **主要用户** | 旧项目 | 大多数现有项目 | 新项目 + 自定义部署 |

### 选择建议

```
你想部署到 XNNPACK (ARM)   → PT2E + XNNPACKQuantizer
你想部署到 TensorRT        → PT2E + 自定义 Quantizer (或直接用 TensorRT 的量化工具)
你有复杂的控制流模型        → PT2E
你的模型很简单, 需要快速验证 → FX Mode
你在读旧代码                → Eager Mode (理解历史)
```

---

## 13. QAT 训练实战技巧

### 13.1 BN 冻结

```python
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()                        # 用 running stats
            m.weight.requires_grad_(False)  # 不更新 γ
            m.bias.requires_grad_(False)    # 不更新 β
```

### 13.2 Observer 调度

```python
# 标准调度: 前 3 epoch 开 observer, 之后关
from torch.ao.quantization import disable_observer, enable_observer

for epoch in range(num_epochs):
    if epoch < 3:
        prepared_model.apply(enable_observer)
    else:
        prepared_model.apply(disable_observer)

    train_one_epoch(prepared_model, train_loader, optimizer)
    validate(prepared_model, val_loader)
```

### 13.3 学习率

QAT 是微调不是从头训练。`lr = 1e-4 ~ 5e-5`（比原始训练的 lr 低 10-100 倍）。使用 cosine annealing。太大会 "忘记" 预训练权重。

### 13.4 第一层和最后一层不量化

```
第一层: 输入是原始像素值/embedding — 范围已知且稳定, 量化的收益小
最后一层: 输出是 logits/bbox — 精度敏感, 量化可能影响最终预测

通用规则:
  model.conv1 / model.embedding  → qconfig = None
  model.fc / model.head          → qconfig = None
```

---

## 14. 动手实验

| # | 实验 | 时间 | 产出 |
|---|------|:--:|------|
| 1 | 用 PT2E 跑通 MiniModel 的完整 QAT (capture → prepare → train → convert) | 45min | 一个完整的 QAT 管线 |
| 2 | `print_tabular()` 打印 prepare 前后的图，手动标注哪些节点是 FQ | 30min | 标注对比图 |
| 3 | 在 MiniModel 中加一个 skip connection (`out + x`) → 重新 prepare，观察 Add 前的共享 FQ | 30min | 理解共享 FQ 机制 |
| 4 | 用不同的 Quantizer 配置对比 (per-channel weight vs per-tensor weight) | 30min | 精度差异表 |
| 5 | MobileNetV3 用 FX Mode + PT2E 各跑一次 QAT，对比代码量和最终精度 | 1h | 两种模式的实战对比 |

---

## 检验标准

- [ ] 能画出 PT2E 的完整五步管线图 (capture → quantize → prepare → train → convert)
- [ ] 能说出 `capture_pre_autograd_graph` 对 MiniModel 做了什么——列出每个节点的 op 类型和输入
- [ ] 能说出不同 op (Conv/Add/ReLU/Softmax) 的 FQ 插入策略——为什么 Add 要共享、ReLU 不需 FQ
- [ ] 能追踪一个 tensor 穿过 prepare 后的图——每一步的 shape + dtype + 值域变化
- [ ] 能用 `print_tabular()` 打印图并手动标注 FQ 节点
- [ ] 能解释 `observer_enabled` 和 `fake_quant_enabled` 三个阶段的切换逻辑
- [ ] 能在 MobileNetV3 上跑通 PT2E QAT，拿到 INT8 模型
- [ ] 能说出 Eager / FX / PT2E 的核心区别——选哪个、为什么

---

> 💡 **学习建议**：这一章的核心不是记住 API，而是建立**图变换的直觉**。
>
> 学习路线：
> 1. 先在 MiniModel 上跑一遍完整的 PT2E 管线，用 `print_tabular()` 看每一步的图变化
> 2. 然后读 §5 "算子级分派"——这里回答了"为什么 Add 要共享 FQ 而 ReLU 不需要"
> 3. 最后读 §6 "tensor 数据流追踪"——把每一步的 tensor 变化串起来
>
> **掌握了 §5 的算子分派表，你就懂了 PT2E 量化的一半。另一半在 §4 的 FQ 插入逻辑。**
>
> Next: [Stage 1.5: QAT 训练深度剖析 — 从"能跑"到"理解为什么"](./Stage1.5_QAT训练深度剖析.md)
