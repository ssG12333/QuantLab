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
10. [9. Observer / FakeQuantize 核心模块](#9-observer--fakequantize-核心模块) — 含 tensor 级生命周期
11. [10. fuse_modules：Conv+BN 融合](#10-fuse_modulesconvbn-融合)
12. [11. Quantizer 深度拆解：PT2E 的"量化大脑" 🆕](#11-quantizer-深度拆解pt2e-的量化大脑)
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
        _conv2d, arg4_1, arg5_1, arg2_1, arg3_1, True, 0.1, 0.00001
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

### 9.6 Observer 的 tensor 级操作与完整生命周期

上面讲了 Observer 的"类结构"和 FakeQuantize 的"状态机"。这一节下沉到 tensor 级别——**一个 Observer 实例从创建到销毁，对穿过它的 tensor 做了什么**。

**Observer.forward() 的逐行拆解**：

```python
# 假设一个 MovingAverageMinMaxObserver 实例，observer_enabled=1
# 输入: x_orig [1, 64, 32, 32] FP32，来自上层 Conv 的输出

def forward(self, x_orig):
    # Step 1: detach — 切断 min/max 更新的梯度链
    x = x_orig.detach()
    # ★ 为什么 detach？min/max 的更新是"统计量收集"，不需要反向传播
    #   如果 x 连着计算图，min() 和 max() 操作也会被 autograd 追踪
    #   → 不需要计算 min 的梯度（min 不可导），detach 后更清晰

    # Step 2: 更新 min/max
    if self.qscheme == per_tensor_affine:
        # per-tensor: 取整个 tensor 的 min/max
        min_cur = x.min()    # 标量
        max_cur = x.max()    # 标量
        if self.averaging_constant > 0:
            # EMA 更新: new = old + α × (observed - old)
            self.min_val += self.averaging_constant * (min_cur - self.min_val)
            self.max_val += self.averaging_constant * (max_cur - self.max_val)
        else:
            self.min_val = min_cur
            self.max_val = max_cur

    elif self.qscheme == per_channel_symmetric:
        # per-channel: dim 0 是 channel，沿其余 dim 取 min/max
        x_flat = x.reshape(x.shape[0], -1)              # [64, 1024]
        min_cur = x_flat.min(dim=1).values              # [64] 每通道的 min
        max_cur = x_flat.max(dim=1).values              # [64] 每通道的 max
        # EMA 更新同上...

    # Step 3: 返回原始输入 — 不修改!
    return x_orig
    # ★ 关键设计: Observer 是"透明的" — 下游 op 完全不知道 Observer 的存在
    #   Observer 只做统计收集, 不改变 tensor 的值
    #   实际量化注入是 FakeQuantize (继承自 Observer) 在 fake_quant_enabled=1 时做的事
```

**Observer 收集到的数据长什么样**：

```python
# 不同 Observer 的 min_val / max_val 的 shape

# MinMaxObserver (per-tensor):
min_val = tensor(-1.2345)  # 标量 — 一个值管所有元素
max_val = tensor(3.4567)

# PerChannelMinMaxObserver (per-channel, 64 通道):
min_val = tensor([-1.2, -0.5, ..., -0.3])  # [64] — 每通道一个
max_val = tensor([3.4, 2.1, ..., 1.8])

# MovingAverageMinMaxObserver (per-tensor, EMA):
# 每个 batch forward 后 min_val/max_val 被 EMA 公式更新
# 初始值 = inf/-inf → 第 1 个 batch: min_val = observed_min (直接赋值)
# 第 2 个 batch: min_val = old + α × (new - old) → 平滑过渡
```

**Observer 的完整生命周期**：

```
Phase 1: CREATION — prepare_qat_pt2e 内部
─────────────────────────────────────────
  prepare() 读取 QuantizationSpec.observer_or_fake_quant_ctr
  → 创建 Observer 实例 (如 MinMaxObserver.with_args(dtype=torch.quint8))
  → graph_module.add_submodule("activation_post_process_0", obs)
  → 在图边插入 call_module("activation_post_process_0")
  → 状态: observer_enabled=1, fake_quant_enabled=1 (QAT 默认都开)

Phase 2: CALIBRATION — QAT 前 2-3 epoch
─────────────────────────────────────────
  每个 batch forward:
    obs.forward(tensor) →
      - 更新 min_val / max_val (EMA 平滑)
      - calculate_qparams() 重新计算 scale / zero_point
      - 如果 fake_quant_enabled=1: 用新 scale 做 fake_quantize → 返回离散化 tensor
      - 返回的 tensor 只有 256 个可能值 (INT8) 或更少 (低比特)
  关键: scale 每个 batch 都在更新! 这个阶段 weight 暴露在不断变化的量化噪声下

Phase 3: FREEZE — QAT 剩余 epoch
─────────────────────────────────────────
  model.apply(disable_observer) → observer_enabled=0
  每个 batch forward:
    obs.forward(tensor) →
      - min_val / max_val 不再更新 ← scale 冻住了!
      - calculate_qparams() 用冻结后的 scale (保持不变)
      - fake_quant_enabled=1 → 量化噪声还在 (但用的是固定的 scale)
  关键: weight 现在学习适应一个稳定的量化网格 (不是每次 batch 都变)

Phase 4: CONVERT — convert_pt2e 内部
─────────────────────────────────────────
  convert_pt2e 遍历所有 call_module(FQ) 节点:
    - 读取 scale, zero_point 从 Observer buffer 中取出 (最终固定值)
    - 删除 Observer 模块
    - 替换为: quantize → dequantize 操作对 (用取出的 scale/zp)
  Observer 实例被释放 → 存在的证据只剩图里的 Q/DQ 节点
```

**Observer 在 PTQ vs QAT 中的不同行为**：

```
PTQ (prepare_pt2e):
  Observer = 纯统计收集器
  - observer_enabled=1, fake_quant_enabled=0 (永远不做 fake_quant!)
  - 校准阶段: 跑 N 个 batch → Observer 收集 min/max → 算出 scale/zp
  - 没有 QAT 训练 → Observer 不会再次更新
  - convert_pt2e: Observer 的 scale/zp → Q/DQ 节点

QAT (prepare_qat_pt2e):
  Observer → 升级为 FakeQuantize
  - observer_enabled=1, fake_quant_enabled=1 (两个都开!)
  - 校准阶段: Observer 更新 scale + FakeQuant 注入噪声
  - QAT 训练: Observer 冻结, FakeQuant 继续注入噪声
  - convert_pt2e: FakeQuantize 的 scale/zp → Q/DQ 节点

关键差异: PTQ 从来没有 fake_quant — 校准完直接转换
         QAT 有完整的 fake_quant — 模型训练在量化噪声下
```

**Observer 在 PT2E 图中的位置 — 一个具体例子**：

```
prepare 后的图 (MiniModel, 简化):

  %x (placeholder, [1,3,32,32])
    ↓
  %fq_act_0 : call_module[activation_post_process_0] (%x)
    │  ↑ 这个模块是 FakeQuantize(MinMaxObserver)
    │  ↑ observer_enabled=1 → 每个 batch 更新 min_val/max_val
    │  ↑ fake_quant_enabled=1 → 量化噪声注入
    │  ↑ output = fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)
    ↓
  %x_q : [1,3,32,32] FP32, 256 个离散值
    ↓
  %conv2d : call_function[aten.conv2d] (%x_q, %w_q, %bias)
    ↓
  %fq_act_1 : call_module[activation_post_process_1] (%conv2d)
    │  ↑ 另一个 Observer 实例 — 独立的 min_val/max_val!
    │  ↑ 这一层激活的 scale 和上一层不同 (数据分布不同)
    ↓
  %y_q : [1,64,32,32] FP32, 256 个离散值
    ↓
  ...

每个 call_module(FQ) 节点 = 一个独立的 Observer 实例
每个实例有自己的 min_val / max_val / scale / zero_point
不同层的 scale 可以完全不同 — 这就是量化能"自适应"各层分布的关键
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

## 11. Quantizer 深度拆解：PT2E 的"量化大脑"

PT2E 用 `Quantizer` 替代 FX Mode 的 `QConfigMapping`。但 Quantizer 不只是"配置替换"——
它是一个完整的**图分析 + 标注引擎**。这节把它从 API 到内部实现彻底拆开。

> 注：QConfigMapping（FX Mode 使用）的 API 可参考 [§11 的旧版本](./QConfigMapping)，本章聚焦 PT2E 的 Quantizer。

### 11.1 Quantizer 的完整 API

```python
from torch.ao.quantization.quantizer import Quantizer

class Quantizer(ABC):
    """PT2E 的量化标注引擎 — 基类定义了三个核心方法"""

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        """Step 1 (可选): 在标注前对图做预处理变换
        例如: XNNPACKQuantizer 在这里融合 Conv+BN 模式
        如果不 override → 什么都不做
        """
        return model

    def annotate(self, model: GraphModule) -> GraphModule:
        """Step 2 (必须): 遍历图节点, 给每条边打上 QuantizationSpec 标注
        核心逻辑: 对每个 call_function 节点, 根据 op 类型决定:
          - input[0] 需要什么 Spec  (新建/共享/推导/固定?)
          - input[1] 需要什么 Spec
          - output 需要什么 Spec
        标注存储在 node.meta["quantization_annotation"] 中
        """
        raise NotImplementedError

    def validate(self, model: GraphModule) -> None:
        """Step 3 (可选): 验证标注是否正确
        例如: XNNPACKQuantizer 检查没有未标注的 op、没有冲突的共享 Spec
        如果不 override → 什么都不做
        """
        pass
```

**三个方法在 prepare 中的调用顺序**：

```python
# quantize_pt2e.py: prepare_qat_pt2e 内部

def prepare_qat_pt2e(model, quantizer):
    # Step 1: 预处理 (可选)
    quantizer.transform_for_annotation(model)
    # Step 2: 标注 (核心)
    quantizer.annotate(model)
    # Step 3: 验证 (可选)
    quantizer.validate(model)
    # Step 4: FQ 插入 (由 prepare.py 执行, 不归 Quantizer 管)
    model = prepare(model, node_name_to_scope, is_qat=True)
    return model
```

### 11.2 annotate() 的内部工作机制

`annotate()` 是 Quantizer 的核心——它遍历图中的**每个节点**，对节点的**每条输入边**做标注决策。

**遍历图的逻辑**：

```python
class XNNPACKQuantizer(Quantizer):
    def annotate(self, model: GraphModule) -> GraphModule:
        for node in model.graph.nodes:
            # 只处理 call_function 节点 (ATen op 调用)
            #   placeholder (输入) — 不标注
            #   get_attr (参数) — 不标注 (参数被消费它的 op 标注)
            #   output — 不标注
            if node.op != 'call_function':
                continue

            # ★ 核心分派: 根据 node.target (ATen op 类型) 决定标注策略
            if node.target in CONV_OPS:
                self._annotate_conv(node)
            elif node.target in LINEAR_OPS:
                self._annotate_linear(node)
            elif node.target == torch.ops.aten.add.Tensor:
                self._annotate_add(node)
            elif node.target == torch.ops.aten.cat.default:
                self._annotate_cat(node)
            elif node.target in PASSTHROUGH_OPS:
                continue  # ReLU, MaxPool 等 — 不标注
            elif node.target in FIXED_SCALE_OPS:
                self._annotate_fixed_scale(node)
            # 未匹配的 op → 不标注 (保守策略: 未知 op 不量化)
```

**标注存储格式**：

```python
# node.meta["quantization_annotation"] = QuantizationAnnotation(
#     input_qspec_map={
#         node.args[0]: QuantizationSpec(...),    # input[0] 的 Spec
#         node.args[1]: SharedQuantizationSpec(...),  # input[1] 共享
#         node.args[2]: DerivedQuantizationSpec(...), # bias 推导
#     },
#     output_qspec=QuantizationSpec(...),  # 输出边 (可选)
#     _annotated=True,
# )
```

**具体标注过程 — Conv2d 为例**：

```python
def _annotate_conv(self, node):
    """对 aten.conv2d(input, weight, bias, ...) 的三个参数标注"""
    act_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,     # 激活: per-tensor
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(...),
    )
    wt_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,  # 权重: per-channel!
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(...),
    )
    bias_qspec = DerivedQuantizationSpec(
        derived_from=[(node, node.args[0]), (node, node.args[1])],
        derive_qparams_fn=derive_bias_qparams,
        dtype=torch.int32,
    )

    # 写入标注
    annotate_edge(node, 0, act_qspec)   # input activation → 新建 FQ
    annotate_edge(node, 1, wt_qspec)    # weight → 新建 per-channel FQ
    annotate_edge(node, 2, bias_qspec)  # bias → 推导
```

**具体标注过程 — Add 为例**：

```python
def _annotate_add(self, node):
    """对 aten.add(a, b) 的两个输入标注 — 第二个共享第一个的 FQ"""
    annotate_edge(node, 0, QuantizationSpec(...))     # input[0] → 新建 FQ
    annotate_edge(node, 1, SharedQuantizationSpec(     # input[1] → 共享!
        edge_or_node=(node, node.args[0])             # "和 input[0] 共享"
    ))
```

### 11.3 分派表：Quantizer 怎么识别不同 op

PT2E 图里的节点是 call_function，`node.target` 就是 ATen op。Quantizer 通过匹配 `node.target` 来识别 op 类型：

```python
# XNNPACKQuantizer 的 op 分派表 (简化版)

CONV_OPS = [
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv_transpose2d.default,
]

LINEAR_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,      # Linear 的 ATen 底层形式
]

ADD_LIKE = [
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
]

# ★ 关键: capture 后的图用 ATen op 而非 nn.Module!
# conv2d → torch.ops.aten.conv2d.default
# ReLU  → torch.ops.aten.relu.default
# Add   → torch.ops.aten.add.Tensor
```

### 11.4 set_global / set_object_type / set_module_name 怎么工作的

用户调用 `quantizer.set_global(config)` 时，Quantizer 把这些"规则"存起来：

```python
class XNNPACKQuantizer(Quantizer):
    def __init__(self):
        self.global_config = None
        self.object_type_configs = {}      # {nn.Conv2d: config}
        self.module_name_configs = {}      # {"conv1": config}

    def set_global(self, config):
        self.global_config = config

    def set_object_type(self, obj_type, config):
        self.object_type_configs[obj_type] = config

    def set_module_name(self, name, config):
        self.module_name_configs[name] = config

    # annotate() 中使用这些配置:
    def annotate(self, model):
        for node in model.graph.nodes:
            # Step 1: 根据 node.target 映射回 nn.Module 类型 (通过 node.meta)
            nn_module_stack = node.meta.get("nn_module_stack", {})
            module_type = self._get_module_type(nn_module_stack)

            # Step 2: 按优先级查找配置
            #   精确名 > 正则 > 类型 > 全局
            config = self._resolve_config_for_node(node, module_type)

            # Step 3: 用 config 标注
            if config:
                self._annotate_node(node, config)
```

**优先级链的解析逻辑**：

```python
def _resolve_config_for_node(self, node, module_type):
    # 1. 精确模块名 → 最高优先级
    module_name = self._get_module_name(node)
    if module_name in self.module_name_configs:
        return self.module_name_configs[module_name]

    # 2. 按对象类型 (nn.Conv2d / nn.Linear 等)
    if module_type in self.object_type_configs:
        return self.object_type_configs[module_type]

    # 3. 全局默认
    if self.global_config:
        return self.global_config

    # 4. 都没匹配 → 不量化
    return None
```

### 11.5 从零写一个自定义 Quantizer

把上面的知识串起来——手写一个指定"只量化 Conv2d 的 weight (4-bit)、不量化激活"的 Quantizer：

```python
from torch.ao.quantization.quantizer import (
    Quantizer, QuantizationSpec, SharedQuantizationSpec,
    DerivedQuantizationSpec, FixedQParamsQuantizationSpec,
)
from torch.ao.quantization.observer import PerChannelMinMaxObserver

class WeightOnlyQuantizer(Quantizer):
    """自定义 Quantizer: 只量化 Conv2d 的 weight 到 4-bit, 激活不量化"""

    def __init__(self, n_bits=4):
        super().__init__()
        self.n_bits = n_bits

    def transform_for_annotation(self, model):
        # 不做预处理
        return model

    def annotate(self, model):
        for node in model.graph.nodes:
            if node.op != 'call_function':
                continue

            # ★ 只标注 Conv2d 的 weight
            if node.target == torch.ops.aten.conv2d.default:
                # input[0] (激活): 不标注 → 不量化
                # input[1] (weight): per-channel 量化
                qmax = 2 ** (self.n_bits - 1) - 1
                self.annotations[(node, node.args[1])] = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-qmax,
                    quant_max=qmax,
                    qscheme=torch.per_channel_symmetric,
                    observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        quant_min=-qmax,
                        quant_max=qmax,
                    ),
                )
                # input[2] (bias): 推导 (从 weight scale 推导 bias scale)
                self.annotations[(node, node.args[2])] = DerivedQuantizationSpec(
                    derived_from=[(node, node.args[1])],
                    derive_qparams_fn=lambda obs: (
                        obs[0].calculate_qparams()[0],
                        torch.tensor([0], dtype=torch.int32)
                    ),
                    dtype=torch.int32,
                    quant_min=-(2**31),
                    quant_max=2**31-1,
                    qscheme=torch.per_tensor_symmetric,
                )

            # Add / ReLU / Linear 等 → 全部不标注 → 保持 FP32
            # (未标注的 op = conv2d 的 input[0] = 永远不量化)

    def validate(self, model):
        # 验证: 所有 call_function 节点要么被标注, 要么不是我们关心的 op
        for node in model.graph.nodes:
            if node.op == 'call_function':
                has_annotation = any(
                    (node, arg) in self.annotations
                    for arg in node.args if isinstance(arg, torch.fx.Node)
                )
                # WeightOnly 策略: 未标注 = 故意保持 FP32, 不报错
                pass

# 使用:
quantizer = WeightOnlyQuantizer(n_bits=4)
prepared = prepare_qat_pt2e(gm, quantizer)
# prepare 后的图: 只有 weight 路径上有 FQ 节点, 激活路径保持 FP32
```

### 11.6 QConfigMapping vs Quantizer — 什么时候用哪个

```
QConfigMapping (FX Mode):
  - 适合: 标准 CNN/ResNet 模型的量化
  - 定位: 给标准 PyTorch 模型 (nn.Module 树结构) 做量化
  - 配置粒度: 按模块名 / 模块类型 / 正则
  - 缺点: 不支持自定义 op 的量化逻辑, 后端适配隐藏在 Observer 选择中

Quantizer (PT2E): ★ 推荐
  - 适合: 任何模型 (包括自定义 op、动态控制流)
  - 定位: 给 ATen 图做量化标注
  - 配置粒度: 按 ATen op 类型 / 图节点关系
  - 优点: 完全可定制 — 想量化什么、怎么量化, 完全由你控制
  - 换后端 = 换 Quantizer: XNNPACK vs TensorRT vs 自定义

如果你是"用标准模型做标准量化" → QConfigMapping 够用
如果你是"自定义量化策略、非标准 op、特定后端的精度优化" → 必须用 Quantizer
```

---

### 11.7 Q/DQ 插入规则全表：哪些算子要 Q/DQ，哪些不要

> 这是从 [Stage 0 §7.8 Q/DQ 概念框架](../Stage0_量化基础与硬件基石.md) 展开的**完整工程速查表**。Stage 0 给了你核心判据——"值变 → Q/DQ，形变/搬运 → 不插"——这里把那个判据展开到每一种 ATen op、给出 ASCII 数据流图、常见误判、以及对应的 QSpec 代码映射。

---

#### 11.7.1 五类重新梳理（含数据流图）

**类别 A：INT8 kernel 算子 → 全 Q/DQ**

这是量化推理的"主角"——硬件有专门的 INT8 指令。

```
   input (FP32)
      │
      ▼
  ┌──────────────┐
  │ Quantize      │  ← Q_input:  r → q_a ∈ [0,255] (per-tensor, 非对称)
  └──────────────┘
      │ INT8                         ┌─────────────────────┐
      ▼                              │ weight (FP32)       │
  ┌──────────────┐                   │   ↓                 │
  │ Conv / Linear │ ← INT8×INT8     │ Quantize            │ ← Q_weight: per-channel, 对称
  │ ↓ INT32 acc   │   乘加            │   ↓ INT8            │
  │ ↓ requantize  │   INT32→INT8     │                     │
  └──────────────┘                   └─────────────────────┘
      │ INT8
      ▼
  ┌──────────────┐
  │ Dequantize    │  ← DQ:  q_out → r_out (FP32)
  └──────────────┘
      │ FP32 ← 下一层自己再决定 Q 不 Q
```

| 算子 | 权重 Q | 激活输入 Q | 激活输出 DQ | weight Q 粒度 |
|------|:--:|:--:|:--:|------|
| Conv1d / Conv2d / Conv3d | ✅ | ✅ | ✅ | per-channel (axis=0) |
| ConvTranspose1d/2d/3d | ✅ | ✅ | ✅ | per-channel (axis=0) |
| Linear (FC) | ✅ | ✅ | ✅ | per-channel (axis=0) |
| LSTM / GRU (INT8 kernel) | ✅ | ✅ | ✅ | per-channel |

**在 PT2E 里怎么实现**：weight → `QuantizationSpec(dtype=torch.int8, ...)`；activation input → `QuantizationSpec(dtype=torch.int8, ...)`；activation output → `QuantizationSpec(dtype=torch.int8, ...)`（这三条 Spec 各自独立，共享输出端也可以选无 Spec）。

---

**类别 B：多输入算子 → 共享 Q + DQ**

Add、Concat、Mul 有多个输入。如果 branch1 的 S₁=0.02、branch2 的 S₂=0.05，同一个实数对应不同整数——直接运算无意义。

```
  关键问题:
    branch1: q₁=100, S₁=0.02 → 实数值 2.0
    branch2: q₂=40,  S₂=0.05 → 实数值 2.0
    → 整数 100 ≠ 40，ADD(100,40) = 140 无意义
```

**解决方案**：两路先统一到同一个 scale。

```
  branch1 (INT8, S₁=0.02)
      │
      ▼
  ┌──────────────┐
  │ Dequantize    │  ← 回到 FP32
  └──────────────┘
      │ FP32
      ▼
  ┌──────────────┐
  │ Quantize      │  ← 用共享 S_shared 重新量化
  └──────────────┘               branch2 同样走 DQ → Q(shared)
      │ INT8 (S_shared)           │
      ▼                          ▼
  ┌──────────────┐
  │ Add / Concat  │ ← 两个输入 scale 相同 → 整数运算有意义
  └──────────────┘
      │ INT8
      ▼
  ┌──────────────┐
  │ Dequantize    │
  └──────────────┘
      │ FP32
```

| 算子 | 输入侧 | 输出侧 | 特殊要求 |
|------|:--:|:--:|------|
| Add | 共享 Q | DQ | 两输入 scale 必须统一 |
| Sub | 共享 Q | DQ | 同上 |
| Mul | 共享 Q | DQ | INT8 乘法有指令，但 scale 必须统一 |
| Concat | 共享 Q | DQ | 拼接 dim 上每段 scale 要相同 |
| Cat | 共享 Q | DQ | 同 Concat |

**在 PT2E 里怎么实现**：输入侧 → `SharedQuantizationSpec(edge_or_node)`，所有输入分支指向**同一个** FQ/ Observer 实例。

---

**类别 C：没有 INT8 kernel → FP32 fallback → DQ + Q**

会改变值，但硬件没有对应 INT8 指令。

```
  input (INT8)
      │
      ▼
  ┌──────────────┐
  │ Dequantize    │  ← 先 DQ: q → r (FP32)
  └──────────────┘
      │ FP32
      ▼
  ┌──────────────┐
  │ Sigmoid / Tanh│  ← FP32 计算
  │ / Softmax     │
  │ / LayerNorm   │
  │ / GELU(无LUT) │
  └──────────────┘
      │ FP32
      ▼
  ┌──────────────┐
  │ Quantize      │  ← 再 Q: r → q (INT8)
  └──────────────┘
      │ INT8
```

| 算子 | 输入侧 | 输出侧 | 备注 |
|------|:--:|:--:|------|
| Sigmoid | DQ | Q | 有 LUT 则跳到类别 D |
| Tanh | DQ | Q | 有 LUT 则跳到类别 D |
| GELU | DQ | Q | 有 LUT 则跳到类别 D |
| SiLU / Swish | DQ | Q | 同上 |
| Softmax | DQ | Q / 保持 FP32 | LUT 需要 64KB（256×256），太大，通常 FP32 fallback |
| LayerNorm | DQ | Q | 需要算 mean/std，没有 INT8 kernel |
| BatchNorm | — | — | fuse 阶段已吸收进 Conv，推理图中不存在 |

---

**类别 D：纯搬运 → 不插任何 Q/DQ**

不改变元素的值——只改形状、顺序、或选子集。

```
  INT8 [64, 27]  --reshape-->  INT8 [64, 3, 3, 3]
  ↑ 1728 个 INT8             ↑ 还是那 1728 个 INT8
  值不变 → 不需要 DQ→FP32→reshape→Q→INT8
```

| 算子 | Q/DQ？ | 原因 |
|------|:--:|------|
| Reshape / View / Flatten | ❌ | 只改 shape/stride |
| Transpose / Permute | ❌ | 只改 stride |
| Squeeze / Unsqueeze | ❌ | 增删维度 |
| Slice / Split / Chunk | ❌ | 切子集 |
| Pad / ZeroPad | ❌ | 填 Z 值（INT8 的零） |
| **MaxPool2d / AvgPool2d** | ❌ | Max 比大小在 INT8 下等价；Avg 在 INT32 累加取整 |
| Dropout | ❌ | 推理时 Identity |
| Identity | ❌ | 透传 |
| **ReLU** | ❌ | LUT 查表：`lut[q] = max(q, Z)`，不需要 Q/DQ |

**Pool 不需要 Q/DQ 的证明**：

```
MaxPool (2×2, stride=2) 消费 INT8:
  INT8 [1,64,56,56]
      │  对每个 2×2 窗口取 max INT8
      ▼
  INT8 [1,64,28,28]
  
  INT8 比大小 = FP32 比大小（同一个 scale 下是单调映射）:
    int8: max(10,25,8,42) = 42
    fp32: max(0.2,0.5,0.16,0.84) = 0.84 → 量化回去 = 42 ✓
```

---

**类别 E：特殊情形**

| 组件 | Q/DQ | 原因 |
|------|:--:|------|
| Bias | Derived | `S_bias = S_w × S_a`，不插独立 Q。→ `DerivedQuantizationSpec` |
| BatchNorm | ❌ | fuse 阶段吸收进 Conv，不存在 |
| 模型输入 | Q（首层之前） | 输入 FP32 → 第一层之前必须 Q |
| 模型输出 | DQ（尾层之后） | Logits/BBox 需要 FP32 |
| Residual Add | 共享 Q（类别 B） | ResNet skip connection |
| Multi-Head Attn | 各 MatMul 独立 Q/DQ | QKV 各走各的 Linear，softmax 走类别 C |

---

#### 11.7.2 完整速查表（30 种算子，按字母序）

```
算子              权重 Q    输入 Q    输出 DQ   对应 QSpec 类型
──────────────────────────────────────────────────────────────────
Add                       共享 Q       ✅       SharedQuantizationSpec
AvgPool2d                  ❌         ❌       (不标注)
BatchNorm                  —          —       (图中不存在)
Bias              Derived  —          —       DerivedQuantizationSpec
Cat                        共享 Q       ✅       SharedQuantizationSpec
Chunk                       ❌         ❌       (不标注)
Conv1d/2d/3d        ✅      ✅         ✅       weight+input→QSpec, output→QSpec
ConvTranspose       ✅      ✅         ✅       weight+input→QSpec, output→QSpec
Concat                      共享 Q       ✅       SharedQuantizationSpec
Dropout                      ❌         ❌       (不标注)
Flatten                      ❌         ❌       (不标注)
GELU                        DQ          Q       input→DQ(不标注), output→QSpec
Identity                     ❌         ❌       (不标注)
LayerNorm                   DQ          Q       input→DQ(不标注), output→QSpec
Linear               ✅      ✅         ✅       weight+input→QSpec, output→QSpec
LSTM                 ✅      ✅         ✅       weight+input→QSpec, output→QSpec
MaxPool2d                    ❌         ❌       (不标注)
Mul                        共享 Q       ✅       SharedQuantizationSpec
Pad                          ❌         ❌       (不标注)
Permute                      ❌         ❌       (不标注)
ReLU                         ❌         ❌       (不标注 — LUT 消费 INT8)
Reshape                      ❌         ❌       (不标注)
Sigmoid                     DQ          Q       input→DQ(不标注), output→QSpec
SiLU/Swish                  DQ          Q       input→DQ(不标注), output→QSpec
Slice                        ❌         ❌       (不标注)
Softmax                     DQ          Q/FP32  input→DQ(不标注), output→QSpec
Split                        ❌         ❌       (不标注)
Squeeze                      ❌         ❌       (不标注)
Sub                         共享 Q       ✅       SharedQuantizationSpec
Tanh                        DQ          Q       input→DQ(不标注), output→QSpec
Transpose                    ❌         ❌       (不标注)
Unsqueeze                    ❌         ❌       (不标注)
View                         ❌         ❌       (不标注)
```

#### 11.7.3 常见误判

```
误判 1: "Reshape 前要 DQ，reshape 完再 Q"
  错。INT8 值不变，加了 DQ→reshape→Q 是三重浪费。

误判 2: "MaxPool 对 INT8 输入需要 DQ"
  错。Max 比大小在 INT8 和 FP32 下结果一致。

误判 3: "所有 Activation 函数都要 DQ→FP32→Q"
  错。ReLU 有 LUT（等价于 clamp(q, Z, 255)），Sigmoid/Tanh/GELU 也有 LUT。
  只有 Softmax 和 LayerNorm 绕不开 FP32 fallback。

误判 4: "Concat 本身要插 Q"
  错。Q/DQ 插在输入分支上（共享 scale），Concat 本身只是拼。

误判 5: "Pool 后面的 Conv 不需要输入 Q"
  错。Pool 输出 INT8 带上一层的 output scale。下一个 Conv 的 input Q
  是从校准数据重新估算的——和 Pool 的那个 scale 不同。
```

#### 11.7.4 从速查表到 Quantizer 代码

上面表的最后一列"对应 QSpec 类型"可以直接映射到 PT2E Quantizer：

```
速查表 "✅ Q"         →  QuantizationSpec(dtype=torch.int8, ...)
速查表 "共享 Q"       →  SharedQuantizationSpec(edge_or_node)
速查表 "Derived"     →  DerivedQuantizationSpec(derived_from)
速查表 "DQ"           →  输入侧不标注 QSpec（靠前一层输出 DQ 自然回到 FP32）
速查表 "❌" / "(不标注)" →  annotate() 里对该 op/edge 不调用 set_*
```

**"不标注" ≠ "标注传递"**——一个常见混淆点：

```
"不标注" (Reshape / Transpose / Pool / etc.):
  annotate(node):  ← 什么都不做，直接跳过
  效果: 该 op 前后没有 Q/DQ 节点
  数据在 FP32 域流动: ...DQ → Reshape(FP32) → Q...
                          ↑ 前一个 op 的 DQ    ↑ 后一个 op 的 Q
  Reshape 自己什么都不插，透明躺在 FP32 区域里

"共享 Q" (Add / Concat / Mul):
  annotate(node):  edge_a.set_(SharedQuantizationSpec(anchor=edge_a))
                   edge_b.set_(SharedQuantizationSpec(anchor=edge_a))
  效果: 两个输入边指向同一个 Q 实例——是"强制共用"不是"什么都不做"
```

不标注之所以能工作，是因为 prepare 阶段只根据 annotation map 来决定插不插 Q/DQ——没有 annotation 的地方，prepare 就当它不存在，该 op 自然落在前一个 DQ 和后一个 Q 之间的 FP32 走廊。

这就是 §11.3 的 `CONV_OPS`、`ADD_LIKE`、`PASSTHROUGH_OPS` 分组背后的设计逻辑——分组不是随机的，是按 Q/DQ 需求分的类。

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

QAT 是微调不是从头训练。`lr = 0.0001 ~ 0.00005`（比原始训练的 lr 低 10-100 倍）。使用 cosine annealing。太大会 "忘记" 预训练权重。

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
