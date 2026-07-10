# Stage 1: PyTorch 原生 QAT — 三种模式从入门到精通

> ⏱ 预计学习时间：15-25 小时 | 🎯 难度：⭐⭐
>
> **目标**：**彻底吃透 PyTorch QAT 的三种 API 模式和底层图改写机制**，能在 YOLO / ResNet / MobileNet 上根据场景选择合适的模式。
>
> **阅读建议**：这不是 API 速查手册，是一个从头讲起的故事。每一节都回答了"为什么 PyTorch 要这样设计"和"如果不这样做会怎样"。建议读完一节动手跑一下代码，再读下一节。

---

## 目录

1. [开篇：从一个训练中的 ResNet 说起](#开篇从一个训练中的-resnet-说起)
2. [知识总览：三代 API 的进化逻辑](#知识总览三代-api-的进化逻辑)
3. [1. FakeQuantize 状态机：QAT 的核心引擎](#1-fakequantize-状态机qat-的核心引擎)
4. [2. Eager Mode QAT：最基础的模式，理解"发生了什么"](#2-eager-mode-qat最基础的模式理解发生了什么)
5. [3. FX Graph Mode QAT：图改写的力量](#3-fx-graph-mode-qat图改写的力量)
6. [4. PyTorch 2 Export (PT2E) QAT：未来的标准](#4-pytorch-2-export-pt2e-qat未来的标准)
7. [5. 源码深潜：prepare_qat_fx 内部发生了什么](#5-源码深潜prepare_qat_fx-内部发生了什么)
8. [6. 源码深潜：convert_fx 内部发生了什么](#6-源码深潜convert_fx-内部发生了什么)
9. [7. Conv-BN-ReLU 融合：数学原理 + 图操作](#7-conv-bn-relu-融合数学原理--图操作)
10. [8. 自定义量化策略：QConfigMapping 和 BackendConfig](#8-自定义量化策略qconfigmapping-和-backendconfig)
11. [9. QAT 训练实战技巧](#9-qat-训练实战技巧)
12. [10. 动手实验](#10-动手实验)
13. [检验标准](#检验标准)

---

## 开篇：从一个训练中的 ResNet 说起

想象一下这个场景：你有一个已经在 ImageNet 上训练好的 ResNet18，top-1 准确率 69.76%。你想把它部署到手机上，但 FP32 的模型要 44MB，手机上跑一次推理要 80ms——太慢了。你知道 INT8 量化可以把模型缩小 4 倍，但直接用 PTQ（训后量化）的话，精度会从 69.76% 掉到 64% 左右。

这时候你需要 QAT：**让模型在训练过程中"感受"量化的存在，学会在量化噪声下依然做出正确判断**。

但"让模型感受量化"这句话在代码层面到底是什么意思？PyTorch 是怎么做到的？这就是整个 Stage 1 要回答的问题。

先给你一个直觉。假设你在学投篮，眼睛被蒙上了一层毛玻璃（相当于量化引起的信息损失）。如果你突然被蒙上眼就去比赛，肯定会投偏很多（这就是 PTQ）。但如果你在训练中就戴着毛玻璃投篮，你的肌肉会逐渐适应，最终即使看不清楚也能投进——这就是 QAT 的核心思想。而 PyTorch 的 FakeQuantize，就是那层"毛玻璃"。

---

## 知识总览：三代 API 的进化逻辑

PyTorch 量化 API 经历了三代演进。但不要把这三代理解成"1.0 → 2.0 → 3.0"的简单升级，它们背后反映了 PyTorch 团队对"量化"这件事理解的变化。

```
2018 ─────→ 2020 ─────→ 2022 ─────→ 2024+
Eager Mode    FX Mode     PT2E       Stable PT2E
  (旧)         (主流)     (原型)      (新标准)
```

### Eager Mode（2018-2020）：朴素的想法

最早的量化 API 设计思路很简单：**量化就是往模型里塞一些节点**。在 Eager Mode 下，`prepare_qat(model)` 会递归遍历所有子模块，在每个需要量化的模块内部，给 weight 挂一个 FakeQuantize，给输出挂一个 FakeQuantize。然后训练就行了。

这个方案直观，但有一个致命问题：**FakeQuantize 被塞在 `forward()` 函数里面，你看不到它。** 想象你在 debug 一个 100 层的网络，你想找出哪一层的量化误差最大——在 Eager Mode 下你只能靠打印 log，因为 FakeQuantize 不是计算图上的独立节点，它"藏"在模块内部。

### FX Graph Mode（2020-2024）：图即一切

PyTorch 1.8 引入了 `torch.fx`——一个能把任意 `nn.Module` 转成一张**有向无环图（DAG）**的工具。这张图把模型的所有操作（卷积、激活、加法、拼接...）展开成显式的节点。

这对量化意味着什么？**FakeQuantize 不再"藏"在模块里，而是作为独立的图节点插入**。你可以 `print_readable()` 看到整张图，定位哪一个 FakeQuantize 出了问题。你可以对图做任意变换（融合、删除、重排），不再受限于模块的层级结构。

**这就是 FX Graph Mode 的核心哲学：所有的量化操作，都是对计算图的变换。**

### PT2E（2022-现在）：为 export 而生

FX Graph Mode 很好，但它的 `symbolic_trace` 有一个硬伤：**它只是"跑一遍 forward 记录执行路径"，不是真正的程序分析**。如果你的模型里有一个 `if x.sum() > 0: return x*2 else: return x*3`，`symbolic_trace` 只记录了实际执行的那条分支，而不是整个程序。

PT2E 用 `torch.export` 替代了 `symbolic_trace`。`torch.export` 是一个真正的程序 tracer，它会分析 AST（抽象语法树），展开所有控制流。此外，PT2E 引入了 `Quantizer` 这个可插拔的后端量化器——不同硬件（XNNPACK、TensorRT、高通 NPU）有不同的量化约束，Quantizer 封装了这些差异。

**学习策略**：主力使用 FX Graph Mode，了解 Eager Mode（因为 AIMET 等框架的内部类似它），跟上 PT2E（因为它是未来）。

---

## 1. FakeQuantize 状态机：QAT 的核心引擎

> 如果你整个 Stage 1 只记住一个概念，就记住这个状态机。它是理解所有 QAT 框架（PyTorch、AIMET、NNCF、PPQ）的通用钥匙。

### 1.1 先理解问题，再理解方案

QAT 训练需要分阶段进行。为什么要分阶段？我们用投篮的比喻来想：

- **阶段 1（观察）**：你先不蒙眼，正常投篮。教练在旁边记录你每次投篮的落点范围（min/max）。这个阶段的目标是搞清楚"量化网格"应该铺在哪个区间。
- **阶段 2（适应）**：戴上毛玻璃，基于阶段 1 确定的网格参数，反复练习。你的肌肉逐渐学会在模糊视觉下调整发力。这个阶段 grid 是固定的，只调整你的"投篮参数"（网络权重）。
- **阶段 3（比赛）**：正式比赛，戴毛玻璃，不调整任何东西，纯测试。

对应到 QAT，这三个阶段分别对应不同的 `observer_enabled` 和 `fake_quant_enabled` 状态：

```
阶段 1（Calibration / 观察期）—— 通常只有前几个 epoch 或前几个 batch:
  observer_enabled  = True   → 我正在记录激活值的 min 和 max
  fake_quant_enabled = False  → 我还没开始做量化，数据正常通过

  目的：收集 min / max → 算出最佳的 scale 和 zero_point

阶段 2（QAT Training / 适应期）—— 大部分 epoch：
  observer_enabled  = False  → scale 已经固定了，不再更新
  fake_quant_enabled = True   → 现在开始做量化-反量化，模型感受量化噪声

  目的：让模型在量化噪声下学会补偿

阶段 3（Evaluation / 验证期）—— 每个 epoch 结束时的验证 & 最终测试：
  observer_enabled  = False
  fake_quant_enabled = True
  model.eval()               → BN 切换到 running stats

  目的：测试模型在真正推理条件下的表现
```

这里有个容易被忽略的细节：**为什么阶段 2 中 `fake_quant_enabled = True` 但 observer 关掉了？**

因为如果你在训练过程中持续更新 scale（observer 开着），那每个 batch 的 scale 都会轻微不同。这意味着前一个 batch 模型学到的是"在 scale=0.03 的量化噪声下如何补偿"，下一个 batch 噪声参数变了，之前的补偿就白学了。**固定的 scale 让优化目标稳定，模型才能收敛。**

不过也有例外——LSQ（Learned Step Size Quantization）就是故意让 scale 变成可学习参数，通过梯度来调整。那是 Stage 2 的内容，但你现在需要先理解"固定 scale 的 QAT"是怎么回事，才能理解 LSQ 为什么要创新。

### 1.2 一张图理解状态切换

```
           calibrate()
  ┌────────────┐ ─────────────→ ┌────────────────┐
  │  Observing │                │  QAT Training  │
  │  obv=T     │                │  obv=F, fq=T   │
  │  fq=F      │                │  (scale固定)   │
  └────────────┘                └───────┬────────┘
                                        │
                                   convert()
                                        │
                                        ▼
                                ┌────────────────┐
                                │  INT8 Model    │
                                │  真正 INT8 推理│
                                └────────────────┘
```

注意这张图的最后一跳：`convert()`。QAT 训练结束后，FakeQuantize 节点在模型里面还是"假的"——它们只是把 float 离散化又转回 float，本质上还是 FP32 计算。`convert()` 这一步把所有 FakeQuantize 替换为真正的量化/反量化节点（Quantize / DeQuantize），此时模型才开始做真正的 INT8 推理。**在 QAT 训练期间，你实际上一直在用 FP32 算，只是数据被 FakeQuantize 离散化了。**

### 1.3 源码层面的状态管理：两个独立开关为什么比一个模式标志好

现在我们看代码。PyTorch 用两个独立开关而不是一个 `mode` 枚举（比如 `mode="observe"`），这是一个深思熟虑的设计决策。

为什么？因为真实场景中，你可能需要**有些层永远不量化、有些层同时观察+量化**。如果只有一个 `mode`，你要么全部观察、要么全部量化，无法混合。而两个独立开关的组合给了最大灵活性：

| observer_enabled | fake_quant_enabled | 含义 |
|---|---|---|
| True | False | 只观察不量化（calibration 期） |
| False | True | 只量化不更新 scale（QAT 训练期） |
| True | True | 同时观察和量化（某些高级场景） |
| False | False | 直通，不量化（FP32 基线对比 / 特殊层） |

```python
# ===== 深入理解 FakeQuantize 的状态管理 =====
# 对应 PyTorch 源码: torch/ao/quantization/fake_quantize.py
#
# 在你读下面代码之前，先记住一件事：
# 这个类的 forward() 函数在 QAT 训练期间会被调用几万次，
# 所以它被设计得非常高效（两个 flag 检查只是简单的 if 判断）。
# 但同时它又要提供足够的灵活性来支持各种场景。

import torch
import torch.nn as nn

class FakeQuantStateTracker(nn.Module):
    """
    模拟 PyTorch FakeQuantize 的完整状态机

    为什么继承 nn.Module 而不是普通类？
    因为 FakeQuantize 需要作为子模块被注册到模型中——
    这样它才能跟着 model.to(device)、model.state_dict() 等操作。
    """

    def __init__(self, quant_min=-128, quant_max=127):
        super().__init__()

        # 量化范围：对于有符号 INT8，这是 -128 到 127
        self.quant_min = quant_min
        self.quant_max = quant_max

        # ===== 两个核心开关 =====
        # 为什么用 tensor 而不是 bool？
        # 因为在 TorchScript 模式下，bool 类型的操作可能不被支持。
        # 用 uint8 tensor 可以确保在 eager / script / fx 三种模式下行为一致。
        self.observer_enabled = torch.tensor([1], dtype=torch.uint8)
        self.fake_quant_enabled = torch.tensor([1], dtype=torch.uint8)

        # ===== 统计量 — 用 register_buffer 注册 =====
        # register_buffer 做三件事：
        # 1. 跟着 model.to(device) 自动迁移到 GPU
        # 2. 包含在 model.state_dict() 中（可以被保存/加载）
        # 3. 不作为可训练参数（optimizer 不会更新它）
        # 对于 scale 和 zero_point 这种"校准得到、训练时固定"的量，用 buffer 正合适
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

    # ===== 四个状态控制函数 =====
    # PyTorch 提供了 enable/disable_observer 和 enable/disable_fake_quant
    # 四个函数，而不是一个 set_mode() 函数。
    # 这让你可以用 model.apply(disable_observer) 一键冻结所有层的 observer

    def enable_observer(self):
        self.observer_enabled[0] = 1

    def disable_observer(self):
        """冻结 scale —— QAT 训练期最关键的操作"""
        self.observer_enabled[0] = 0

    def enable_fake_quant(self):
        self.fake_quant_enabled[0] = 1

    def disable_fake_quant(self):
        """不做量化-反量化——用于对比实验（看裸 FP32 精度）"""
        self.fake_quant_enabled[0] = 0

    def calculate_qparams(self):
        """从收集的 min/max 计算 scale 和 zero_point"""
        abs_max = torch.maximum(self.min_val.abs(), self.max_val.abs())
        self.scale = abs_max / (self.quant_max / 2.0)
        self.zero_point.zero_()  # 对称量化 zero_point = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        这是 QAT 的灵魂。每一步都值得理解。

        先思考一个问题：forward 函数返回的是什么类型？
        答：还是 float32。FakeQuantize 做的事是：
           float32 → 量化成 int8 → 反量化回 float32
        所以语义上 x 被"离散化"了，但数据类型还是 float32。
        这就是为什么它叫 "Fake" Quantize —— 它模拟了量化的效果，
        但实际计算还是 FP32。

        再思考一个问题：torch.round() 的梯度怎么处理？
        round() 的导数处处为 0（除了在 .5 处不连续），如果直接用，
        梯度流就断了。PyTorch 对 round() 使用了 Straight-Through Estimator
        (STE)：前向做真实的 round，反向直接把梯度传过去，当作
        round() 不存在。这意味着：
          - 前向：你确实看到了量化效果（输出是离散值）
          - 反向：梯度穿透了 round()，权重能够被更新来补偿量化误差
        """
        if self.observer_enabled[0] == 1:
            # 使用 detach() 是因为我们不想让 min/max 的更新
            # 影响梯度计算图。min/max 只是统计量，不参与反向传播。
            self.min_val = torch.min(self.min_val, x.detach().min())
            self.max_val = torch.max(self.max_val, x.detach().max())
            self.calculate_qparams()

        if self.fake_quant_enabled[0] == 1:
            # 量化
            x_q = torch.round(x / self.scale + self.zero_point)
            x_q = torch.clamp(x_q, self.quant_min, self.quant_max)
            # 反量化 —— x 现在是 float 但值域被限制在 256 个离散值内
            x = (x_q - self.zero_point) * self.scale

        return x
```

### 1.4 用一个具体数值跑一遍，建立直觉

```python
# ===== 让我们过一遍具体的数值 =====
fq = FakeQuantStateTracker()

# --- 阶段 1: 观察 ---
fq.enable_observer()
fq.disable_fake_quant()

# 模拟第一批数据通过
x1 = torch.tensor([0.5, -0.3, 1.2, -0.8, 2.1])
out1 = fq(x1)
print(f"输入: {x1.tolist()}")
print(f"输出: {out1.tolist()}")  # 应该等于输入（没有做量化）
print(f"scale 更新为: {fq.scale.item():.4f}")  # ≈ 2.1/127 ≈ 0.0165

# 模拟第二批数据通过——min/max 继续拓宽
x2 = torch.tensor([3.5, -2.0, 0.1])  # 这批数据范围更大
out2 = fq(x2)
print(f"第二批后 scale 更新为: {fq.scale.item():.4f}")  # ≈ 3.5/127 ≈ 0.0276

# --- 阶段 2: QAT 训练 ---
fq.disable_observer()  # scale 冻结在 0.0276
fq.enable_fake_quant()  # 开始做量化

x3 = torch.tensor([1.5, 0.8, -1.2])
out3 = fq(x3)
print(f"输入: {x3.tolist()}")
print(f"FakeQuant 后: {out3.tolist()}")
# 你会看到输出被"离散化"了——相邻的两个接近值可能变得相同
# 比如 1.5/0.0276≈54.3→round→54→*0.0276≈1.49
# 这就是"量化噪声"——原始 1.5 变成了 1.49

# 验证 STE：梯度是否穿透了？
x_grad = torch.tensor([1.5, 0.8, -1.2], requires_grad=True)
fq.enable_fake_quant()
fq.disable_observer()
out_grad = fq(x_grad)
loss = out_grad.sum()
loss.backward()
print(f"梯度: {x_grad.grad}")  # 应该是 [1, 1, 1]——STE 让梯度直通
```

**关键发现**：第二阶段的输出 1.49 和输入 1.5 之间的差距（0.01）就是量化误差。QAT 的目标就是让网络的其他部分学会"忽略"这个误差——比如下一层的权重会自动调整，让 1.49 产生的效果和 1.5 一样。

### 1.5 别忘了 eps：一个小参数能让你 debug 一整天

```python
# ===== Observer 中的 eps =====
# 在 torch/ao/quantization/observer.py 中，每个 observer 初始化都有一个 eps 参数

eps = torch.finfo(torch.float32).eps  # ~1.19e-7

# 为什么需要它？考虑这个 edge case：
# 某层激活值恰好全部是同一个常数（比如 2.0）。
# min = max = 2.0
# S = (2.0 - 2.0) / 255 = 0  → 除零！
# eps 确保 scale 最少有 eps 大小，避免 NaN。

# 你几乎永远不会遇到这个 edge case，但一旦遇到，没有 eps 的话，
# 你不会看到报错，而是看到 loss = NaN，然后你 debug 三天。
```

---

## 2. Eager Mode QAT：最基础的模式，理解"发生了什么"

> **学习定位**：Eager Mode 在 PyTorch 2.x 中已标记 deprecated（即将废弃）。但我们仍然学它，而且要仔细学。原因很简单——**Eager Mode 的设计最"原始"，它暴露出来的东西最多**。理解 Eager Mode 的 prepare_qat 对一个 module 做了什么，就读懂了所有 QAT 框架的底层逻辑。AIMET、NNCF、PPQ 的内部虽然更复杂，但核心机制和 Eager Mode 是同构的。

### 2.1 先理解 Eager Mode 的哲学

Eager Mode 的哲学可以概括为八个字：**"在模块树上挂装饰品"**。

想象一棵圣诞树（你的模型）。树干是 `nn.Module` 的继承结构（ResNet > layer1 > BasicBlock > conv1 > ...）。你要在树上挂彩灯（FakeQuantize）。Eager Mode 的做法是：**遍历整棵树的每一个树枝，找到需要装饰的位置，挂上去**。

具体到代码，`prepare_qat(model)` 做的事情是：

1. 递归遍历 `model` 的所有子模块
2. 对每一个有 `.qconfig` 属性的模块：
   - 为 weight 创建一个 `weight_fake_quant`（一个 FakeQuantize 实例）
   - 为该模块的输出创建一个 `activation_post_process`（也是一个 FakeQuantize 实例）
   - **修改这个模块的 `forward` 方法，在内部插入量化操作**

第 3 步是 Eager Mode 最巧妙也最 hacky 的地方。它不是在图层面插入新节点，而是**直接替换了模块的 forward 函数**。这就是为什么你无法在 FX 图上看到 FakeQuantize——因为它在模块内部，不在图里。

### 2.2 Eager Mode 的核心 API 调用链

让我们用一张流程图抓住主线：

```
FP32 Model                                                          INT8 Model
    │                                                                   ▲
    ├── 1. model.qconfig = get_default_qat_qconfig("fbgemm")            │
    │       每一层拿到一个"量化偏好设置"                                  │
    │                                                                   │
    ├── 2. fuse_modules(model, [["conv", "bn", "relu"]])               │
    │       把 Conv+BN+ReLU 融合成单个模块（数学等价的 FP32 变换）        │
    │       ⚠️ 必须在 prepare_qat 之前做！                               │
    │                                                                   │
    ├── 3. model_prepared = prepare_qat(model)                         │
    │       ★ 遍历 model 的子模块，在每个需要量化的地方插入 FakeQuantize │
    │       这是整个 QAT pipeline 的灵魂                                 │
    │                                                                   │
    ├── 4. train(model_prepared)  ← QAT 训练（10-20 epochs）           │
    │                                                                   │
    └── 5. model_int8 = convert(model_prepared.eval()) ───────────────┘
            把 FakeQuantize 替换为真量化节点
```

### 2.3 Step by Step 走一遍完整代码

下面不是"把代码扔给你就跑"，而是每一步都跟你讲明白**这一行在干什么、不这么写会怎样**。

```python
# ===== Eager Mode QAT: ResNet18 + ImageNet 完整示例 =====
import torch
import torch.nn as nn
import torchvision
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    fuse_modules,
    QConfig,
    MinMaxObserver, MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

# ═══════════════════════════════════════════════════════════
# Step 1: 加载预训练模型
# ═══════════════════════════════════════════════════════════
model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
model.train()  # ★ QAT 必须用 .train() 模式！
# 为什么？因为 QAT 需要 FakeQuantize 的 observer 在前几个 epoch 收集统计量，
# 而 observer 在 .eval() 模式下可能不工作（取决于实现）。
# 另外，dropout 在 .train() 和 .eval() 下行为不同，
# 但 QAT 场景下通常 dropout 概率已经很小的——预训练模型微调时一般不依赖 dropout。

# ═══════════════════════════════════════════════════════════
# Step 2: 融合 Conv+BN+ReLU
# ═══════════════════════════════════════════════════════════
# 这一步的直觉：训练好的模型做推理时，Conv+BN 可以合并成一个操作。
# 如果不在量化前合并，FakeQuantize 就会插在 Conv 和 BN 之间，
# 那时候 BN 的输入已经是量化后的值了——这和训练时完全不同，会导致精度大幅下降。
#
# fuse_modules 的输入是一个 list of lists，每个子列表是一个融合组。
# 比如 ["conv1", "bn1", "relu"] 表示把 module.conv1、module.bn1、module.relu
# 融合成一个 nni.ConvBnReLU2d 模块。
#
# ⚠️ Eager Mode 最痛苦的地方：要手动列出所有融合路径！
# 对于 ResNet18 有 20+ 个 BasicBlock，每个有 2 个 Conv+BN——得写 40+ 行。
# 这也是 FX Mode 要解决的问题之一（自动识别融合 pattern）。
model = fuse_modules(model, [
    ["conv1", "bn1", "relu"],                          # 第一层
    ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"],
    ["layer1.0.conv2", "layer1.0.bn2"],                # 没有 ReLU！
    ["layer1.1.conv1", "layer1.1.bn1", "layer1.1.relu"],
    ["layer1.1.conv2", "layer1.1.bn2"],
    # ... 实际中需要遍历所有 BasicBlock 生成完整列表
])

# ═══════════════════════════════════════════════════════════
# Step 3: 设置 qconfig
# ═══════════════════════════════════════════════════════════
# qconfig 是一个"量化偏好"对象，包含两个核心信息：
#   activation: 激活值用什么 Observer + FakeQuantize
#   weight:     权重用什么 Observer + FakeQuantize
#
# 为什么权重和激活要分开配置？
# 因为它们的分布特征完全不同：
#   权重: 大致对称，以 0 为中心 → 用对称量化 + per-channel
#   激活（ReLU 后）: 只在正半轴，有偏态 → 用非对称量化 + per-tensor
model.qconfig = get_default_qat_qconfig("fbgemm")
# fbgemm 的默认 QAT 配置:
#   weight:     PerChannelMinMaxObserver (per-channel symmetric, qint8)
#   activation: MovingAverageMinMaxObserver (per-tensor affine, quint8)

# 你也可以自己定制（当默认配置不满足需求时）：
# model.qconfig = QConfig(
#     activation=HistogramObserver.with_args(
#         dtype=torch.quint8,
#         qscheme=torch.per_tensor_affine,
#         reduce_range=False,
#     ),
#     weight=PerChannelMinMaxObserver.with_args(
#         dtype=torch.qint8,
#         qscheme=torch.per_channel_symmetric,
#     ),
# )

# ═══════════════════════════════════════════════════════════
# Step 4: prepare_qat —— ★ 整个 QAT 的灵魂
# ═══════════════════════════════════════════════════════════
# 这是 Eager Mode QAT 最核心的一行代码。它在内部做了什么？
#
# 对模型树上的每个模块 m：
#   如果 m 有 .qconfig 属性（且不为 None）：
#     1. 创建 weight_fake_quant = m.qconfig.weight()
#        （是一个 FakeQuantize 实例，继承自 Observer）
#     2. 创建 activation_post_process = m.qconfig.activation()
#     3. 把这两个附加到 m 上
#     4. 修改 m 的 forward 方法，在计算前后插入量化操作
#
# 你 notice 到第 4 步的特殊之处了吗？
# Eager Mode 实际上 WRAPS 了原始的 forward 方法。
# 这就是为什么 FX graph 上你看不到 FakeQuantize 节点——
# 它藏在被修改后的 forward 函数里面。
model_prepared = prepare_qat(model, inplace=False)

# 看一眼：哪些模块被加了量化节点？
print("=== 量化前向传播中的假量化节点 ===")
for name, module in model_prepared.named_modules():
    if hasattr(module, 'weight_fake_quant'):
        print(f"  ✓ {name}: 有权重量化")

# ═══════════════════════════════════════════════════════════
# Step 5: 配置训练组件
# ═══════════════════════════════════════════════════════════

# 学习率 —— QAT 的一个关键超参数
# 这里用 1e-4 而不是预训练时的 0.1，因为：
# QAT 是"微调"不是"从头训练"。FP32 权重已经是一个很好的解，
# 你只需要在这个解附近探索对量化噪声鲁棒的新解。
# 太大的学习率 = 你离开了原来的好解，掉到更差的区域。
optimizer = torch.optim.SGD(model_prepared.parameters(), lr=1e-4, momentum=0.9)

# ★ 冻结 BN —— 你必须做，否则 QAT 可能比 PTQ 还差
# 为什么？
# QAT 训练时，FakeQuantize 改变了激活值的分布。
# 如果 BN 仍然用 batch stats 更新 running_mean / running_var：
#   - 每个 epoch 的激活分布因为 FakeQuant 而不同
#   - BN 的 running stats 在不同 epoch 之间"漂移"
#   - 最终 saved 的 BN stats 和任何单个 epoch 的分布都不匹配 → 精度崩塌
for module in model_prepared.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()                       # 用 running stats，不用 batch stats
        module.weight.requires_grad_(False) # 不训练 γ
        module.bias.requires_grad_(False)   # 不训练 β

# ═══════════════════════════════════════════════════════════
# Step 6: QAT 训练循环
# ═══════════════════════════════════════════════════════════
num_epochs = 10
disable_observer_epoch = 3  # observer 在第 3 个 epoch 后冻结

for epoch in range(num_epochs):
    model_prepared.train()  # !!! 不要省略这行

    # observer 调度：前 3 个 epoch 收集 scale，之后固定
    if epoch >= disable_observer_epoch:
        model_prepared.apply(torch.ao.quantization.disable_observer)

    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()
        output = model_prepared(images)  # forward 自动包含 FakeQuant
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证阶段
    model_prepared.eval()
    model_prepared.apply(torch.ao.quantization.disable_observer)
    acc = evaluate(model_prepared, val_loader)
    print(f"Epoch {epoch}: QAT Accuracy = {acc:.2f}%")

# ═══════════════════════════════════════════════════════════
# Step 7: convert —— 从"假量化"到"真量化"
# ═══════════════════════════════════════════════════════════
model_prepared.eval()
model_prepared.cpu()
model_int8 = convert(model_prepared, inplace=False)
# convert() 的工作：
#   1. 冻结所有 FakeQuantize 的 scale / zero_point（不再更新）
#   2. 把 FakeQuantize 替换为真正的 Quantize / DeQuantize
#   3. 如果 backend 支持 fused INT8 op，则做融合
#      （如 Conv2d → QuantizedConv2d）
```

### 2.4 Eager Mode 的痛点（记住这些，你就能理解 FX Mode 为什么诞生）

学习 Eager Mode 最好的方式，是理解它的局限。这些局限不是 API 设计的"bug"，而是"在模块树上挂装饰品"这个哲学必然导致的后果：

| 痛点 | 根因 | 具体表现 |
|------|------|----------|
| **手动融合** | `fuse_modules` 需要精确指定每一条融合路径 | ResNet50 有 53 个 Conv+BN+ReLU，你要写 53 行 |
| **FakeQuantize 不可见** | 它藏在被 wrap 过的 `forward()` 里，不在计算图上 | 无法打印图来 debug，只能靠 forward hook |
| **动态控制流不支持** | 如果你的模型有 `if/while`，Eager Mode 不知道在哪插 FakeQuantize | `if x.sum()>0: ...` 导致 prepare_qat 行为不确定 |
| **自定义 op 难扩展** | 需要手动实现 `QuantStub`/`DeQuantStub` | 写一个新算子的量化支持要上百行 |
| **和后端解耦不够** | `qconfig` 同时包含了"什么 observer"和"什么 backend"的信息 | 换一个 backend 可能要改所有层的 qconfig |

这些问题在 FX Graph Mode 中全部得到解决。那 FX 是怎么做到的？它不把 FakeQuantize 藏在模块里了——**它把整个模型变成一张 DAG，然后在 DAG 上插入 FakeQuantize 作为独立节点。**

---

## 3. FX Graph Mode QAT：图改写的力量

### 3.1 先建立"图"的直觉

在讲 FX 量化之前，你需要先理解 `torch.fx` 是什么。这不是一个量化专用工具——它是 PyTorch 的通用程序变换框架。量化只是它的一个应用。

想象你要修改一篇英文文章。在 Eager Mode（模块树）的世界里，你只能修改"段落"和"句子"（模块），不能直接修改单词。而 FX 把整篇文章拆成一个个**单词（图节点）**——你可以精确地在任意一个词后面插入一个新词、删除一个词、或者替换一个词。

用代码来说明这个概念：

```python
import torch
import torch.fx

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)      # 节点1: call_module("conv")
        x = self.relu(x)      # 节点2: call_module("relu")
        x = self.pool(x)      # 节点3: call_module("pool")
        return x              # 节点4: output

model = TinyNet()
# ☆ symbolic_trace —— 把 Module 变成 Graph
traced = torch.fx.symbolic_trace(model)

# GraphModule 有三个重要的东西你可以检查：
#   1. traced.graph  —— 一张 DAG
#   2. traced.code   —— 从 DAG 重新生成的 Python 代码（可读！）
#   3. traced.graph.nodes —— 节点列表

print(traced.code)
# 输出：
# def forward(self, x):
#     conv = self.conv(x)
#     relu = self.relu(conv)
#     pool = self.pool(relu)
#     return pool

# 现在你可以在这张图上做任意变换——比如在 relu 后面插入一个 print：
for node in traced.graph.nodes:
    if node.op == 'call_module' and node.target == 'relu':
        with traced.graph.inserting_after(node):
            # 创建新节点：调用 print 函数
            new_node = traced.graph.call_function(print, args=(node,))
            # 把原来用 relu 输出的节点，改成用 print 的输出
            # （这样 print 的返回值送到下一层，数据流不被中断）
            node.replace_all_uses_with(new_node)
            # 但 print 本身仍然以 relu 为输入
            new_node.args = (node,)

traced.recompile()
# 现在 traced.code 里能看到 print(relu) 被插入了
```

**量化就是在这个框架下插入 FakeQuantize 节点**。理解了这个，你就理解了 FX 量化的全部。

### 3.2 FX 量化管线：宏观视角

```
FP32 Module (nn.Module)
    │
    ├─ symbolic_trace(model)                       ← 变成一张 DAG
    │    → GraphModule
    │
    ├─ prepare_qat_fx(gm, qconfig_mapping, example_inputs)
    │    核心步骤（按顺序）:
    │      1. 自动识别融合 pattern（Conv+BN+ReLU 等）
    │      2. 融合并交换为 QAT 模块
    │      3. 遍历图节点，根据 qconfig_mapping 标注哪些节点需要被量化
    │      4. 修改图：在标注的节点前后插入 FakeQuantize 节点
    │      5. 把 observer 注册到 GraphModule 的子模块中
    │    → 一张插入了 FakeQuantize 节点的 DAG
    │
    ├─ train(gm)                                   ← QAT 训练
    │
    └─ convert_fx(gm)
         把 FakeQuantize 替换为真正的 Q/DQ 节点
         → INT8 推理图
```

### 3.3 完整代码走读

```python
# ===== FX Graph Mode QAT: MobileNetV3 + ImageNet =====
# 这次我们用 MobileNetV3，因为它是为移动端设计的，
# 包含 depthwise conv、SE block 等更有趣的结构，
# 能帮你更好地理解 FX 的优势

import torch
import torch.nn as nn
import torchvision
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import (
    prepare_qat_fx,
    convert_fx,
)
from torch.ao.quantization import (
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    QConfig,
)

# ═══════════════════════════════════════════════════════════
# Step 1: 加载模型 + 准备 example_inputs
# ═══════════════════════════════════════════════════════════
model = torchvision.models.mobilenet_v3_small(
    weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
).train()

# example_inputs — FX 需要它来做 symbolic_trace
# symbolic_trace 会实际跑一次 forward，记录执行路径
# 所以 example_inputs 的形状必须和真实输入一致（batch size 可以不同）
example_inputs = (torch.randn(1, 3, 224, 224),)

# ═══════════════════════════════════════════════════════════
# Step 2: 配置 QConfigMapping
# ═══════════════════════════════════════════════════════════
# FX Mode 最大的进步之一：用 QConfigMapping 替代了 QConfig
#
# QConfig 只能对整个 model 设置一个统一的量化策略。
# QConfigMapping 允许你按四个维度匹配不同的策略：
#   1. 精确模块名（"features.0.0"）
#   2. 正则表达式（"layer3.*conv"）
#   3. 模块类型（torch.nn.Conv2d）
#   4. 全局默认
# 优先级从上到下递减（精确匹配优先）

# 最简单的用法：用默认配置
qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
# qnnpack: ARM 移动端推理后端。activation 用 HistogramObserver。
# fbgemm:  x86 服务器后端。activation 也用 HistogramObserver。
# 两者核心区别在于对特定 op 的支持范围。

# === 进阶用法：完全自定义 ===
qconfig_mapping = QConfigMapping()

# 全局默认 QConfig
default_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    ),
)
qconfig_mapping.set_global(default_qconfig)

# 第一层和最后一层不量化 — 这是一个经过大量实验验证的经验：
# 第一层处理原始 RGB 像素，输入的范围固定 [0, 255]，量化后颜色失真明显
# 最后一层分类头的输出维度很小（1000 类），量化损失大但参数少，不值得
qconfig_mapping.set_module_name("features.0.0", None)  # 第一层
qconfig_mapping.set_module_name("classifier", None)     # 最后一层

# ═══════════════════════════════════════════════════════════
# Step 3: prepare_qat_fx —— 在图层面插入 FakeQuantize
# ═══════════════════════════════════════════════════════════
model_prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# ═══════════════════════════════════════════════════════════
# Step 4: ★ 打印图！这是 FX Mode 最强大的能力 ★
# ═══════════════════════════════════════════════════════════
print("=== QAT 准备后的 FX 图 ===")
model_prepared.graph.print_tabular()
# 你会看到类似这样的输出（简化版）：
#
# opcode         name               target                     args
# --------       ----------         --------------------       --------
# placeholder    x                  x                          ()
# call_module    a_p_p_0            activation_post_process_0  (x,)          ← FakeQuantize!
# call_module    features_0_0       features.0.0               (a_p_p_0,)    ← Conv
# call_module    a_p_p_1            activation_post_process_1  (features_0_0,) ← FakeQuantize!
# call_module    features_0_1       features.0.1               (a_p_p_1,)    ← 下一个 op
# ...             ...                ...                        ...
# output         output             output                     (...)
#
# 看到了吗？activation_post_process_* 就是 FakeQuantize 节点，
# 它们是独立存在的图节点，不是藏在模块内部的"幽灵"！

# 你也可以用代码遍历量化节点
for node in model_prepared.graph.nodes:
    if 'activation_post_process' in str(node.target):
        print(f"  量化节点 {node.name}: 输入={node.args[0].name}")

# ═══════════════════════════════════════════════════════════
# Step 5: 准备训练
# ═══════════════════════════════════════════════════════════
# 冻结 BN（和 Eager Mode 一样的逻辑）
for module in model_prepared.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)

optimizer = torch.optim.SGD(model_prepared.parameters(), lr=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ═══════════════════════════════════════════════════════════
# Step 6: QAT 训练
# ═══════════════════════════════════════════════════════════
for epoch in range(10):
    model_prepared.train()

    # observer 调度
    if epoch < 3:
        model_prepared.apply(torch.ao.quantization.enable_observer)
        print(f"  Epoch {epoch}: Observer ON  — 正在收集 scale")
    else:
        model_prepared.apply(torch.ao.quantization.disable_observer)
        print(f"  Epoch {epoch}: Observer OFF — scale 固定，纯 QAT 训练")

    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()
        output = model_prepared(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # 验证
    model_prepared.eval()
    model_prepared.apply(torch.ao.quantization.disable_observer)
    with torch.no_grad():
        correct = total = 0
        for images, targets in val_loader:
            images, targets = images.cuda(), targets.cuda()
            output = model_prepared(images)
            correct += (output.argmax(1) == targets).sum().item()
            total += targets.size(0)
        acc = correct / total * 100
    print(f"Epoch {epoch}: QAT Val Accuracy = {acc:.2f}%")

# ═══════════════════════════════════════════════════════════
# Step 7: convert_fx — 从"假量化"到 INT8
# ═══════════════════════════════════════════════════════════
model_prepared.eval()
model_prepared.cpu()
model_int8 = convert_fx(model_prepared)

print("\n=== 转换后的 INT8 图 ===")
model_int8.graph.print_tabular()
# 你会看到 FakeQuantize 节点变成了 Quantize / Dequantize 节点，
# 部分 Conv 节点也变成了 QuantizedConv2d 这样的 INT8 原生 op

# ═══════════════════════════════════════════════════════════
# Step 8: 导出
# ═══════════════════════════════════════════════════════════
model_int8.eval()
dummy_input = (torch.randn(1, 3, 224, 224),)

# ONNX 导出（带 QDQ 节点 — opset >= 13 才支持）
torch.onnx.export(
    model_int8, dummy_input, "mobilenetv3_int8.onnx",
    opset_version=17,  # QDQ 节点从 opset 13 开始支持
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

---

## 4. PyTorch 2 Export (PT2E) QAT：未来的标准

### 4.1 先搞清楚：为什么 PT2E 要存在？

FX Graph Mode 已经很好了，为什么 PyTorch 团队还要做一个新的 PT2E？

答案在于一个根本性限制：**`symbolic_trace` 是一个"假 trace"**。它不是真正的程序分析。它只是传一个 dummy input 进去，跑一遍 forward，**记录"这条执行路径上发生了什么"**。如果模型里有这样的代码：

```python
class BadForFX(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:       # ← FX 无法处理！
            return x * 2
        return x * 3
```

`symbolic_trace` 只能记录一个分支（比如 x.sum() > 0 的那条），另一个分支就丢了。生成的图不完整。

而 `torch.export` 是一个真正的 **Python AST tracer**。它不跑 forward，而是分析源码的抽象语法树，展开所有控制流。这意味着 export 能处理 `symbolic_trace` 处理不了的模型。

此外，PT2E 引入了一个非常重要的抽象：**Quantizer**。不同后端（Intel、ARM、高通、NVIDIA）对量化的支持不一样：
- 有的后端支持 per-channel 量化，有的只支持 per-tensor
- 有的后端支持非对称量化，有的只支持对称
- 有的后端支持 INT8 Add，有的只支持 FP32 Add → 需要 QDQ 包围

在 FX Mode 里，这些差异隐藏在 qconfig 的具体 Observer 选择中，不够显式。PT2E 把"后端的量化约束"打包成一个独立的 `Quantizer` 对象，**换后端 = 换 Quantizer**。

### 4.2 PT2E vs FX：核心差异一张表

| 对比维度 | FX Graph Mode | PT2E |
|----------|---------------|------|
| 图捕获方式 | `symbolic_trace`（跑一遍 forward 记录路径） | `torch.export`（AST 级别程序分析） |
| 控制流 | 不支持动态 `if/while` | 支持（会展开到图里） |
| 形状约束 | trace 时固定形状 | 支持符号化维度 (`torch.export.Dim`) |
| 后端适配 | 隐藏在 QConfig 的 Observer 选择中 | 显式的 `Quantizer` 对象，可插拔 |
| 与 torch.compile 兼容 | 不兼容 | 天然兼容 |
| 成熟度 | 稳定，生产可用 | 快速迭代中，PyTorch 2.5+ 推荐 |

### 4.3 完整 PT2E QAT 流程

```python
# ===== PT2E QAT: ResNet18 + ImageNet =====
import torch
import torchvision
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# ═══════════════════════════════════════════════════════════
# Step 1: 加载模型
# ═══════════════════════════════════════════════════════════
model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
).train()
example_inputs = (torch.randn(2, 3, 224, 224),)

# ═══════════════════════════════════════════════════════════
# Step 2: 捕获计算图（注意这里和 FX 不同了！）
# ═══════════════════════════════════════════════════════════
# PT2E 用 capture_pre_autograd_graph（或 torch.export），
# 而不是 symbolic_trace
from torch._export import capture_pre_autograd_graph
exported_model = capture_pre_autograd_graph(model, example_inputs)
# 返回的仍然是 GraphModule，但捕获方式不同
# "pre_autograd" 表示在 autograd 节点被加入之前捕获图
# 这样得到的是"干净的"前向计算图，不包含 .backward() 相关节点

print("=== 捕获的计算图 ===")
exported_model.graph.print_tabular()

# ═══════════════════════════════════════════════════════════
# Step 3: 配置 Quantizer — PT2E 独有的抽象
# ═══════════════════════════════════════════════════════════
# Quantizer 回答了这些问题：
#   - 哪些 op 应该被量化？（Conv? Linear? Add? MatMul?）
#   - 用什么方案量化？（对称/非对称、per-tensor/per-channel）
#   - 量化比特数是多少？（8-bit、4-bit）
#   - 是否做 QAT？（is_qat=True）
quantizer = XNNPACKQuantizer()

quantizer.set_global(
    get_symmetric_quantization_config(
        is_qat=True,           # ★ 告诉 Quantizer：我们要做 QAT
        is_per_channel=True,   # 权重 per-channel
    )
)
# quantizer.set_module_name("fc", None)  # 不量化最后一层

# ═══════════════════════════════════════════════════════════
# Step 4: prepare_qat_pt2e
# ═══════════════════════════════════════════════════════════
# 内部做了四件事（按顺序）：
#   1. quantizer.annotate(exported_model)
#      → 在图节点上标注 qspec（量化规格）
#   2. quantizer.validate(exported_model)
#      → 检查配置和后端能力是否匹配
#   3. _fuse_conv_bn_qat(exported_model)
#      → 在图层面融合 Conv+BN（不是模块层面！）
#   4. prepare(exported_model, ..., is_qat=True)
#      → 在标注了 qspec 的地方插入 FakeQuantize
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

# ═══════════════════════════════════════════════════════════
# Step 5: QAT 训练（和 FX / Eager 一样的循环）
# ═══════════════════════════════════════════════════════════
for epoch in range(10):
    prepared_model.train()
    if epoch < 3:
        prepared_model.apply(torch.ao.quantization.enable_observer)
    else:
        prepared_model.apply(torch.ao.quantization.disable_observer)
    train_one_epoch(prepared_model, ...)

    prepared_model.eval()
    val_acc = evaluate(prepared_model, ...)
    print(f"Epoch {epoch}: QAT Accuracy = {val_acc:.2f}%")

# ═══════════════════════════════════════════════════════════
# Step 6: convert_pt2e
# ═══════════════════════════════════════════════════════════
prepared_model.eval()
quantized_model = convert_pt2e(prepared_model)
# 修复 eval 模式的细节（如 dropout 等）
torch.ao.quantization.move_exported_model_to_eval(quantized_model)

# ═══════════════════════════════════════════════════════════
# Step 7: 保存
# ═══════════════════════════════════════════════════════════
torch.export.save(quantized_model, "resnet18_quant.pt2")
```

---

## 5. 源码深潜：prepare_qat_fx 内部发生了什么

> 这一节是本文档最硬的部分。读完后，你就不再是 PyTorch 量化的"用户"，而是可以修改它的"开发者"。

### 5.1 宏观六步

`prepare_qat_fx` 不是魔法——它是六个有明确职责的步骤的组合。每一层都可以单独理解和 debug：

```
prepare_qat_fx(model, qconfig_mapping, example_inputs)
    │
    ├── Step 1: symbolic_trace(model) → GraphModule (gm)
    │     把 Module 转成 DAG。所有后续操作都在 DAG 上进行。
    │
    ├── Step 2: _fuse_fx(gm, qconfig_mapping)
    │     自动扫描图中的融合 pattern（Conv+BN+ReLU 等），做模块级别融合。
    │     这和 Eager Mode 的手动 fuse_modules 是同一件事，但全自动了。
    │
    ├── Step 3: _qat_swap_modules(gm, qconfig_mapping)
    │     把融合后的 PTQ 模块换成 QAT 版本。
    │     例如 nni.ConvBnReLU2d → nniqat.ConvBnReLU2d
    │     QAT 版本内部已经预埋了 weight_fake_quant 和 activation_post_process。
    │
    ├── Step 4: 量化标注 (annotation)
    │     遍历 DAG 节点，根据 QConfigMapping 的规则，
    │     决定每个节点是否需要被量化、用什么 observer。
    │
    ├── Step 5: 插入 observer / fake_quantize 节点
    │     ★ 图变换的核心操作。在需要量化的 op 的输出后，
    │     创建一个新的 call_module 节点（指向一个 observer 实例）。
    │     然后重新连接数据边：op → observer → 后续消费者。
    │
    └── Step 6: 返回修改后的 GraphModule
```

### 5.2 Step 3 详解：为什么模块要 "swap"

你可能会问：融合后的模块（比如 `nni.ConvBnReLU2d`）不是已经包含 Conv + BN + ReLU 了吗？为什么还要换成 QAT 版本？

答案是：**两个模块的计算结果一样，但内部结构不同**。

- `nni.ConvBnReLU2d`（PTQ 版本）：内部是直接的 Conv + BN(eval) + ReLU，**没有 FakeQuantize**。给 PTQ 用。
- `nniqat.ConvBnReLU2d`（QAT 版本）：**内部已经有 weight_fake_quant（对权重做假量化）和 activation_post_process（对输出做假量化）**。给 QAT 用。

两者计算的结果在数学上等价（如果在 QAT 模块中 observer 是关闭的），但 QAT 版本提供了"量化噪声注入"的机制。

```python
# ===== 模块 swap 的映射表（源码中的定义） =====
# 位置: torch/ao/quantization/fx/fusion_patterns.py
#
# PTQ 模块              →  QAT 模块
# ─────────────────────────────────────────────
# nni.ConvBn1d           → nniqat.ConvBn1d
# nni.ConvBn2d           → nniqat.ConvBn2d
# nni.ConvBn3d           → nniqat.ConvBn3d
# nni.ConvBnReLU1d       → nniqat.ConvBnReLU1d
# nni.ConvBnReLU2d       → nniqat.ConvBnReLU2d
# nni.ConvReLU1d         → nniqat.ConvReLU1d
# nni.ConvReLU2d         → nniqat.ConvReLU2d
# nni.ConvReLU3d         → nniqat.ConvReLU3d
# nni.LinearReLU         → nniqat.LinearReLU
# nni.LinearBn1d         → nniqat.LinearBn1d

def _qat_swap_modules_simplified(root: torch.nn.Module, qconfig_mapping):
    """
    遍历所有子模块，遇到 PTQ 融合模块就换成 QAT 版本。
    """
    for name, child in root.named_children():
        if type(child) in _MODULE_TO_QAT_MODULE:
            # 找到对应的 QAT 类
            qat_cls = _MODULE_TO_QAT_MODULE[type(child)]

            # 创建 QAT 版本（from_float 是 PyTorch 中常见的模式：
            # 从一个 float 模块创建一个其量化版本）
            qat_child = qat_cls.from_float(child, qconfig=qconfig)

            # 替换！
            setattr(root, name, qat_child)
```

### 5.3 Step 4-5 详解：量化标注 + 图变换

这是整个 FX 量化最核心的操作——**修改 DAG**。

理解下面的代码，你就理解了 FX 量化的 90%：

```python
# ===== 量化标注和 FakeQuantize 插入（简化版源码解读） =====

def annotate_and_insert_observers(model: torch.fx.GraphModule,
                                   qconfig_mapping):
    """
    这函数做两件事：
    1. 标注（annotation）：决定哪些节点要量化、用什么 observer
    2. 插入（insertion）：在图上创建新节点，重新连接数据边

    关键术语：
    - 图节点 (Node): 计算图上的一个操作。有四种 op 类型：
        placeholder   — 输入
        call_module   — 调用一个 nn.Module
        call_function — 调用一个函数（如 torch.add）
        output        — 输出
    - 数据边 (data edge): 节点之间的数据依赖关系，通过 args/kwargs 表示
    - replace_all_uses_with: FX 的重定向函数，把"原来用 node 输出的地方"
                             全部改成用另一个节点的输出
    """

    for node in list(model.graph.nodes):
        # 跳过不需要量化的节点
        qconfig = qconfig_mapping.get_qconfig(node)
        if qconfig is None:
            continue

        # === 创建 activation observer（FakeQuantize）模块 ===
        act_observer = qconfig.activation()
        obs_name = f"activation_post_process_{len(list(model.named_modules()))}"
        model.add_submodule(obs_name, act_observer)

        # === ★ 图变换：在 op 后面插入 observer 节点 ===
        # 1. 用 inserting_after 告诉 FX "我要在 node 后面操作"
        with model.graph.inserting_after(node):
            # 2. 创建一个新的图节点，代表 "调用 obs_name 这个模块"
            new_obs_node = model.graph.create_node(
                'call_module',
                obs_name,
                args=(node,),     # observer 的输入是 node 的输出
                kwargs={},
            )

        # 3. ★ 重新连接数据流 —— 这是最巧妙的一步 ★
        #
        # 在这之前，数据流是这样的：
        #   ... → node → consumer1, consumer2, ...
        #
        # node.replace_all_uses_with(new_obs_node) 会把数据流变成：
        #   ... → node → new_obs_node → consumer1, consumer2, ...
        #
        # 但这里有一个微妙的点：new_obs_node 的 args 本来设置的是 (node,)，
        # replace_all_uses_with 可能会把 new_obs_node.args 也改掉！
        # （因为 new_obs_node 自己也"使用"了 node 的输出）
        #
        # 所以 replace_all_uses_with 之后，我们要重新设置 new_obs_node 的 args：
        node.replace_all_uses_with(new_obs_node)
        new_obs_node.args = (node,)  # 恢复 observer 的输入指向

    # 图修改完成后必须做 lint 检查（验证图结构合法性）和重新编译
    model.graph.lint()
    model.recompile()
```

**这段代码中有一个非常容易踩的坑**：

`node.replace_all_uses_with(new_obs_node)` 之后，原来流向 `consumer1, consumer2` 的数据现在先经过 `new_obs_node`。但是 `new_obs_node` 自己也在 `node` 的 "uses" 列表中（因为 `new_obs_node.args = (node,)`）。所以 `replace_all_uses_with` 可能把 `new_obs_node.args` 也替换了——导致 observer 的输入指向自己（循环依赖）。

修复方法是：`replace_all_uses_with` 后手动恢复 `new_obs_node.args = (node,)`。PyTorch 的真实源码中也是这样处理的。

---

## 6. 源码深潜：convert_fx 内部发生了什么

QAT 训练结束后，FakeQuantize 需要被替换为**真正的量化/反量化节点**。这就是 `convert_fx` 的工作。

```python
# ===== convert_fx 的核心逻辑（简化版） =====
# 原始位置: torch/ao/quantization/fx/convert.py

def convert_fx_simplified(model: torch.fx.GraphModule):
    """
    QAT 训练后 → 真正的 INT8 推理模型。

    核心操作（按顺序）：
    1. 找到所有 FakeQuantize 节点
    2. 读取其训练得到的 scale / zero_point
    3. 替换为真正的 Q/DQ 节点
    4. 融合 DQ + op + Q 为 INT8 原生 op（如果 backend 支持）
    """

    for node in list(model.graph.nodes):
        # 只处理 FakeQuantize 节点
        if 'activation_post_process_' not in str(node.target):
            continue

        observer = model.get_submodule(node.target)

        # 读取 QAT 训练的最终 scale 和 zero_point
        scale = observer.scale        # 这是 QAT 训练后固定的值
        zero_point = observer.zero_point

        # 关键概念：
        #   FakeQuantize → 训练时用（float 离散化 → float，梯度可传）
        #   Quantize + DeQuantize → 推理时用（真正在图上表示 Q/DQ 语义）
        #
        # convert 做的事就是把前者变成后者。
        # 对于支持 INT8 原生 op 的 backend（如 fbgemm 的 QuantizedConv2d）：
        #   DQ + Conv + Q 直接融合为一个 INT8 Conv，连 Q/DQ 节点都省了。

    model.recompile()
    return model
```

**关于量化 op 替换的补充说明：**

PyTorch 维护了一份"哪些浮点 op 有对应的 INT8 版本"的映射表（在 `torch/ao/quantization/fx/quantization_patterns.py` 中）。并不是所有 op 都有 INT8 版本：

```
有 INT8 版本的 op（可以融合为量化 op）:
  Conv2d           → QuantizedConv2d
  Conv2d + ReLU    → QuantizedConvReLU2d    （一个融合的 INT8 op 完成 Conv+ReLU！）
  Linear           → QuantizedLinear
  Linear + ReLU    → QuantizedLinearReLU

没有 INT8 版本的 op（需要 Q→DQ→FP op→Q→DQ 包围）:
  Add、Concat、Reshape、Sigmoid、Tanh、...
  这些 op 的量化方式是：
    输入先 DQ（反量化→FP32）
    → 做 FP32 的 Add/Concat/...
    → 输出再 Q（量化→INT8）
  显然这有额外的精度损失（反复量化/反量化），
  这也是为什么 PTQ/QAT 中"图融合"如此重要——
  好的融合能减少不必要的 Q/DQ 节点。
```

---

## 7. Conv-BN-ReLU 融合：数学原理 + 图操作

### 7.1 为什么融合是 QAT 的必须手段

想象 Conv 的输出经过 FakeQuantize，再送入 BN。BN 期望的输入分布是训练时的"正常 FP32 Conv 输出"。但现在 Conv 的输出被 FakeQuantize 离散化了——BN 看到的分布变了。这意味着：

1. **如果不融合**：BN 的 running stats（mean / var）是基于 FP32 激活值算的，但现在输入是量化后的激活值，统计量不匹配
2. **如果融合**：BN 的权重和 bias 被"折入"了 Conv 的权重和 bias，推理时只有一个 Conv 操作，没有 BN——也就不存在这个不匹配问题

### 7.2 融合的数学推导

推导本身其实很简单——就是初等代数替换。但理解每个符号的意义很重要：

```python
# Conv + BN 的等价变换（推理模式）
#
# 原始数据流:
#   x ──→ [Conv] ──→ y_conv ──→ [BatchNorm] ──→ y_bn
#
# 其中:
#   y_conv = W * x + b_conv          (1)  — Conv 的输出
#   y_bn   = γ * (y_conv - μ) / √(σ² + ε) + β   (2)  — BN 推理模式的输出
#            ↑ 推理模式下，μ 和 σ 是 running_mean 和 running_var，
#              不是 batch stats。它们是训练期间累积的固定常数。
#
# 把 (1) 代入 (2):
#   y_bn = γ * (W*x + b_conv - μ) / √(σ² + ε) + β
#        = [γ / √(σ² + ε)] * W * x + [γ * (b_conv - μ) / √(σ² + ε) + β]
#        =     W_fused      * x +          b_fused
#
# 结论：Conv + BN（推理模式）可以精确等价为一个新的 Conv，
# 其权重和偏置如上式计算。没有任何近似！
# 唯一的误差来自 FP32 的有限精度。

def fuse_conv_bn_weights(conv_weight, conv_bias,
                          bn_weight, bn_bias,
                          bn_running_mean, bn_running_var, bn_eps):
    """计算融合后的 Conv 权重和偏置"""

    # k = γ / √(σ² + ε)  — 每个通道一个值
    k = bn_weight / torch.sqrt(bn_running_var + bn_eps)

    # W_fused = W * k   — k 需要广播到 weight 的每个空间位置
    # view(-1, 1, 1, 1) 是为了和 weight 的 [out_c, in_c, h, w] 对齐
    fused_weight = conv_weight * k.view(-1, 1, 1, 1)

    if conv_bias is not None:
        fused_bias = (conv_bias - bn_running_mean) * k + bn_bias
    else:
        fused_bias = bn_bias - bn_running_mean * k

    return fused_weight, fused_bias
```

### 7.3 动手验证

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(16, 32, 3, padding=1)
bn = nn.BatchNorm2d(32)
x = torch.randn(4, 16, 32, 32)

# 原始前向（BN 推理模式）
bn.eval()
y_raw = bn(conv(x))

# 融合后
fused_w, fused_b = fuse_conv_bn_weights(
    conv.weight, conv.bias,
    bn.weight, bn.bias,
    bn.running_mean, bn.running_var, bn.eps,
)
y_fused = nn.functional.conv2d(x, fused_w, fused_b, padding=1)

diff = (y_raw - y_fused).abs().max().item()
print(f"最大差异: {diff:.2e}")
assert diff < 1e-4, f"融合不精确！"
print("✅ Conv+BN 融合在 FP32 精度下是精确的")
```

### 7.4 PT2E 中的图层面融合（另一套思路）

Eager Mode / FX Mode 的融合是在**模块层面**做——把 `nn.Conv2d` + `nn.BatchNorm2d` + `nn.ReLU` 三个模块替换成一个 `nni.ConvBnReLU2d`。

PT2E 的融合思路不同——它是在**图层面**做。为什么？因为 `torch.export` 导出的图里，Conv 和 BN 不是模块，而是独立的图节点（`aten.conv2d` 和 `aten.batch_norm`）。你没法"换一个模块"，只能"替换一组节点"。

```
PT2E 导出的原始图:
  %conv_out = aten.conv2d(%x, %w, %b, ...)
  %bn_tuple = aten.batch_norm(%conv_out, %bn_w, %bn_b, %mean, %var, ...)
  %bn_out   = operator.getitem(%bn_tuple, 0)   ← BN 返回 tuple，取第一个
  %relu_out = aten.relu(%bn_out)

PT2E 融合后:
  %fused_w  = ...  (计算好的融合权重)
  %fused_b  = ...  (计算好的融合偏置)
  %conv_out = aten.conv2d(%x, %fused_w, %fused_b, ...)
  %relu_out = aten.relu(%conv_out)
```

---

## 8. 自定义量化策略：QConfigMapping 和 BackendConfig

### 8.1 QConfigMapping：多层匹配的优先级系统

`QConfigMapping` 相当于一个"层到量化策略的映射表"，支持 4 种匹配方式，按优先级排序：

```
优先级 高 → 低（先匹配先生效，后续不覆盖）

1. set_module_name("conv1", qconfig)           ← 精确匹配到 conv1
2. set_module_name_regex("layer3.*conv", ...)  ← 正则匹配
3. set_object_type(torch.nn.Conv2d, ...)       ← 匹配所有 Conv2d
4. set_global(qconfig)                          ← 兜底默认
```

举个例子说明优先级规则：

```python
qmap = QConfigMapping()
qmap.set_global(qconfig_A)                    # 优先级最低——所有层默认
qmap.set_object_type(torch.nn.Conv2d, qconfig_B) # 覆盖所有 Conv2d
qmap.set_module_name("layer3.0.conv1", qconfig_C) # 最高优先级——只覆盖这一层

# layer3.0.conv1 → qconfig_C (精确匹配)
# layer1.0.conv1 → qconfig_B (类型匹配)
# fc (Linear)     → qconfig_A (全局匹配)
```

### 8.2 BackendConfig："这个后端能做什么"

不同后端能做的量化不一样。`BackendConfig` 就是"后端的量化能力说明书"：

```python
from torch.ao.quantization.backend_config import (
    get_fbgemm_backend_config,
    get_qnnpack_backend_config,
)

cfg = get_fbgemm_backend_config()
# BackendConfig 定义了三种信息:
#   1. 哪些 op pattern 可以被量化（如 Conv2d, Linear, Add）
#   2. 这些 op 融合后的模块是什么（如 Conv+ReLU → NNI.ConvReLU2d）
#   3. 这些 op 的 QAT 版本模块是什么

# 举例：fbgemm 支持直接量化 Add —
#   QuantizedAdd 可以直接做 INT8 加法，不需要 DQ → FP32 Add → Q
# 而 qnnpack 不支持——Add 只能用 DQ → FP32 Add → Q 包围
# 这个差异就记录在 BackendConfig 里
```

---

## 9. QAT 训练实战技巧

### 9.1 BN 冻结：不做这步你的 QAT 大概率崩

这是 QAT 训练中最重要的一个操作，没有之一。

**问题根源**：QAT 训练期间，FakeQuantize 改变了激活值的分布。如果 BN 仍然以训练模式运行（用当前 batch 的 stats 更新 running stats），那每次 forward 的 running stats 都在变化，导致：
- BN 的输出在 epoch 之间不稳定
- FakeQuantize 的 scale 是基于不同分布的 BN 输出算的
- 优化器面对的是一个"移动的靶子"

**解决方案**：冻结 BN——无论模型模式是 train 还是 eval，BN 一律用 eval 模式：

```python
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()                       # 使用 running stats
            module.weight.requires_grad_(False)  # 不更新 γ
            module.bias.requires_grad_(False)    # 不更新 β
```

### 9.2 Observer 调度：什么时候冻结 scale

```python
"""
标准策略：前 2-3 个 epoch 收集 scale，之后固定。

为什么是 2-3 个 epoch 而不是更多？
  - 太短（1 epoch）：基于少量数据的 min/max 不可靠
  - 太长（5+ epoch）：浪费训练时间，真正的 QAT 还没开始
  - 太小 batch 的情况：需要更多 epoch 让 observer 看到足够的数据分布

特殊情况：
  如果 calibration 数据和训练数据分布不同，
  应该用专门的校准集（calibration set）单独跑 observer，
  然后加载固定 scale 再开始 QAT 训练。
"""

class QATObserverScheduler:
    def __init__(self, model, disable_observer_epoch=3):
        self.model = model
        self.disable_observer_epoch = disable_observer_epoch

    def step(self, epoch):
        if epoch < self.disable_observer_epoch:
            self.model.apply(torch.ao.quantization.enable_observer)
        else:
            self.model.apply(torch.ao.quantization.disable_observer)
```

### 9.3 学习率设置

```
经验规则（来自大量实验和论文）:
  - QAT LR = FP32 最终 LR 的 1/10 ~ 1/100
  - 典型值：1e-4 （对 ResNet 类模型）
  - 时间表：cosine annealing 从 1e-4 衰减到 1e-6

直觉：
  QAT 是"微调"不是"从头训练"。FP32 权重已经在一个很好的位置（局部最优）。
  LR 太大会让模型跳出这个最优区。
  LR 太小会导致对量化噪声的补偿不够（梯度太小，更新不进去）。

推荐方案：
  optimizer = SGD(lr=1e-4, momentum=0.9, weight_decay=1e-4)
  scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
```

### 9.4 常见错误排查手册

| 症状 | 最可能的原因 | 怎么排查和修 |
|------|-------------|------------|
| QAT 精度反而比 PTQ 差 | Observer 用的 epoch 太短，scale 没收敛 | 增加 observer epoch 数到 5；用 calibration set 单独校准后加载 |
| 导出的 ONNX 模型大小没变 | 忘了调 `convert()`，导出的还是 float 模型 | 在 export 前确认 `model_int8 = convert_fx(model_prepared)` 已执行 |
| ONNX 没有 QDQ 节点 | `opset_version` 低于 13 | 导出时指定 `opset_version=17` |
| 训练 loss 剧烈震荡 | BN 未冻结，每个 batch 的 BN stats 不同 | 对所有 BN 模块调用 `.eval()` 并关闭梯度 |
| FX trace 报错 "Cannot trace through dynamic control flow" | 模型中有 `if/while` 依赖 tensor 值 | 重构模型去掉动态控制流；或切换到 PT2E |
| 量化后某些通道全是 0 | 该通道的输出范围过大，clip 掉了所有值 | 对该层使用非对称量化或 per-channel 量化 |
| reduce_range=True 导致图像偏黑 | reduce_range 把范围从 [-128,127] 缩小到 [-64,63] | 改成 reduce_range=False |

---

## 10. 动手实验

### 实验 1：可视化 FakeQuantize 的效果（30 分钟）

目标：亲眼看到 FakeQuantize 如何把连续的 FP32 值变成离散的。

```python
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").train()
model.qconfig = get_default_qat_qconfig("fbgemm")

dummy_input = torch.randn(1, 3, 224, 224)

# 用 hook 收集 FakeQuantize 前后的激活值
activations_before = []
activations_after = []

def hook_before(module, input, output):
    activations_before.append(output.detach().cpu())

def hook_after(module, input, output):
    activations_after.append(output.detach().cpu())

hook1 = model.conv1.register_forward_hook(hook_before)
model_qat = prepare_qat(model)
hook2 = model_qat.conv1.register_forward_hook(hook_after)
model_qat(dummy_input)
hook1.remove(); hook2.remove()

# 画图：对比量化前后的分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(activations_before[0].flatten().numpy(), bins=100)
axes[0].set_title("Before FakeQuantize (连续 FP32)")
axes[1].hist(activations_after[0].flatten().numpy(), bins=100)
axes[1].set_title("After FakeQuantize (离散化后的值)")
plt.savefig("fakequant_distribution.png")
print("✅ 查看 fakequant_distribution.png —— 注意第二个图只有 256 根柱子（INT8 的 256 个值）")
```

### 实验 2：三种模式对比实验（45 分钟）

用 Eager / FX / PT2E 三种模式对同一个 MobileNetV2 做 QAT，记录代码行数和最终精度：

```python
# 对比维度：
#   1. 代码量（不含 import 的行数）
#   2. Debug 便利性（能不能打印图）
#   3. 最终 Top-1 accuracy
#   4. 从开始写代码到跑通的时间
#
# 期望结果（2024 年的标准）:
#   FX Mode: 代码最少、最易 debug、精度最好
#   Eager Mode: 代码最冗余、难 debug、精度可能略低（因为手动融合不完整）
#   PT2E: 代码适中、最灵活（多后端支持）、精度和 FX 持平
```

### 实验 3：QAT vs FP32 训练曲线对比（30 分钟）

```python
# 两个相同的初始模型:
#   A: 继续 FP32 微调（baseline）
#   B: QAT 训练
# 
# 每 epoch 记录 train loss + val accuracy
# 画在同一张图上
#
# 分析点:
#   1. QAT 的初始 loss 比 FP32 高（量化噪声导致）
#   2. 随着训练，QAT 的 loss 逐渐接近 FP32
#   3. 但 QAT 可能永远追不平 FP32——差距取决于量化比特数
```

---

## 检验标准

完成这个阶段后，你应该能达到以下水平：

- [ ] **能画出** FakeQuantize 状态机图，讲清楚 QAT 三阶段各自的两个 flag 状态，以及为什么这样设计
- [ ] **能用 Eager Mode** 完成 ResNet18 QAT，理解 `prepare_qat` 对 forward 方法的修改
- [ ] **能用 FX Graph Mode** 完成 MobileNetV3 QAT，并用 `print_tabular()` 定位每一个 FakeQuantize 在图中的位置
- [ ] **能用 PT2E Mode** 完成 QAT，说出 PT2E 和 FX 在"图是怎么来的"这一步的根本区别
- [ ] **能手写** `prepare_qat_fx` 的核心逻辑（不是背代码，是理解每一步在图上做了什么操作）
- [ ] **能推导** Conv+BN 融合的数学等价性，并用代码验证
- [ ] **能配置** QConfigMapping 的四种优先级来实现"第一层和最后一层不量化、中间层用不同策略"
- [ ] **能排查** QAT 训练中的 BN 未冻结 / observer 时序 / LR 过大等常见错误

---

> 💡 **学习建议**：这个阶段最有价值的操作是 **"看图"** — 训练前把 FX graph 打印出来，观察 FakeQuantize 节点在哪、叫什么名字、和前后节点怎么连接的。观察 convert 前后的图变化。**看着图训练，你才能真正理解"量化训练"四个字在计算图层面意味着什么。**
>
> Next: [Stage 2: QAT 核心算法 — LSQ 与可微量化参数](./QAT_LEARNING_ROADMAP.md#阶段-2qat-核心算法--lsq-与可微量化参数)
