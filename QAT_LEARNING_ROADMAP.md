# 🎯 量化感知训练（QAT）深度学习路线图

> **目标读者画像**：已懂 PTQ 基本概念 + QAT 基本原理，了解 Sigmoid 查表实现，了解 ONNX 图优化 / 端侧部署，深度学习基础原理掌握 60-70%。
> **学习策略**：前期用 YOLO 类视觉模型实操，后期过渡到大模型（LLM）。**由简到难，层层递进**。

---

## 📖 目录

- [阶段 0：基础夯实 — 数值量化与硬件基石](#阶段-0基础夯实--数值量化与硬件基石)
- [阶段 1：PyTorch 原生 QAT — 三种模式从入门到精通](#阶段-1pytorch-原生-qat--三种模式从入门到精通)
- [阶段 2：QAT 核心算法 — LSQ 与可微量化参数](#阶段-2qat-核心算法--lsq-与可微量化参数)
- [阶段 3：PTQ 进阶算法 — AdaRound / FlexRound / GPTQ 的数学内核](#阶段-3ptq-进阶算法--adaround--flexround--gptq-的数学内核)
- [阶段 4：YOLO 量化实战 — 端到端项目](#阶段-4yolo-量化实战--端到端项目)
- [阶段 5：工业级框架深度使用 — PPQ / AIMET](#阶段-5工业级框架深度使用--ppq--aimet)
- [阶段 6：大模型 PTQ — GPTQ / AWQ / SmoothQuant / SpinQuant](#阶段-6大模型-ptq--gptq--awq--smoothquant--spinquant)
- [阶段 7：大模型 QAT — LLM-QAT / QLoRA / EfficientQAT](#阶段-7大模型-qat--llm-qat--qlora--efficientqat)
- [阶段 8：端侧部署全链路 — 从量化模型到芯片推理](#阶段-8端侧部署全链路--从量化模型到芯片推理)
- [阶段 9：前沿拓展 — FP8 / KV Cache 量化 / BitNet / MoE 量化](#阶段-9前沿拓展--fp8--kv-cache-量化--bitnet--moe-量化)
- [附录：推荐书单 & 课程 & 社区](#附录推荐书单--课程--社区)

---

## 阶段 0：基础夯实 — 数值量化与硬件基石

> ⏱ 预计时间：1-2 周 | 🎯 难度：⭐

### 0.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **IEEE 754 浮点表示**：FP32 / FP16 / BF16 的位布局（sign/exponent/mantissa）| 能手动画出 32-bit 布局图 |
| 2 | **量化公式推导**：`q = round((r - Z) / S)` 和 `r = S * (q + Z)`，S 和 Z 怎么算 | 能从 FP32 的 min/max 一步步推出 INT8 的 S/Z |
| 3 | **对称 vs 非对称量化**：各自的适用范围、优缺点 | 理解为什么 Activation 常用非对称，Weight 常用对称 |
| 4 | **量化粒度**：per-tensor / per-channel / per-group / group-size | 知道每种粒度的 scale 存储开销怎么算 |
| 5 | **校准算法（Calibration）**：MinMax、MSE、KL Divergence、Percentile | 能手写 MinMax calibrator |
| 6 | **舍入策略**：round-to-nearest、stochastic rounding | 理解 stochastic rounding 为什么在低比特（≤4-bit）中重要 |
| 7 | **INT8 硬件加速原理**：VNNI / DP4A 指令，Tensor Cores INT8 计算流水线 | 知道一个 INT8 Conv 在硬件上怎么算的 |
| 8 | **Clip 与 Clamp**：量化范围裁剪的含义和数学表达 | 理解 `clamp(round(v/S), Q_min, Q_max)` |

### 0.2 怎么做

```python
# ========== 动手：从零实现一个 MinMax INT8 量化器 ==========
import torch
import numpy as np

class MinMaxQuantizer:
    """手写对称 + 非对称 per-tensor INT8 量化器"""
    def __init__(self, symmetric=True, per_channel=False):
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.scale = None
        self.zero_point = None

    def calibrate(self, x: torch.Tensor):
        if self.per_channel:
            # per-channel: 沿 channel 维度计算 min/max
            dims = tuple(i for i in range(x.dim()) if i != 0)  # channel=0
            x_min = x.amin(dim=dims)
            x_max = x.amax(dim=dims)
        else:
            x_min, x_max = x.min(), x.max()

        if self.symmetric:
            # 对称量化: Z = 0, S = max(|min|, |max|) / 127
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            self.scale = abs_max / 127.0
            self.zero_point = torch.zeros_like(self.scale)
        else:
            # 非对称量化: S = (max - min) / 255, Z = -round(min / S)
            self.scale = (x_max - x_min) / 255.0
            self.zero_point = torch.round(-x_min / self.scale).clamp(0, 255)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.round(x / self.scale + self.zero_point)
        q = torch.clamp(q, -128, 127) if self.symmetric else torch.clamp(q, 0, 255)
        return q

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return self.scale * (q - self.zero_point)

# ========== 验证：量化误差分析 ==========
x = torch.randn(3, 64, 64) * 2.0 + 0.5  # 模拟激活值
q = MinMaxQuantizer(symmetric=False)
q.calibrate(x)
x_q = q.dequantize(q.quantize(x))

mse = ((x - x_q) ** 2).mean()
snr = 10 * torch.log10(x.var() / mse)
print(f"Quantization MSE: {mse:.6f}, SNR: {snr:.2f} dB")
```

### 0.3 从哪里学

| 资源类型 | 具体资源 | 重点读什么 |
|----------|----------|------------|
| 📄 论文 | [Deep Learning with Limited Numerical Precision (ICML 2015)](https://arxiv.org/abs/1502.02551) | 量化训练的早期探索，理解动态固定点 |
| 📄 论文 | [Quantization and Training of Neural Networks (CVPR 2018)](https://arxiv.org/abs/1712.05877) | 经典的 INT8 量化推理方案，Google 出品 |
| 📖 教程 | [PyTorch Quantization 官方文档](https://pytorch.org/docs/stable/quantization.html) | Numeric Suite 调试工具的使用 |
| 📖 博文 | [Lei Mao: Quantization for Neural Networks](https://leimao.github.io/) | 公式推导清晰，有代码 |
| 🎥 视频 | [MIT 6.S191: Hardware Acceleration for Deep Learning](https://www.youtube.com/watch?v=5An7jFJm_0E) | INT8 Tensor Core 硬件原理 |
| 🛠 代码 | [OscarSavolainen/Quantization-Tutorials](https://github.com/OscarSavolainen/Quantization-Tutorials) | 从零实现量化 + QAT |

### 0.4 检验标准

- [ ] 能徒手推导 FP32 → INT8 的完整量化/反量化过程
- [ ] 能写出 MinMax / KL / MSE 三种校准器的 Python 实现
- [ ] 能解释为什么 4-bit 以下 stochastic rounding 变得重要
- [ ] 能画出 per-tensor vs per-channel vs per-group 的 scale 存储示意图

---

## 阶段 1：PyTorch 原生 QAT — 三种模式从入门到精通

> ⏱ 预计时间：2-3 周 | 🎯 难度：⭐⭐

PyTorch 的 QAT API 经历了三代演进，**每一代都要吃透**，因为不同框架（AIMET、PPQ、NNCF）的底层都借鉴了这些设计模式。

### 1.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **FakeQuantize 原理**：Observer + FakeQuant 的内部状态机 | 能讲清楚 `observer_enabled / fake_quant_enabled` 两个 flag 在三个阶段的切换 |
| 2 | **QConfig / QConfigMapping**：如何为不同 op 指定不同的量化策略 | 知道为什么 Conv 用 per-channel，Linear 用 per-tensor |
| 3 | **Fuse Modules**：Conv+BN+ReLU 融合的数学原理 | 能手算 Conv+BN 融合后的新 weight 和 bias |
| 4 | **Observer 类型**：MinMaxObserver / MovingAverageMinMaxObserver / HistogramObserver / PerChannelMinMaxObserver | 知道每种适用的场景 |
| 5 | **FX Graph Mode 的图改写机制**：`prepare_qat_fx` 在图上做了什么 | 能用 `print_readable()` 打印量化前后的图结构 |
| 6 | **PT2E (PyTorch 2 Export) 模式**：`capture_pre_autograd_graph` → `prepare_qat_pt2e` → `convert_pt2e` | 理解 PT2E 和 FX 的根本区别（export-based vs trace-based） |
| 7 | **QAT 训练技巧**：BN 冻结、Observer 停止、学习率调度 | 知道什么时候 disable observer |
| 8 | **Backend 适配**：fbgemm vs qnnpack vs x86 vs ARM | 知道每种 backend 对 op 的支持情况 |

### 1.2 怎么做

#### 路线 1.2a：Eager Mode QAT（理解历史）

```python
import torch
from torch.ao.quantization import (
    default_qconfig, prepare_qat, convert,
    prepare, fuse_modules
)

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").train()

# Step 1: 融合 Conv+BN+ReLU
model = fuse_modules(model, [["conv1", "bn1", "relu"]])

# Step 2: 设置 qconfig
model.qconfig = default_qconfig  # fbgemm: act=histogram_obs, wt=per_channel_minmax

# Step 3: QAT 准备
model_prepared = prepare_qat(model)

# Step 4: QAT 训练循环
for epoch in range(5):
    model_prepared.train()
    for data, target in train_loader:
        # forward pass 自动包含 FakeQuant
        output = model_prepared(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 验证阶段关闭 observer
    model_prepared.apply(torch.ao.quantization.disable_observer)
    model_prepared.eval()
    # ... 验证 ...

# Step 5: 转换为真 INT8 模型
model_prepared.eval()
model_int8 = convert(model_prepared)
```

#### 路线 1.2b：FX Graph Mode QAT（主力）

```python
import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

model = torchvision.models.mobilenet_v3_small(weights="DEFAULT").train()
example_inputs = (torch.randn(1, 3, 224, 224),)

# QConfigMapping — 精细控制每一层
qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")

# 手动定制：第一层和最后一层保持 FP32（常见技巧）
qconfig_mapping.set_global(torch.ao.quantization.get_default_qat_qconfig("qnnpack"))

# 准备 QAT
model_prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# 打印图结构
model_prepared.graph.print_tabular()

# QAT 训练（同 eager mode）
# ...

# 转换为 INT8
model_prepared.eval()
model_int8 = convert_fx(model_prepared)

# 保存 + 导出 ONNX
torch.onnx.export(model_int8, example_inputs, "model_int8.onnx",
                  opset_version=17)
```

#### 路线 1.2c：PT2E (PyTorch 2 Export) Mode QAT（未来方向）

```python
import torch
from torch.export import export_for_training  # PyTorch 2.5+
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").train()
example_inputs = (torch.randn(2, 3, 224, 224),)

# Step 1: Capture graph
# PyTorch < 2.5: from torch._export import capture_pre_autograd_graph
exported_model = capture_pre_autograd_graph(model, example_inputs)

# Step 2: 配置后端 Quantizer
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_qat=True))

# Step 3: Prepare QAT
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

# Step 4: 训练（同标准训练循环）
# 建议在几个 epoch 后 disable observer
for epoch in range(num_epochs):
    train_one_epoch(prepared_model, ...)
    if epoch >= 3:
        prepared_model.apply(torch.ao.quantization.disable_observer)

# Step 5: Convert
quantized_model = convert_pt2e(prepared_model)
torch.ao.quantization.move_exported_model_to_eval(quantized_model)
```

### 1.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 📖 官方 | [PyTorch 2 Export QAT Tutorial](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html) | PT2E QAT 完整教程（ResNet18 + ImageNet） |
| 📖 官方 | [FX Graph Mode Quantization](https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization) | FX 图模式的 API 和调试方法 |
| 🛠 GitHub | [OscarSavolainen/Quantization-Tutorials](https://github.com/OscarSavolainen/Quantization-Tutorials) | YouTube 配套代码，ResNet QAT FX Mode |
| 📄 设计文档 | [PyTorch Quantization Design Proposal](https://github.com/pytorch/pytorch/wiki/Quantization-Design-Proposal) | 理解 API 的设计哲学 |
| 🎥 视频 | OscarSavolainen 量化教程系列 (YouTube) | 一步步手把手教学 |

### 1.4 关键细节：QAT 训练技巧

```python
# 技巧 1：BN 冻结 — QAT 时 BatchNorm 统计量应该冻结
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

# 技巧 2：Observer 停止 — 前几个 epoch 收集统计数据，之后冻结
def disable_observers(model):
    model.apply(torch.ao.quantization.disable_observer)

# 技巧 3：QAT 学习率调度 — 通常从 FP32 模型 final LR 开始
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)  # 低学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 完整 QAT 调度示例
for epoch in range(10):
    if epoch == 0:
        model.apply(torch.ao.quantization.enable_observer)    # epoch 0: 开启 observer
    if epoch >= 3:
        model.apply(torch.ao.quantization.disable_observer)   # epoch 3+: 停止更新 scale

    train_one_epoch(...)
    scheduler.step()
```

### 1.5 检验标准

- [ ] 能在一个 MobileNetV3 上跑通 Eager / FX / PT2E 三种模式的 QAT
- [ ] 能解释 `prepare_qat_fx` 在 FX 图上插入了哪些节点（打印图对比）
- [ ] 能说出 BN 冻结的原因（QAT 时 forward path 里的 FakeQuant 会改变 activation 分布）
- [ ] 能量化前后的模型大小和推理速度

---

## 阶段 2：QAT 核心算法 — LSQ 与可微量化参数

> ⏱ 预计时间：2-3 周 | 🎯 难度：⭐⭐⭐

这是 QAT 理论的**核心中的核心**。LSQ 是 QAT 领域的里程碑，几乎所有后续 QAT 方法（包括 AIMET 的 QAT）都基于 LSQ 的思想。

### 2.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **STE (Straight-Through Estimator)**：为什么需要 STE？STE 的数学问题 | 能手写 STE 的前向和反向代码 |
| 2 | **LSQ 的核心公式**：`v̂ = round(clip(v/s, -Q_N, Q_P)) * s`，s 是可学习参数 | 能从公式推导出 s 的梯度表达式 |
| 3 | **LSQ 梯度缩放 (Gradient Scaling)**：`g = 1/sqrt(N_W * Q_P)` | 理解为什么需要对 step size 梯度缩放 |
| 4 | **LSQ+**：将 zero_point 也变为可学习参数，推广到非对称量化 | 理解 zero_point 的梯度怎么推 |
| 5 | **PACT (Parameterized Clipping Activation)**：学习 clip 上界 | 对比 PACT 和 LSQ 的异同 |
| 6 | **DoReFa-Net**：低比特权重量化 + 激活量化 | 理解不同比特宽的量化函数定义 |
| 7 | **QAT vs PTQ 的理论差异**：为什么 QAT 在 ≤4-bit 时远优于 PTQ | 理解量化噪声在训练过程中的"补偿"机制 |

### 2.2 怎么做

#### 2.2a：从零实现 LSQ（核心代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSQQuantizerFunction(torch.autograd.Function):
    """
    LSQ 的底层 autograd 实现：
    - 前向：对输入做 FakeQuant
    - 反向：对 scale 和输入分别计算梯度（STE）
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, n_bits, symmetric):
        """
        x: float tensor
        scale: 可学习参数 (per-tensor 或 per-channel)
        n_bits: 量化比特数
        """
        if symmetric:
            qmin = -2 ** (n_bits - 1)
            qmax = 2 ** (n_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** n_bits - 1

        # 量化 + 反量化 (FakeQuant)
        x_scaled = x / scale
        x_rounded = torch.round(x_scaled)
        x_clamped = torch.clamp(x_rounded, qmin, qmax)
        x_quant = x_clamped * scale

        # 保存用于反向传播
        ctx.save_for_backward(x, scale, x_scaled, x_clamped)
        ctx.qmin, ctx.qmax = qmin, qmax
        ctx.n_bits = n_bits
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：
        - 对 x: 直通估计器 (STE) — 直接回传梯度
        - 对 scale: 按照 LSQ 论文的梯度公式
        """
        x, scale, x_scaled, x_clamped = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        n_bits = ctx.n_bits

        # === 对输入 x 的梯度: STE ===
        # 在 (qmin, qmax) 范围内的位置直接回传梯度
        grad_x = grad_output.clone()
        # 超出量化范围的位置梯度置零（hard clip 不可导）
        grad_x = torch.where(
            (x_scaled >= qmin) & (x_scaled <= qmax),
            grad_x,
            torch.zeros_like(grad_x)
        )

        # === 对 scale 的梯度: LSQ 论文核心 ===
        # ∂v̂/∂s ≈ -v/s + round(v/s)        when qmin < v/s < qmax
        #          qmin                        when v/s <= qmin
        #          qmax                        when v/s >= qmax
        grad_scale_inner = -x_scaled + x_clamped  # 论文公式 (6)
        grad_scale_inner = torch.where(
            (x_scaled >= qmin) & (x_scaled <= qmax),
            grad_scale_inner,
            torch.where(x_scaled < qmin,
                        torch.full_like(grad_scale_inner, float(qmin)),
                        torch.full_like(grad_scale_inner, float(qmax)))
        )

        grad_scale = (grad_output * grad_scale_inner).sum()

        # === LSQ Gradient Scaling ===
        # g = 1 / sqrt(N_W * Q_P) — 论文公式 (13)
        g = 1.0 / (x.numel() * qmax) ** 0.5
        grad_scale = grad_scale * g

        return grad_x, grad_scale, None, None, None


class LSQQuantizer(nn.Module):
    """可插入任意网络的 LSQ 量化模块"""
    def __init__(self, n_bits=4, symmetric=True, init_scale=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        # scale 是可学习参数
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        fq = LSQQuantizerFunction.apply
        return fq(x, self.scale, None, self.n_bits, self.symmetric)


# ========== 使用示例：将 LSQ 插入 ResNet ==========
class Conv2dLSQ(nn.Module):
    """带 LSQ 量化的 Conv2d 层"""
    def __init__(self, in_c, out_c, k, stride=1, padding=0, n_bits_w=4, n_bits_a=4):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride, padding)
        self.weight_quant = LSQQuantizer(n_bits_w, symmetric=True)
        self.act_quant = LSQQuantizer(n_bits_a, symmetric=False)

    def forward(self, x):
        w_q = self.weight_quant(self.conv.weight)  # 量化权重
        x_q = self.act_quant(x)                      # 量化激活
        return F.conv2d(x_q, w_q, stride=self.conv.stride,
                        padding=self.conv.padding)
```

#### 2.2b：复现 LSQ 论文实验

```python
import torchvision
from torchvision import transforms, datasets

# 用自实现的 LSQ 做 ResNet18 ImageNet 4-bit QAT
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# 将 Conv2d 替换为 Conv2dLSQ
# ... (替换代码)
# QAT 训练 90 epochs, lr=1e-3, cosine schedule
# 目标: 4-bit 精度不低于 FP32 的 1%
```

### 2.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 📄 必读 | [LSQ: Learned Step Size Quantization (ICLR 2020)](https://arxiv.org/abs/1902.08153) | 从头到尾读 3 遍，公式 (5)-(13) 必须手推 |
| 📄 扩展 | [LSQ+ (Qualcomm)](https://arxiv.org/abs/2004.09576) | zero_point 可学习化，非对称量化扩展 |
| 📄 对比 | [PACT: Parameterized Clipping Activation (ICML 2018)](https://arxiv.org/abs/1805.06085) | 只学习 clip 上界，与 LSQ 对比理解 |
| 📄 历史 | [DoReFa-Net (2016)](https://arxiv.org/abs/1606.06160) | 低比特训练的开山之作 |
| 🛠 代码 | [zhutmost/lsq-net](https://github.com/zhutmost/lsq-net) | LSQ 非官方 PyTorch 实现 |
| 🛠 代码 | [Kelvinyu1117/LSQ-implementation](https://github.com/Kelvinyu1117/LSQ-implementation) | 另一个参考实现 |
| 📖 中文 | [论文阅读——LSQ (CSDN)](https://blog.csdn.net/qq_37151108/article/details/108666779) | 中文解读，公式推导详细 |

### 2.4 关键数学推导（必须手推）

#### LSQ 的 scale 梯度推导

```
前向: v̂ = round(clamp(v/s, -Q_N, Q_P)) * s

令 v_n = v/s, v̄ = clamp(v_n, -Q_N, Q_P), v̂ = round(v̄) * s

反向: ∂v̂/∂s 需要通过 STE 近似

当 -Q_N < v_n < Q_P 时:
  ∂v̂/∂s = ∂(round(v/s) * s)/∂s
         ≈ ∂((v/s) * s)/∂s           # STE: round(x) ≈ x 在反向
         = ∂v/∂s = 0                  # 不对！v 不是 s 的函数
  ❌ 以上推导错误！

✅ 正确推导:
  在 STE 框架下，round(z) 的反向导数是 1。
  所以 v̂ = round(v/s) * s，对 s 求导：

  ∂v̂/∂s = round(v/s) + s * ∂round(v/s)/∂s
         = round(v/s) + s * (∂round(v/s)/∂(v/s)) * ∂(v/s)/∂s
         = round(v/s) + s * 1 * (-v/s²)         # STE: round'(z) ≈ 1
         = round(v/s) - v/s

  ∴ ∂v̂/∂s = -v/s + round(v/s)  ← 这就是 LSQ 的公式 (6)！
```

### 2.5 检验标准

- [ ] 能手推 LSQ 公式 (6) 和 (13)，向身边人讲清楚
- [ ] 在 MNIST 上用自实现的 LSQ 跑通 2-bit QAT
- [ ] 对比 LSQ 与普通 QAT（固定 scale）在 4-bit 下的精度差异
- [ ] 理解 Gradient Scaling 的公式含义：为什么分母是 `sqrt(N * Q_P)`？

---

## 阶段 3：PTQ 进阶算法 — AdaRound / FlexRound / GPTQ 的数学内核

> ⏱ 预计时间：3-4 周 | 🎯 难度：⭐⭐⭐⭐

**这一段是打通 QAT 和 PTQ 任督二脉的关键**。虽然你目标是 QAT，但现代 QAT（如 AIMET）大量借鉴了 PTQ 的优化技术（Cross-Layer Equalization、AdaRound），而且 GPTQ 的自适应舍入和 LSQ 的 learnable rounding 在数学上是相通的。

### 3.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **AdaRound 的理论框架**：Taylor 展开 → 二次无约束二值优化 (QUBO) | 能手推 Taylor 展开到 QUBO 的完整过程 |
| 2 | **连续松弛 (Soft Relaxation)**：离散 rounding 选择 → 连续 sigmoid 近似 | 理解公式 `s = σ((h(W) - 0.5) / τ)` 中 τ 的作用 |
| 3 | **FlexRound 的 Element-wise Division**：与 AdaRound element-wise addition 的区别 | 能画出两种舍入方案的对比图 |
| 4 | **OBQ → GPTQ 的演进**：per-row quantization → Hessian-based error compensation | 理解为什么要用 Hessian inverse |
| 5 | **Block-wise Reconstruction**：逐层重建 vs 逐块重建 | 理解 `min ||W_l * X - Ŵ_l * X||_2^2` 的优化目标 |
| 6 | **Cross-Layer Equalization (CLE)**：利用 ReLU 的 scale 不变性进行层间均衡 | 理解为什么 CLE 对 MobileNet 效果显著 |
| 7 | **Bias Correction**：量化后 bias 的补偿 | 手算 `E[y_fp] - E[y_q]` 并修正 bias |

### 3.2 怎么做

#### 3.2a：从零实现 AdaRound

```python
import torch
import torch.nn as nn

class AdaRound(nn.Module):
    """
    AdaRound: 通过连续松弛学习每个权重的舍入方向（向上 or 向下）

    核心思想:
    - 普通量化: floor(x/s) 或 round(x/s)
    - AdaRound: floor(x/s) + h, 其中 h ∈ {0, 1} 是可学习的
    - 优化 h 使得每层的输出重建误差最小
    """
    def __init__(self, weight_shape, alpha=0.5, beta=2.0):
        super().__init__()
        # h 是连续变量，通过 sigmoid 映射到 [0, 1]
        self.h = nn.Parameter(torch.full(weight_shape, alpha))

    def forward(self, w_scaled):
        """
        w_scaled: w / s — 已经除以 scale 的权重
        返回: 软量化后的权重（连续版本，用于训练）
        """
        # 软舍入: floor(z) + σ((h - 0.5) / τ)
        w_floor = torch.floor(w_scaled)
        # Sigmoid 映射将 h 变换到 [0, 1]，控制舍入方向
        w_soft = w_floor + torch.sigmoid(self.h)  # 可微
        return w_soft

    def get_hard_rounding(self, w_scaled):
        """训练结束后，硬的 0/1 决定"""
        return torch.floor(w_scaled) + (self.h >= 0).float()


def adaround_layer(weight, scale, input_data, n_iter=10000, lr=1e-3):
    """
    对一层做 AdaRound 优化

    Args:
        weight: float weight tensor
        scale: 量化 scale
        input_data: 校准数据通过该层后的输入 X
    """
    # 量化网格
    w_scaled = weight / scale

    # 初始化 AdaRound 模块
    adaround = AdaRound(w_scaled.shape)

    # 优化 h，最小化输出重建误差
    opt = torch.optim.Adam([adaround.h], lr=lr)

    for i in range(n_iter):
        opt.zero_grad()

        # 软量化
        w_soft_q = adaround(w_scaled) * scale

        # 输出重建误差
        output_fp = input_data @ weight.T   # 全精度输出
        output_q = input_data @ w_soft_q.T  # 量化输出
        loss = ((output_fp - output_q) ** 2).mean()

        loss.backward()
        opt.step()

        # 逐渐降低 sigmoid 温度，使逼近硬舍入
        if i % 1000 == 0:
            print(f"Iter {i}, Loss: {loss.item():.6f}")

    return adaround.get_hard_rounding(w_scaled) * scale
```

#### 3.2b：理解 Cross-Layer Equalization (CLE)

```python
def cross_layer_equalization(conv1_weight, conv2_weight):
    """
    利用 ReLU 的 scale 不变性: s * ReLU(x / s) = ReLU(x)

    对 MobileNet 的 depthwise + pointwise conv pair:
    - conv1: depthwise, per-channel 权重
    - conv2: pointwise (1x1), per-channel 权重
    """

    # Step 1: 计算均衡因子 S_i
    with torch.no_grad():
        # S_i = sqrt(max(|W1[i,:,:,:]|) / max(|W2[:,i,:,:]|))
        range1 = conv1_weight.abs().amax(dim=(1,2,3))  # per-output-channel
        range2 = conv2_weight.abs().amax(dim=(0,2,3))  # per-input-channel
        S = torch.sqrt(range1 * range2) / range1

        # Step 2: 均衡（不改变网络输出！）
        # conv1: W1'[i,:,:,:] = W1[i,:,:,:] / S[i]
        conv1_weight = conv1_weight / S[:, None, None, None]

        # conv2: W2'[:,i,:,:] = W2[:,i,:,:] * S[i]
        conv2_weight = conv2_weight * S[None, :, None, None]

        # Step 3 (可选): 吸收到下一层 BN
        # BN: y = γ * (x - μ) / σ + β
        # 将 S 吸收到 BN 的 γ 中（如果 conv 后接 BN）

    return conv1_weight, conv2_weight
```

### 3.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 📄 必读 | [AdaRound (ICML 2020)](https://arxiv.org/abs/2004.10568) | Taylor 展开 → QUBO → Soft Relaxation 完整推导 |
| 📄 必读 | [FlexRound (ICML 2023)](https://arxiv.org/abs/2306.00317) | element-wise division 替代 addition |
| 📄 必读 | [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323) | OBQ → GPTQ 的规模化改造 |
| 📄 | [OBQ (NeurIPS 2022)](https://arxiv.org/abs/2208.11580) | GPTQ 的前置，理解 per-row quantization |
| 📄 | [DFQ: Data-Free Quantization (ICCV 2019)](https://arxiv.org/abs/1906.04721) | CLE + Bias Correction 的原始论文 |
| 📄 | [LRQ (NAACL 2025)](https://arxiv.org/abs/2407.11534) | FlexRound 的改进版，低秩权重缩放 |
| 📖 解读 | [AdaRound 论文详解 (腾讯云)](https://cloud.tencent.cn/developer/article/1799252) | 中文详解 |
| 🛠 代码 | [FlexRound GitHub](https://github.com/onliwad101/FlexRound_LRQ) | FlexRound + LRQ 官方实现 |

### 3.4 检验标准

- [ ] 能手推 AdaRound 的 Taylor 展开 → QUBO 转化过程
- [ ] 用一个 3 层 MLP 复现 AdaRound，对比 round-to-nearest 的误差
- [ ] 理解 GPTQ 的 "Lazy Batch Update" 为什么可以用 Cholesky 分解加速
- [ ] 搞清楚 CLE 为什么对 MobileNet 这种 depthwise separable conv 有效（画图）

---

## 阶段 4：YOLO 量化实战 — 端到端项目

> ⏱ 预计时间：2-3 周 | 🎯 难度：⭐⭐⭐

现在你有理论了，开始做项目。YOLO 是理想的第一个量化目标：**检测头的量化难点 + NMS 的量化适配**。

### 4.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **检测网络的量化敏感点**：检测头（regression）比分类头对量化更敏感 | 知道为什么 bbox 回归需要更高精度 |
| 2 | **SiLU/Swish 激活函数的量化**：不像 ReLU 那样容易量化 | 知道 SiLU x/sigmoid 的量化近似方案 |
| 3 | **NMS 的量化适配**：NMS 是 FP 操作，量化模型需要保持 FP32 bbox decode | 理解 decode + NMS 的精度保持策略 |
| 4 | **Conv+BN+SiLU 融合**：Fuse 后 SiLU 如何处理 | 知道融合顺序和量化节点插入位置 |
| 5 | **mAP 评估 vs Top-1 Accuracy**：检测任务的评估指标 | 知道为什么 mAP drop 0.5 是"可接受"的 |
| 6 | **TensorRT INT8 推理**：从 QAT 模型导出到 TensorRT 引擎 | 理解 calibration cache vs QAT 的关系 |

### 4.2 怎么做

#### 项目 A：YOLOv5/v8/v11 QAT（推荐从 YOLOv8 开始）

```python
# ========== YOLOv8 QAT 完整流程 ==========
from ultralytics import YOLO
import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# Step 1: 加载 YOLO 模型
model = YOLO("yolov8n.pt").model.model  # 提取内部 nn.Module
model.train()

# Step 2: 融合 Conv+BN+SiLU
# 注意 YOLOv8 用 SiLU 不是 ReLU，融合策略不同
# (需要手动写 fuse pattern)

# Step 3: QAT 准备
qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
model_prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# Step 4: QAT 训练（关键！）
# 检测损失 = 分类损失 + bbox 回归损失 + DFL 损失
for epoch in range(20):
    model_prepared.train()
    for batch in coco_loader:
        images, targets = batch
        loss, loss_items = model_prepared(images)  # YOLO 内置损失
        loss.backward()
        optimizer.step()

    # 每 5 个 epoch 评估 mAP
    if epoch % 5 == 0:
        mAP = evaluate_coco(model_prepared, coco_val)
        print(f"Epoch {epoch}, mAP@0.5: {mAP:.4f}")

# Step 5: 转换为 INT8
model_int8 = convert_fx(model_prepared.eval())

# Step 6: 导出 ONNX → TensorRT
torch.onnx.export(model_int8, example_inputs, "yolov8n_int8.onnx",
                  opset_version=17,
                  input_names=["images"],
                  output_names=["output0"])
```

#### 项目 B：对比实验 — PTQ vs QAT on YOLO

```python
# ========== 同一个 YOLO 模型，对比 PTQ 和 QAT ==========

# 方案 1: PTQ (calibration only, no training)
model_ptq = ptq_quantize(yolo_model, calib_loader)   # mAP: 下降 ~15-25%
print(f"PTQ mAP: {ptq_mAP}")

# 方案 2: QAT (5 epochs)
model_qat_5ep = qat_quantize(yolo_model, train_loader, epochs=5)  # mAP: 下降 ~3-5%
print(f"QAT (5ep) mAP: {qat_mAP}")

# 方案 3: QAT (20 epochs) + LSQ
model_qat_lsq = lsq_quantize(yolo_model, train_loader, epochs=20)  # mAP: 下降 <2%
print(f"QAT+LSQ mAP: {qat_mAP}")

# ========== 关键洞察 ==========
# 1. PTQ 在 ≤8-bit 的检测任务上通常表现很差
# 2. QAT 可以大幅恢复精度
# 3. LSQ 进一步缩小与 FP32 的差距
```

#### 实验记录模板

| 实验 | 方法 | 比特宽 | mAP@0.5 | mAP@0.5:0.95 | 模型大小 | 推理速度 (ms) |
|------|------|--------|---------|--------------|----------|---------------|
| 基线 | FP32 | 32 | 37.3 | 26.1 | 12.1 MB | 8.5 |
| 实验1 | PTQ MinMax | 8 | 30.2 | 19.8 | 3.4 MB | 3.2 |
| 实验2 | PTQ MSE | 8 | 31.5 | 20.1 | 3.4 MB | 3.2 |
| 实验3 | QAT (5 ep) | 8 | 35.1 | 24.3 | 3.4 MB | 3.2 |
| 实验4 | QAT+LSQ (20 ep) | 8 | 36.2 | 25.4 | 3.4 MB | 3.2 |
| 实验5 | QAT+LSQ | 4 | 33.8 | 23.1 | 2.1 MB | 2.1 |

### 4.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 🛠 GitHub | [wangyi2019py/yolov8-QAT3](https://github.com/wangyi2019py/yolov8-QAT3) | YOLOv8 QAT PyTorch + TensorRT 双路径实现 |
| 📖 社区 | [Ultralytics YOLOv11 QAT](https://community.ultralytics.com/t/ultralytics-yolov11-qat/926) | Callback 方式注入 QAT |
| 📖 教程 | [AMD Quark YOLO-X](https://quark.docs.amd.com/release-0.9/pytorch/sample_yolo_x_tiny_quant.html) | AMD NPU 上的 YOLO 量化 |
| 📖 教程 | [YOLOv5 INT8 QAT (百度)](https://developer.baidu.com/article/details/3321995) | 中文教程 |
| 🛠 | [OpenVINO NNCF YOLOv8](https://github.com/openvinotoolkit/openvino_notebooks) | Intel 的量化 pipeline |


### 4.4 检验标准

- [ ] 在 COCO 2017 val 上跑出 PTQ vs QAT vs QAT+LSQ 的完整对比表
- [ ] 量化后的 YOLO 模型导出 ONNX 并用 onnxruntime 推理
- [ ] 理解为什么检测头的 regression branch 比 classification branch 对量化更敏感
- [ ] 能量化前后的 NMS 输出差异，理解精度损失的来源

---

## 阶段 5：工业级框架深度使用 — PPQ / AIMET

> ⏱ 预计时间：3-4 周 | 🎯 难度：⭐⭐⭐⭐

学完理论和 YOLO 实战后，开始用工业级框架。PPQ（商汤）和 AIMET（高通）是两个代表性框架，各有侧重。

### 5.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **PPQ 的量化 IR 设计**：`QuantableOperation` / `QuantableVariable` 的抽象 | 理解 PPQ 如何用 IR 统一描述不同后端的量化 |
| 2 | **PPQ 的 Pass 系统**：27 个独立量化优化 Pass | 至少吃透 5 个关键 Pass：LayerEqualization / BiasCorrection / LSQ / ChannelSplit / Fusion |
| 3 | **PPQ 的多后端导出**：ONNX QDQ / TensorRT / OpenVINO / NCNN / SNPE | 至少跑通 ONNX QDQ 和 TensorRT 两条路径 |
| 4 | **AIMET QuantSim**：模拟量化的核心机制 | 理解 AIMET 怎么在 PyTorch 层间插入量化/反量化节点 |
| 5 | **AIMET QAT 模式**：带 Range Learning vs 不带 Range Learning | 理解 scale/offset 作为可学习参数的具体实现 |
| 6 | **AIMET CLE / Bias Correction / AdaRound** 在 AIMET 中的使用 | 走通 PTQ → QAT 的完整工作流 |
| 7 | **AIMET → QAIRT 部署链路**：模型 → QuantSim → ONNX + encodings → DLC | 至少理解整个链路的数据格式转换 |

### 5.2 怎么做

#### 5.2a：PPQ 实战

```python
# ========== PPQ 完整量化流程 ==========
from ppq import QuantizationSettingFactory, QuantSetting, TargetPlatform
from ppq.api import quantize_torch_model, export_ppq_graph
from ppq.executor import TorchExecutor
import torchvision

# Step 1: 加载模型
model = torchvision.models.mobilenet_v2(weights="DEFAULT")
dummy_input = torch.randn(1, 3, 224, 224)

# Step 2: 配置量化设置
q_setting = QuantizationSettingFactory.default_setting()
# 启用 LSQ（PPQ 内置）
q_setting.lsq_optimization = True
# 启用层均衡
q_setting.equalization = True
# 启用 bias 校正
q_setting.bias_correction = True

# Step 3: 量化（PPQ 自动做 PTQ）
quantized = quantize_torch_model(
    model=model,
    calib_dataloader=calib_loader,
    calib_steps=32,
    input_shape=(1, 3, 224, 224),
    setting=q_setting,
    platform=TargetPlatform.TRT_INT8,  # 目标后端
)

# Step 4: 导出为 ONNX (带 QDQ 节点)
export_ppq_graph(
    graph=quantized,
    platform=TargetPlatform.ONNX,
    graph_save_to="mobilenet_v2_int8.onnx",
    config_save_to="quant_config.json",
)

# ========== PPQ QAT（在 PTQ 基础上微调） ==========
from ppq.api import enable_qat

# 在 PPQ 的量化图上做 QAT
qat_graph = enable_qat(quantized, dataloader=train_loader)
# 训练...
# 导出...
```

#### 5.2b：AIMET 实战（高通 QAT 完整流程）

```python
# ========== AIMET QAT 完整流程（ResNet18 + ImageNet） ==========
import torch
from torchvision import models
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.bias_correction import bias_correction

# Step 1: 加载预训练模型
model = models.resnet18(weights="IMAGENET1K_V1").eval()
dummy_input = torch.randn(1, 3, 224, 224)

# Step 2: PTQ 初始化 — 应用 CLE + Bias Correction (可选)
# equalize_model() 需要在 BN folded 后调用
# bias_correction() 进一步减小量化误差

# Step 3: 创建 QuantSim — AIMET 的核心量化模拟器
sim = QuantizationSimModel(
    model=model,
    quant_scheme="tf_enhanced",      # 量化方案: tf / tf_enhanced / range_learning
    dummy_input=dummy_input,
    rounding_mode="nearest",
    default_output_bw=8,             # 激活 8-bit
    default_param_bw=8,              # 权重 8-bit
    # config_file 可以传入自定义的每层配置（JSON）
)

# Step 4: 校准 — 计算激活的 scale/offset
def calibration_forward_pass(model, batch):
    model(batch.cuda())

sim.compute_encodings(
    forward_pass_callback=calibration_forward_pass,
    forward_pass_callback_args=5,    # 使用 5 个 batch 校准
)

# Step 5: QAT 微调
# sim.model 是插入了 FakeQuant 的模型
sim.model.train()
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5, momentum=0.9)

for epoch in range(15):
    for data, target in train_loader:
        output = sim.model(data.cuda())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 6: 导出
# 导出 ONNX + encodings 文件 → QAIRT 转 DLC
sim.export(
    path="./export/",
    filename_prefix="resnet18_qat",
    dummy_input=dummy_input.cuda(),
)

# Step 7 (可选): 用 QAIRT 转 DLC 部署到高通设备
# QAIRT Converter: ONNX + .encodings → DLC
```

#### 5.2c：AIMET Range Learning QAT

```python
# ========== AIMET Range Learning QAT ==========
# 与普通 QAT 的区别：scale 和 offset 也是可学习参数（类似 LSQ）
sim_range = QuantizationSimModel(
    model=model,
    quant_scheme="range_learning",   # ← 关键：使用 range_learning scheme
    dummy_input=dummy_input,
    default_output_bw=8,
    default_param_bw=8,
)

sim_range.compute_encodings(calibration_forward_pass, 5)

# Range Learning 训练时：
# - 权重的 scale 随训练更新
# - 激活的 scale/offset 也通过梯度更新
# - 类似 LSQ 的行为
```

### 5.3 从哪里学

| 框架 | 资源 | 链接 |
|------|------|------|
| PPQ | GitHub 官方仓库 | [github.com/OpenPPL/ppq](https://github.com/OpenPPL/ppq) |
| PPQ | 官方文档 + DeepWiki | [PPQ DeepWiki](https://deepwiki.com/OpenPPL/ppq/1-overview) |
| AIMET | 官方文档 v2.0 | [AIMET Quantization Workflow](https://quic.github.io/aimet-pages/releases/2.0.0/userguide/quantization_workflow.html) |
| AIMET | QAT 官方教程 | [AIMET QAT User Guide](https://quic.github.io/aimet-pages/releases/1.35.0/user_guide/quantization_aware_training.html) |
| AIMET | QAT 示例代码 | [GitHub Examples](https://github.com/qualcomm/aimet/blob/develop/Examples/torch/quantization/quantization_aware_training.py) |
| AIMET | 推荐工作流 | [AIMET Recommended Workflow](https://quic.github.io/aimet-pages/releases/2.0.0/userguide/quantization_workflow.html) |

### 5.4 关键对比：PPQ vs AIMET

| 对比维度 | PPQ | AIMET |
|----------|-----|-------|
| 厂商 | 商汤科技 (OpenPPL) | 高通 (Qualcomm) |
| 设计哲学 | 量化编译器的思路（IR → Pass → Backend） | 量化工具箱的思路（Technique → Sim → Export） |
| 后端覆盖 | ONNX, TensorRT, NCNN, OpenVINO, SNPE, DSP | **高通专精（SNPE/QAIRT → DLC → 骁龙芯片）** |
| 核心技术 | 图调度器 (conservative/aggressive) + 27 Pass | CLE + AdaRound + QuantSim + Range Learning QAT |
| 学习价值 | 学习量化编译器的设计模式 | 学习工业级 QAT pipeline |
| 劣势 | QAT 部分相对较新（v0.6.6 才加入） | 对非高通硬件无直接后端 |
| 社区活跃度 | 近 2 年无新提交 | 持续更新，有 Slack 社区 |

### 5.5 检验标准

- [ ] 用 PPQ 的 PTQ 跑通 MobileNetV2 → 导出 ONNX QDQ 模型
- [ ] 用 PPQ 的 QAT (LSQ Optimization) 对比 PTQ 精度
- [ ] 用 AIMET QuantSim 跑通 ResNet18 QAT（不含 range learning）
- [ ] 用 AIMET Range Learning QAT 对比普通 QAT
- [ ] 理解 PPQ 的图调度器 `conservative` vs `aggressive` 的区别
- [ ] 画出 PPQ 的量化 Pass 执行顺序图

---

## 阶段 6：大模型 PTQ — GPTQ / AWQ / SmoothQuant / SpinQuant

> ⏱ 预计时间：3-4 周 | 🎯 难度：⭐⭐⭐⭐

这是 2023-2025 年最重要的进展。在进入 LLM QAT 之前，**必须先彻底理解 LLM 的 PTQ**，因为 LLM QAT 几乎都是建立在 PTQ 的基础上的。

### 6.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **LLM 量化与 CNN 量化的根本区别**：激活值异常值 (outlier) 问题 | 能画出 Transformer 激活值的 heavy-tailed 分布图 |
| 2 | **GPTQ 算法全流程**：OBQ → Hessian inverse → Cholesky → Lazy Batch Update → Group Size | 能手推 GPTQ 的 per-row quantization 误差补偿公式 |
| 3 | **SmoothQuant 的核心思想**：将量化难度从激活迁移到权重 | 理解 `X * diag(s)^-1 * diag(s) * W = (X/s) * (W*s)` |
| 4 | **AWQ 的激活感知**：保护 ~1% 的显著权重（salient weights）| 理解 channel-wise scaling factor 怎么找到 |
| 5 | **SpinQuant 的旋转变换**：用可学习的旋转矩阵消除 outlier | 理解 `A * R * R^T * W^T = (A*R) * (W*R)^T` |
| 6 | **QuaRot 的 Hadamard 旋转**：随机旋转 vs 可学习旋转 | 对比 QuaRot 和 SpinQuant |
| 7 | **W4A16 vs W4A4 vs W4A4KV4** 的 trade-off | 知道每种方案的计算瓶颈在哪 |

### 6.2 怎么做

#### 6.2a：使用 AutoGPTQ 量化 LLaMA

```python
# ========== GPTQ 量化实战 ==========
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 配置 4-bit GPTQ
gptq_config = GPTQConfig(
    bits=4,               # 4-bit 量化
    group_size=128,       # 每 128 列共享 scale
    dataset="c4",         # 校准数据集
    desc_act=False,       # 是否按激活值降序排列（影响精度/速度 trade-off）
    damp_percent=0.01,    # Hessian 阻尼系数
)

# 一键量化
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 验证
perplexity = evaluate_ppl(model, tokenizer, "wikitext2")
print(f"4-bit GPTQ Perplexity: {perplexity}")
```

#### 6.2b：从零实现 GPTQ 核心逻辑（理解底层）

```python
# ========== GPTQ 核心算法的简化实现 ==========
import torch

def gptq_quantize_layer(W, X, bits=4, group_size=128, damp_percent=0.01):
    """
    GPTQ 单层量化（简化版，去掉了 lazy batch update for clarity）

    Args:
        W: shape [out_features, in_features] — 全精度权重
        X: shape [N, in_features] — 校准数据的输入特征
        bits: 量化比特数
        group_size: 分组大小
    """
    dev = W.device
    out_features, in_features = W.shape
    W = W.clone()

    # Step 1: 计算 Hessian H = X^T * X (in_features × in_features)
    H = (X.T @ X) / X.shape[0]
    H = H + damp_percent * torch.diag(H).mean() * torch.eye(in_features, device=dev)

    # Step 2: Cholesky 分解 (用于高效求逆)
    # H^-1 = (L * L^T)^-1
    L = torch.linalg.cholesky(H)

    # Step 3: 逐列量化（GPTQ 按固定顺序）
    Q = torch.zeros_like(W)  # 量化后的权重
    Scales = []
    Zeros = []

    for col in range(in_features):
        w_col = W[:, col]  # out_features 维

        # 3a: 计算该列的量化参数
        col_group = col // group_size
        w_min, w_max = w_col.min(), w_col.max()
        scale = (w_max - w_min) / (2**bits - 1)
        zero = torch.round(-w_min / scale)
        Scales.append(scale); Zeros.append(zero)

        # 3b: 量化
        w_col_q = (torch.round(w_col / scale + zero).clamp(0, 2**bits-1) - zero) * scale
        Q[:, col] = w_col_q

        # 3c: 计算量化误差
        error = w_col - w_col_q  # [out_features]

        # 3d: 用 Hessian 逆更新剩余列（OBQ 的核心更新公式）
        # 获取 H^-1 的第 col 个对角元素和第 col 列
        H_inv_col = torch.cholesky_solve(
            torch.eye(in_features, device=dev)[:, col:col+1], L
        ).squeeze()  # [in_features]

        # 更新公式: W[:, col+1:] -= error / H_inv[col] * H_inv[col+1:]
        if col < in_features - 1:
            correction = (error.unsqueeze(1) / H_inv[col]) * H_inv[col+1:].unsqueeze(0)
            W[:, col+1:] -= correction

    return Q, Scales, Zeros
```

#### 6.2c：SmoothQuant 原理动手

```python
# ========== SmoothQuant: 迁移量化难度 ==========
import torch

def smoothquant_scale(X_calib, W, alpha=0.5):
    """
    计算 SmoothQuant 的 per-channel scaling factor

    核心公式: s_j = max(|X|_j)^alpha / max(|W|_j)^(1-alpha)

    X: [seq_len, in_features]
    W: [out_features, in_features]
    """
    # 激活的 per-channel 最大值
    act_max = X_calib.abs().max(dim=0).values  # [in_features]

    # 权重的 per-channel 最大值
    wt_max = W.abs().max(dim=0).values         # [in_features]

    # 计算平滑因子
    s = (act_max ** alpha) / (wt_max ** (1 - alpha) + 1e-8)

    # 应用到激活和权重
    X_smoothed = X_calib / s            # 激活除 s（变小）
    W_smoothed = W * s.unsqueeze(0)     # 权重乘 s（变大）

    # 验证: 网络输出不变
    # X_smoothed @ W_smoothed.T = (X / s) @ (W * s).T ≈ X @ W.T

    return X_smoothed, W_smoothed, s
```

#### 6.2d：SpinQuant 原理动手

```python
# ========== SpinQuant 核心: 旋转矩阵消除 outlier ==========
import torch

def spinquant_hadamard_transform(x):
    """
    对输入应用 Hadamard 旋转（QuaRot 的思路）
    将 outlier 的能量"平摊"到所有维度
    """
    # x: [batch, seq_len, hidden_dim]
    # Hadamard 矩阵 H: H * H^T = n*I
    # 变换: x' = x * H / sqrt(n)

    import math
    n = x.shape[-1]

    # 递归生成 Hadamard 矩阵
    def hadamard(n):
        if n == 1:
            return torch.tensor([[1.0]])
        H_n1 = hadamard(n // 2)
        top = torch.cat([H_n1, H_n1], dim=1)
        bottom = torch.cat([H_n1, -H_n1], dim=1)
        return torch.cat([top, bottom], dim=0)

    H = hadamard(n).to(x.device)
    return x @ H / math.sqrt(n)


def spinquant_learn_rotation(X_calib, W, n_rotations=4, lr=0.01, steps=100):
    """
    可学习的旋转矩阵（SpinQuant 核心）
    使用 Cayley 变换参数化正交矩阵
    """
    d = X_calib.shape[-1]

    # 用 Cayley 参数化保证正交性: R = (I - A) * (I + A)^-1
    # 其中 A 是反对称矩阵: A = (M - M^T) / 2
    M = torch.randn(d, d, requires_grad=True) * 0.01

    def cayley(M):
        A = (M - M.T) / 2
        I = torch.eye(d, device=M.device)
        return torch.linalg.solve((I + A).T, (I - A).T).T  # R = (I-A)(I+A)^-1

    opt = torch.optim.Adam([M], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        R = cayley(M)

        # 量化旋转变换后的权重
        W_rot = W @ R
        W_q = fake_quantize(W_rot, bits=4)  # 模拟 4-bit 量化
        W_deq = W_q @ R.T  # 反变换

        # 最小化量化误差
        loss = ((W - W_deq) ** 2).mean()
        loss.backward()
        opt.step()

    return cayley(M).detach()
```

### 6.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 📄 必读 | [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323) | OBQ → GPTQ, 每行代码都要理解 |
| 📄 必读 | [SmoothQuant (ICML 2023)](https://arxiv.org/abs/2211.10438) | 核心公式 (3) 的数学直觉 |
| 📄 必读 | [AWQ (MLSys 2024)](https://arxiv.org/abs/2306.00978) | 为什么保护 1% salient channel 就够 |
| 📄 | [SpinQuant (ICLR 2025)](https://arxiv.org/abs/2405.16406) | 旋转不变性 + Cayley 优化 |
| 📄 | [QuaRot (2024)](https://arxiv.org/abs/2404.00456) | Hadamard 随机旋转 |
| 📖 博文 | [HuggingFace GPTQ 集成教程](https://huggingface.co/blog/gptq-integration) | 实用 AutoGPTQ 指南 |
| 🛠 代码 | [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | GPTQ 的工业级实现 |
| 🛠 代码 | [SpinQuant GitHub](https://github.com/facebookresearch/SpinQuant) | Meta 官方实现 |
| 🛠 代码 | [llm-awq (MIT HAN Lab)](https://github.com/mit-han-lab/llm-awq) | AWQ 官方实现 |

### 6.4 检验标准

- [ ] 能手推 GPTQ 的 OBQ 更新公式：`W[:, j+1:] -= e / H_inv[j, j] * H_inv[j, j+1:]`
- [ ] 能解释为什么 SmoothQuant 的 α=0.5 是一个好选择
- [ ] 用 AutoGPTQ 量化 Llama-3-8B → 4-bit，测量 wikitext2 PPL
- [ ] 对比 GPTQ / AWQ / SmoothQuant 在同一个 LLM 上的精度
- [ ] 理解为什么 SpinQuant 比 QuaRot 好（learned vs random rotation）

---

## 阶段 7：大模型 QAT — LLM-QAT / QLoRA / EfficientQAT

> ⏱ 预计时间：3-4 周 | 🎯 难度：⭐⭐⭐⭐⭐

这是最难但也最有价值的一个阶段。LLM 的 QAT 和 CNN 的 QAT 在工程上完全不同：你不可能对 70B 模型做全量 QAT。

### 7.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **LLM-QAT 的数据生成策略**：用大模型自身生成训练数据 | 理解为什么 pre-training data distillation 是必要的 |
| 2 | **QLoRA 的双重量化**：NF4 权重 + 4-bit 量化 LoRA adapter | 理解 double quantization 为什么能进一步节省 0.4 bit/param |
| 3 | **NF4 (NormalFloat4)**：非均匀量化，针对正态分布优化 | 能手写 NF4 的量化/反量化 |
| 4 | **EfficientQAT**：weight-only QAT → weight+activation QAT 两阶段策略 | 理解为何分两阶段训练 |
| 5 | **BitDistiller**：用更大的教师模型蒸馏到量化学生模型 | 理解 KD loss + quantization loss 的联合优化 |
| 6 | **知识蒸馏在 QAT 中的作用**：soft label 引导量化模型恢复精度 | 理解温度参数在 KD + QAT 中的设置 |
| 7 | **PEFT + 量化**：LoRA/QLoRA adapter 在量化模型上的训练 | 理解 base model 冻结 + adapter 训练的计算图 |

### 7.2 怎么做

#### 7.2a：QLoRA 实战（基础）

```python
# ========== QLoRA: 4-bit 模型 + LoRA 微调 ==========
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"

# Step 1: 4-bit 量化配置（NF4）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",      # NormalFloat4 数据类型
    bnb_4bit_use_double_quant=True, # 双重量化（再省 0.4 bit/param）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 BF16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Step 2: QLoRA 适配器配置
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                # LoRA 秩
    lora_alpha=32,       # LoRA 缩放系数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
# trainable params: ~4.2M / total: ~6.7B = 0.06%

# Step 3: 训练（标准 HF Trainer）
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
    ),
    train_dataset=dataset,
)
trainer.train()
```

#### 7.2b：LLM-QAT 理解

```python
# ========== LLM-QAT 核心思想（概念代码） ==========
"""
LLM-QAT 的核心流程:

Phase 1 — Data Generation (Pre-training Data Distillation):
  - 用 FP16 的大模型生成 token 序列
  - 保留 teacher 的 logits 作为软标签
  - 目的：让量化模型"看到"足够多的数据

Phase 2 — QAT with KD:
  - 加载数据 + 软标签
  - 插入 FakeQuantization 节点
  - 联合优化: L = α * L_LM + β * L_KD
    - L_LM: 标准的自回归语言模型损失
    - L_KD: 与 teacher logits 的 KL 散度

Phase 3 — 导出:
  - 转换为真实量化格式
"""

# 伪代码
def llm_qat_training(model_fp16, model_q, tokenizer, dataset):
    """
    model_fp16: 全精度教师模型
    model_q: 插入 FakeQuant 的学生模型
    """
    for batch in dataset:
        # Step 1: 从教师模型获取 logits
        with torch.no_grad():
            teacher_logits = model_fp16(batch['input_ids']).logits

        # Step 2: 学生模型前向（带 FakeQuant）
        student_logits = model_q(batch['input_ids']).logits

        # Step 3: 联合损失
        loss_lm = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            batch['labels'].view(-1)
        )
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean',
        ) * (T ** 2)

        loss = 0.5 * loss_lm + 0.5 * loss_kd

        loss.backward()
        optimizer.step()
```

#### 7.2c：EfficientQAT 两阶段策略

```python
# ========== EfficientQAT 的两阶段训练 ==========
"""
Stage 1: Weight-only Quantization
  - 只量化权重，激活保持 FP16
  - 这阶段训练快，因为激活不需要 FakeQuant
  - 恢复大部分精度损失

Stage 2: Weight + Activation Quantization
  - 在 Stage 1 的基础上，激活也加入量化
  - 训练变慢但进一步恢复精度
  - 最终达到 W4A4 或 W4A8
"""

# 伪代码
model_q = insert_weight_fakequant(model)     # Stage 1: 只量化权重
train(model_q, epochs=5)                     # 权重适应量化噪声

model_q = insert_activation_fakequant(model_q)  # Stage 2: 加入激活量化
train(model_q, epochs=5)                     # 联合适应
```

### 7.3 从哪里学

| 资源 | 链接 | 重点 |
|------|------|------|
| 📄 必读 | [QLoRA (NeurIPS 2023)](https://arxiv.org/abs/2305.14314) | NF4 + Double Quantization |
| 📄 必读 | [LLM-QAT (ICLR 2024)](https://arxiv.org/abs/2305.17888) | Data-free KD for QAT |
| 📄 | [EfficientQAT (2024)](https://arxiv.org/abs/2407.11062) | 两阶段 QAT |
| 📄 | [BitDistiller (2024)](https://arxiv.org/abs/2402.12231) | 蒸馏 + 量化联合 |
| 📖 教程 | [HuggingFace QLoRA Tutorial](https://huggingface.co/blog/4bit-transformers-bitsandbytes) | 手把手 QLoRA |
| 🛠 代码 | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4 实现 |

### 7.4 检验标准

- [ ] 用 QLoRA 微调一个 7B 模型，理解 NF4 的量化表
- [ ] 能解释 double quantization：对 scale 的 scale 做量化
- [ ] 对比 LLM-QAT（KD）和不带 KD 的 QAT 在 7B 模型上的 PPL
- [ ] 理解为什么 EfficientQAT 要分两阶段

---

## 阶段 8：端侧部署全链路 — 从量化模型到芯片推理

> ⏱ 预计时间：2-3 周 | 🎯 难度：⭐⭐⭐⭐

量化做完了，怎么部署？这个阶段打通**模型 → 芯片**的全链路。你已经了解 ONNX 图优化，这里做更深。

### 8.1 你需要吃透什么

| # | 知识点 | 深度要求 |
|---|--------|----------|
| 1 | **ONNX QDQ 格式**：QuantizeLinear / DequantizeLinear 节点在 ONNX 图中的语义 | 能手写一个带 QDQ 节点的 ONNX graph |
| 2 | **图优化 Pass**：Fuse QDQ + Conv、Remove Duplicate DQ 等 | 能写出 3 个常用的 QDQ 优化 Pass |
| 3 | **TensorRT INT8 推理**：Builder Config 中 INT8 flag 的含义 | 理解 calibration cache 与 QAT 模型的区别 |
| 4 | **高通 QAIRT → DLC** 链路：ONNX + encodings → DLC | 理解 encodings 文件的作用 |
| 5 | **ExecuTorch**：PyTorch 官方的端侧推理运行时 | 知道 delegate 的概念 |
| 6 | **量化推理精度 debug**：Numeric Suite / Layer-wise Error Analysis | 能快速定位哪一层的量化误差最大 |
| 7 | **Per-layer vs Per-channel 的硬件支持**：知道哪些硬件支持哪些粒度 | 理解为什么某些硬件不能 per-channel |

### 8.2 怎么做

#### 8.2a：导出 ONNX QDQ 并做图优化

```python
# ========== ONNX QDQ 导出 + 图优化 ==========
import onnx
from onnx import helper, numpy_helper

# Step 1: 从 PyTorch QAT 模型导出 ONNX（带 QDQ）
torch.onnx.export(
    model_int8,
    example_inputs,
    "model_int8_qdq.onnx",
    opset_version=17,          # QDQ 支持从 opset 13 开始
    input_names=["input"],
    output_names=["output"],
)

# Step 2: 加载并检查 ONNX 图
model_onnx = onnx.load("model_int8_qdq.onnx")
for node in model_onnx.graph.node:
    if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        print(f"  [{node.op_type}] {node.name}: {node.input[0]}")

# Step 3: 分析 QDQ 节点插入位置
# 期望的 pattern: DQ → Conv → Q → DQ → Conv → Q → ...
# 优化目标: DQ + Conv + Q 融合为 INT8 Conv
```

#### 8.2b：TensorRT INT8 推理

```python
# ========== TensorRT INT8 推理（带 QAT 模型） ==========
import tensorrt as trt
import pycuda.driver as cuda

# 方案 A: 使用 QAT 模型的 scale（更精确）
# 导出 ONNX QDQ → TensorRT 直接使用 QDQ 中记录的 scale

# 方案 B: TensorRT 自己做校准（不用 QAT 时）
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse(onnx_model.SerializeToString())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
# 关键：设置校准器（如果 ONNX 没有 QDQ）
# config.int8_calibrator = MyCalibrator(calib_data_loader)

engine = builder.build_serialized_network(network, config)
```

#### 8.2c：量化精度 Debug — Layer-wise Error Analysis

```python
# ========== 逐层误差分析 ==========
import torch
from torch.ao.quantization import NumericSuite

def layerwise_error_analysis(model_fp, model_q, dataloader):
    """定位哪些层量化误差最大"""
    errors = {}

    for name, module_fp in model_fp.named_modules():
        if isinstance(module_fp, (torch.nn.Conv2d, torch.nn.Linear)):
            module_q = dict(model_q.named_modules())[name]

            # 收集该层的输入输出
            error_sum = 0.0
            count = 0
            for batch in dataloader:
                with torch.no_grad():
                    out_fp = module_fp(batch)
                    out_q = module_q(batch)
                    # 相对误差
                    error = ((out_fp - out_q) ** 2).mean() / (out_fp ** 2).mean()
                    error_sum += error.item()
                    count += 1

            errors[name] = error_sum / count

    # 排序，定位误差最大的前 10 层
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    print("=== Top 10 layers with largest quantization error ===")
    for name, err in sorted_errors[:10]:
        print(f"  {name}: {err:.6f}")

    return errors
```

### 8.3 从哪里学

| 资源 | 链接 |
|------|------|
| 📖 官方 | [ONNX QDQ Documentation](https://onnx.ai/onnx/operators/) |
| 📖 官方 | [TensorRT INT8 Inference](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-inference) |
| 📖 官方 | [AIMET → QAIRT 部署](https://quic.github.io/aimet-pages/releases/2.0.0/userguide/quantization_workflow.html) |
| 📖 官方 | [ExecuTorch Quantization](https://pytorch.org/executorch/stable/tutorials/quantization-tutorial.html) |
| 📖 教程 | [ONNX Runtime INT8](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) |
| 🛠 工具 | [Netron](https://netron.app/) — ONNX 图可视化 |

### 8.4 检验标准

- [ ] 从 YOLO QAT 模型导出 ONNX QDQ，用 Netron 可视化 QDQ 节点
- [ ] 用 TensorRT builder 构建 INT8 引擎，对比 FP16 vs INT8 的延迟
- [ ] 写一个简单的 ONNX graph pass：融合相邻的 DQ + Conv + Q
- [ ] 画出从 PyTorch → ONNX QDQ → TensorRT 引擎的完整数据流图

---

## 阶段 9：前沿拓展 — FP8 / KV Cache 量化 / BitNet / MoE 量化

> ⏱ 预计时间：持续关注 | 🎯 难度：⭐⭐⭐⭐⭐

你已经在主路上走得很深了。这一阶段的内容是前沿方向，不需要全部吃透，但要知道它们的存在、解决什么问题、和你的工作有什么关系。

### 9.1 前沿方向清单

| # | 方向 | 核心问题 | 你应该知道什么 |
|---|------|----------|----------------|
| 1 | **FP8 量化** | E4M3 / E5M2 格式，比 INT8 更好的动态范围 | FP8 的位布局，与 INT8 的 trade-off |
| 2 | **KV Cache 量化** | 长上下文推理的最大瓶颈 | KV Cache 为什么对长序列特别重要 |
| 3 | **BitNet b1.58** | 三值化（-1, 0, +1），没有浮点乘法 | 1.58-bit 的含义 |
| 4 | **MoE 量化** | Expert 路由不平衡导致部分 expert 被过度量化 | MoE 架构的独特量化挑战 |
| 5 | **Diffusion Model 量化** | 迭代去噪过程中的时序误差累积 | 与 LLM 量化的根本不同 |
| 6 | **Microscaling (MX)** | OCP 标准的块浮点格式 | MXFP4 / MXFP6 的格式定义 |
| 7 | **FP4 训练 (NVIDIA Blackwell)** | 原生 FP4 Tensor Core | NVFP4 格式和应用场景 |

### 9.2 关键论文（持续追踪）

| 方向 | 必读论文 |
|------|----------|
| FP8 | [FP8 Formats for Deep Learning (2022)](https://arxiv.org/abs/2209.05433) |
| KV Cache | [KVQuant (2024)](https://arxiv.org/abs/2401.18079) |
| BitNet | [BitNet b1.58 (2024)](https://arxiv.org/abs/2402.17764) |
| Diffusion | [Q-Diffusion (ICCV 2023)](https://arxiv.org/abs/2302.04304) |
| MX | [Microscaling Data Formats (OCP 2023)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |

### 9.3 从哪里追踪

| 资源 | 链接 |
|------|------|
| 📖 Awesome List | [Awesome Low-Precision Training](https://github.com/Hao840/Awesome-Low-Precision-Training) |
| 📖 Survey | [ZSQ Survey (IJCAI 2025)](https://github.com/snudm-starlab/ZSQ-Survey) |
| 📖 Survey | [A Survey of Quantization in LLMs (2025)](https://arxiv.org/abs/2501.12345) |
| 🔔 追踪 | HuggingFace Daily Papers: `https://huggingface.co/papers` |
| 🔔 追踪 | arXiv cs.LG / cs.CV / cs.CL 每日新论文 |

---

## 附录：推荐书单 & 课程 & 社区

### 📚 书单

| 书名 | 适合阶段 | 备注 |
|------|----------|------|
| **《Efficient Deep Learning》** (2023, Menghani & Singh) | 全阶段 | 量化、剪枝、蒸馏、NAS 全覆盖 |
| **《Deep Learning for Coders with Fastai and PyTorch》** | 阶段 0-2 | PyTorch 实战基础 |
| **《TinyML》** (Warden & Situnayake) | 阶段 8 | 端侧推理入门 |

### 🎓 在线课程

| 课程 | 链接 | 相关度 |
|------|------|--------|
| MIT 6.S191: TinyML & Efficient Deep Learning | [YouTube](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) | ⭐⭐⭐⭐⭐ |
| MIT 6.5940: EfficientML | [课程网站](https://efficientml.ai/) | ⭐⭐⭐⭐⭐ |
| Fast.ai Practical Deep Learning | [fast.ai](https://course.fast.ai/) | ⭐⭐⭐ |

### 💬 社区

| 社区 | 入口 | 适合讨论 |
|------|------|----------|
| PyTorch Discussion Forum | [discuss.pytorch.org](https://discuss.pytorch.org/) | PyTorch QAT 问题 |
| Qualcomm AIMET Slack | [qualcomm-ai.herokuapp.com](https://qualcomm-ai.herokuapp.com/) | AIMET 使用问题 |
| HuggingFace Discord | [discord.gg/huggingface](https://discord.gg/huggingface) | LLM 量化讨论 |
| ONNX Slack | [LF AI & Data](https://slack.lfai.foundation/) | ONNX 图优化 |
| NVIDIA Developer Forum | [forums.developer.nvidia.com](https://forums.developer.nvidia.com/) | TensorRT 量化 |

---

## 🗺️ 学习路线快速参考图

```
阶段 0 (1-2周)     阶段 1 (2-3周)       阶段 2 (2-3周)        阶段 3 (3-4周)
  ┌─────────┐      ┌──────────┐        ┌─────────────┐       ┌───────────────┐
  │量化数学基础│ ──→ │PyTorch QAT│ ──→   │LSQ 核心算法  │ ──→  │AdaRound/GPTQ  │
  │+ 硬件原理 │      │三种模式   │        │+ 从零实现    │       │数学内核       │
  └─────────┘      └──────────┘        └─────────────┘       └───────────────┘
                                                                     │
                                          ┌──────────────────────────┘
                                          ↓
  阶段 4 (2-3周)     阶段 5 (3-4周)       阶段 6 (3-4周)        阶段 7 (3-4周)
  ┌──────────┐      ┌────────────┐       ┌──────────────┐      ┌──────────────┐
  │YOLO QAT  │ ──→  │PPQ + AIMET │ ──→   │LLM PTQ 全家桶│ ──→  │LLM QAT       │
  │端到端实战 │      │工业级框架  │       │GPTQ/Smooth/  │      │QLoRA/Efficient│
  └──────────┘      └────────────┘       │AWQ/SpinQuant │      └──────────────┘
                                          └──────────────┘              │
                                                 │                      │
                                                 ↓                      ↓
                                          阶段 8 (2-3周)        阶段 9 (持续追踪)
                                          ┌────────────┐       ┌──────────────┐
                                          │端侧部署全链路│       │前沿拓展      │
                                          │ONNX→TRT→DLC│       │FP8/KV/BitNet │
                                          └────────────┘       └──────────────┘
```

---

## 📋 每个阶段的 Checklist 汇总

### 阶段 0 ✓
- [ ] 手推 FP32 → INT8 量化公式
- [ ] 手写 MinMax / KL / MSE calibrator
- [ ] 理解 VNNI / DP4A 指令
- [ ] 理解 per-tensor vs per-channel vs per-group

### 阶段 1 ✓
- [ ] 在 MobileNetV3 上跑通 Eager / FX / PT2E 三种模式
- [ ] 打印 FX 图结构对比量化前后
- [ ] 能解释 BN 冻结的原因和时机
- [ ] 对比量化前后的模型大小和速度

### 阶段 2 ✓
- [ ] 手推 LSQ 公式，从零实现 LSQ
- [ ] 在 MNIST/CIFAR10 做 2/3/4-bit QAT 消融实验
- [ ] 理解 LSQ Gradient Scaling 的推导
- [ ] 对比 LSQ vs 固定 scale QAT

### 阶段 3 ✓
- [ ] 手推 AdaRound Taylor 展开 → QUBO
- [ ] 3 层 MLP 复现 AdaRound
- [ ] 理解 GPTQ OBQ 更新公式
- [ ] 理解 CLE + Bias Correction

### 阶段 4 ✓
- [ ] YOLOv8 QAT PTQ vs QAT vs QAT+LSQ 对比
- [ ] 量化模型导出 ONNX 推理
- [ ] 理解检测头回归分支的量化敏感性

### 阶段 5 ✓
- [ ] PPQ PTQ → ONNX QDQ 跑通
- [ ] PPQ QAT (LSQ Optimization) 使用
- [ ] AIMET QuantSim QAT（with/without range learning）
- [ ] 画出 PPQ Pass 执行顺序

### 阶段 6 ✓
- [ ] 手推 GPTQ OBQ 更新公式
- [ ] AutoGPTQ 量化 Llama-3-8B → 4-bit
- [ ] 对比 GPTQ / AWQ / SmoothQuant
- [ ] 理解 SpinQuant 旋转矩阵的作用

### 阶段 7 ✓
- [ ] QLoRA 微调 7B 模型
- [ ] 理解 NF4 + Double Quantization
- [ ] 理解 LLM-QAT 的数据蒸馏策略
- [ ] 理解 EfficientQAT 两阶段原因

### 阶段 8 ✓
- [ ] YOLO QAT → ONNX QDQ → Netron 可视化
- [ ] TensorRT INT8 engine 构建 + 测速
- [ ] 写一个 ONNX QDQ 图优化 pass
- [ ] 画全链路数据流图

### 阶段 9
- [ ] 知道 FP8 E4M3/E5M2 的位布局
- [ ] 知道 KV Cache 量化的价值
- [ ] 知道 BitNet 1.58 的含义

---

> 📅 **预计总时间**：4-6 个月（按每周投入 15-20 小时计算）
> 🎯 **学习后能达到的水平**：
> - 能独立完成从模型训练到端侧部署的全链路量化优化
> - 能读懂最新量化论文并复现核心算法
> - 能参与或主导团队内的量化方案选型与落地
>
> 祝学习顺利！🚀
