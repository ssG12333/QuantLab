# QuantLab · 模型量化深度学习笔记

> 从浮点数的 32-bit 位布局，到 LLaMA 的 4-bit QLoRA 微调——把量化从"调参"变成"可控的系统工程"。
> **🔥 持续更新：从底层数学到工业落地，全程吃透。**

[![Stages](https://img.shields.io/badge/stages-9-blue)](QAT_LEARNING_ROADMAP.md)
[![Docs](https://img.shields.io/badge/docs-2篇-green)](docs/)
[![Status](https://img.shields.io/badge/更新-持续进行中-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 这是什么？

一套面向 **模型量化（Model Quantization）** 的自学笔记，适合：

- 已经知道 PTQ/QAT 大致原理，但**不满足于"能调 API"**，想下沉到框架源码级别理解
- 了解深度学习基础（Sigmoid 查表、ONNX 图优化、端侧部署），但知识比较散，**想融会贯通**
- 想从 YOLO 这种小模型练手，**一路过渡到 LLaMA 级别的大模型量化**
- 想把 PyTorch QAT、AIMET、PPQ、GPTQ、AWQ、SmoothQuant、SpinQuant 这些名词**串成一条完整的知识链**

---

## 快速导航

| 你想... | 从这里开始 |
|---------|-----------|
| 了解全局路线 | [9 阶段学习路线图](QAT_LEARNING_ROADMAP.md) |
| 建立量化直觉 + 啃 PyTorch 源码 | [📖 Stage 0：量化基础与硬件基石](docs/Stage0_量化基础与硬件基石.md) |
| 搞懂 PyTorch QAT 三种模式 | [📖 Stage 1：PyTorch 原生 QAT](docs/Stage1_PyTorch原生QAT三种模式.md) |
| 查已拆解的 PyTorch 模块 | [🔍 源码分析索引](source-notes/README.md) |
| 跑独立可运行代码 | [💻 代码实现目录](code/README.md) |
| 读论文笔记 | [📝 论文笔记目录](paper-notes/README.md) |

---

## 内容结构

```
QuantLab/
│
├── README.md                              ← 你在这里
├── QAT_LEARNING_ROADMAP.md                ← 9 阶段完整路线图
│
├── 📖 docs/                               ← 学习文档（理论 + 源码深潜 + 实验）
│   ├── Stage0_量化基础与硬件基石.md          ✅ 浮点位布局 → 量化公式 → 四种校准器
│   │                                         → VNNI/DP4A/TensorCore 硬件原理
│   │                                         → Observer 继承树 → FakeQuantize 状态机
│   │                                         → QuantStub/DeQuantStub 生命周期
│   │                                         → fuse_modules 内部全流程
│   │
│   ├── Stage1_PyTorch原生QAT三种模式.md      ✅ Eager Mode → FX Graph Mode → PT2E
│   │                                         → prepare_qat_fx 图改写源码分析
│   │                                         → convert_fx 转换管线
│   │                                         → Conv+BN 融合数学推导
│   │                                         → QConfigMapping 优先级系统
│   │
│   ├── Stage2_LSQ与可微量化参数.md           🔜 LSQ 从零实现 + 梯度公式推导
│   ├── Stage3_PTQ进阶算法.md                 📋 AdaRound → FlexRound → GPTQ 数学内核
│   ├── Stage4_YOLO量化实战.md                📋 端到端 PTQ vs QAT vs QAT+LSQ 对比
│   ├── Stage5_工业级框架.md                  📋 PPQ (商汤) + AIMET (高通)
│   ├── Stage6_大模型PTQ.md                   📋 GPTQ/AWQ/SmoothQuant/SpinQuant
│   ├── Stage7_大模型QAT.md                   📋 QLoRA/LLM-QAT/EfficientQAT
│   ├── Stage8_端侧部署全链路.md              📋 ONNX QDQ → TensorRT → QAIRT/DLC
│   └── Stage9_前沿追踪.md                   🔄 FP8/KV Cache/BitNet/MoE
│
├── 🔍 source-notes/                       ← PyTorch 量化模块源码拆解
│   └── observer.py / fake_quantize.py /
│       stubs.py / fuse_modules.py /
│       quantize_fx.py / convert.py ...
│
├── 💻 code/                               ← 独立可运行的实现代码
│   ├── calibrators/                        # MinMax / Percentile / MSE / KL
│   ├── quantizers/                         # 对称 / 非对称量化器
│   ├── lsq/                                # LSQ (Stage 2)
│   ├── adaround/                           # AdaRound (Stage 3)
│   ├── gptq/                               # GPTQ 核心 (Stage 3)
│   ├── yolov8_qat/                         # YOLOv8 QAT 项目 (Stage 4)
│   └── utils/                              # 逐层误差分析等工具
│
└── 📝 paper-notes/                        ← 核心论文阅读笔记
    ├── LSQ (ICLR 2020)
    ├── AdaRound (ICML 2020)
    ├── GPTQ (ICLR 2023)
    ├── AWQ · SmoothQuant · SpinQuant
    └── QLoRA · LLM-QAT · EfficientQAT
```

---

## 9 阶段学习路线

| # | 阶段 | 核心内容 | 难度 |
|---|------|---------|:--:|
| 0 | **量化基础 + 硬件基石** | 浮点位布局、量化公式手推、四种校准器、VNNI/DP4A/TensorCore、PyTorch Observer/FakeQuantize/QuantStub/fuse_modules 源码深潜 | ⭐ |
| 1 | **PyTorch QAT 三种模式** | Eager → FX → PT2E、prepare_qat_fx/convert_fx 图改写源码分析、Conv+BN 融合推导、QConfigMapping | ⭐⭐ |
| 2 | **LSQ 核心算法** | 可学习 scale、手推梯度公式、从零实现 LSQ、LSQ+/PACT/DoReFa-Net | ⭐⭐⭐ |
| 3 | **PTQ 进阶算法** | AdaRound (Taylor→QUBO)、FlexRound (Element-wise Division)、GPTQ (Hessian→Cholesky→Lazy Batch) | ⭐⭐⭐⭐ |
| 4 | **YOLO 量化实战** | YOLOv8/v11 PTQ vs QAT vs QAT+LSQ 对比、检测头敏感度、ONNX QDQ 导出 + TensorRT INT8 推理 | ⭐⭐⭐ |
| 5 | **工业级框架** | PPQ (量化编译器：IR→Pass→Backend)、AIMET (QuantSim + Range Learning QAT → QAIRT/DLC) | ⭐⭐⭐⭐ |
| 6 | **大模型 PTQ 全家桶** | GPTQ/AWQ/SmoothQuant/SpinQuant、Outlier 问题 + 旋转消除 | ⭐⭐⭐⭐ |
| 7 | **大模型 QAT** | QLoRA (NF4+Double Quant)、LLM-QAT (数据蒸馏+KD)、EfficientQAT (两阶段) | ⭐⭐⭐⭐⭐ |
| 8 | **端侧部署全链路** | ONNX QDQ → TensorRT → QAIRT/DLC、逐层误差 Debug、ExecuTorch/llama.cpp GGUF | ⭐⭐⭐⭐ |
| 9 | **前沿追踪** | FP8 (E4M3/E5M2)、KV Cache 量化、BitNet b1.58、MoE 量化、Diffusion 量化 | ⭐⭐⭐⭐⭐ |

---

## 三条学习路径

| 路径 | 时间 | 适合 |
|------|:--:|------|
| 🟢 **快速入门** | 2 周 | Stage 0 + 1：建立量化直觉 + 跑通 PyTorch QAT |
| 🟡 **CV 落地** | 4 周 | + Stage 2~5：LSQ + PTQ 算法 + YOLO 实战 + PPQ/AIMET |
| 🔴 **LLM 全栈** | 8 周 | + Stage 6~9：大模型 PTQ/QAT + 端侧部署 + 前沿追踪 |

---

## 配套框架

学习过程中会深入使用的开源框架：

| 框架 | 用途 | 仓库 |
|------|------|------|
| **PyTorch Quantization** | 原生 QAT/PTQ API + 源码分析 | [pytorch.org](https://pytorch.org/docs/stable/quantization.html) |
| **AIMET** | 高通模型量化工具（QuantSim + Range Learning QAT） | [quic/aimet](https://github.com/quic/aimet) |
| **PPQ** | 商汤量化编译器（IR → 27 Pass → 多后端导出） | [OpenPPL/ppq](https://github.com/OpenPPL/ppq) |
| **AutoGPTQ** | LLM GPTQ 量化（Hessian-based 误差补偿） | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) |
| **bitsandbytes** | QLoRA NF4 + 双重量化 | [bitsandbytes-foundation](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **TensorRT** | NVIDIA INT8/FP8 Tensor Core 推理引擎 | [NVIDIA/TensorRT](https://developer.nvidia.com/tensorrt) |

---

## 我给自己定的原则

- **先问"为什么"，再问"怎么做"**：如果不理解 FakeQuantize 为什么用两个独立 flag 而不是一个 `mode` 枚举，换个场景就懵了。
- **从 YOLO 到 LLaMA，由简到难**：用小模型把 QAT 基本功练到手熟，再上大模型。
- **源码不是"选读"，是"必修"**：observer.py、fake_quantize.py、stubs.py、fuse_modules.py——逐文件拆解。
- **代码和公式互相验证**：每个概念配可运行 Python demo，两条腿走路。
- **对比实验 = 真理**：PTQ vs QAT vs QAT+LSQ、MinMax vs MSE vs KL——不做对比的概念理解是假的。

---

## 当前进度

| 阶段 | 状态 | 完成日 |
|------|:--:|--------|
| Stage 0 · 量化基础 + PyTorch 源码深潜 | ✅ | 2026-07-10 |
| Stage 1 · PyTorch QAT 三种模式 | ✅ | 2026-07-10 |
| Stage 2 · LSQ 核心算法 | 🔜 | — |
| Stage 3 · PTQ 进阶算法 | 📋 | — |
| Stage 4 · YOLO 量化实战 | 📋 | — |
| Stage 5 · PPQ + AIMET | 📋 | — |
| Stage 6 · 大模型 PTQ | 📋 | — |
| Stage 7 · 大模型 QAT | 📋 | — |
| Stage 8 · 端侧部署 | 📋 | — |
| Stage 9 · 前沿追踪 | 🔄 | — |

---

## 📜 版权 & 协议

<p align="center">
  <b>© 2026 <a href="https://github.com/ssG12333">ssG12333</a> — QuantLab · 模型量化深度学习笔记</b><br>
  🔥 持续施工中：从 Stage 0 到 Stage 9，一路吃透<br><br>
  本项目遵循 <a href="LICENSE">MIT License</a> 开源<br>
  欢迎自由学习、分享、二次创作，引用时请注明出处<br><br>
  <b>如果这份笔记帮到了你，请给个 ⭐ Star</b><br>
  你的 Star 是我持续更新的动力 🚀<br><br>
  <a href="https://github.com/ssG12333/QuantLab">
    <img src="https://img.shields.io/github/stars/ssG12333/QuantLab?style=social" alt="GitHub stars">
  </a>
</p>
