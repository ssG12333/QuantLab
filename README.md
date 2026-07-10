# QuantLab · 模型量化深度学习笔记

> 从浮点数的 32-bit 位布局，到 LLaMA 的 4-bit QLoRA 微调——把量化从"调参"变成"可控的系统工程"。
> **🔥 持续更新：从底层数学到工业落地，全程吃透。**
>
> *From IEEE 754 floating-point bit layout to LLaMA 4-bit QLoRA fine-tuning — turning model quantization from "parameter tuning" into systematic engineering.*
> ***🔥 Ongoing: from low-level math to production deployment, one bit at a time.***

[![Stages](https://img.shields.io/badge/stages-9-blue)](QAT_LEARNING_ROADMAP.md)
[![Docs](https://img.shields.io/badge/docs-2篇_articles-green)](docs/)
[![Status](https://img.shields.io/badge/status-持续更新_ongoing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 这是什么？ / What is this?

一套面向 **模型量化 (Model Quantization)** 的自学笔记，适合已经知道 PTQ/QAT 大致原理、但**不满足于"能调 API"**、想下沉到框架源码级别理解的人。

*A self-study notebook on model quantization, for those who already know the basics of PTQ/QAT but want to go deeper — reading framework source code, not just calling APIs.*

我自己的背景：对深度学习基础懂个六七成（Sigmoid 查表、ONNX 图优化、端侧部署），但知识比较散。这个仓库是我**从 YOLO 练手到 LLaMA 量化**的完整学习记录。

*My background: ~70% solid on DL fundamentals (sigmoid LUT, ONNX graph optimization, edge deployment), but knowledge was fragmented. This repo is my complete learning trail — from YOLO practice to LLaMA quantization.*

---

## 快速导航 / Quick Nav

| 你想... / You want to... | 从这里开始 / Start here |
|---|---|
| 了解全局路线 / See the big picture | [9 阶段学习路线图 / Roadmap](QAT_LEARNING_ROADMAP.md) |
| 建立量化直觉 + 啃 PyTorch 源码 / Build intuition + read source | [📖 Stage 0：量化基础与硬件基石](docs/Stage0_量化基础与硬件基石.md) |
| 搞懂 PyTorch QAT 三种模式 / Master PyTorch QAT APIs | [📖 Stage 1：PyTorch 原生 QAT](docs/Stage1_PyTorch原生QAT三种模式.md) |
| 查已拆解的 PyTorch 模块 / Browse source analysis | [🔍 源码分析索引 / Source Notes](source-notes/README.md) |
| 跑独立可运行代码 / Run standalone code | [💻 代码实现 / Code](code/README.md) |
| 读论文笔记 / Read paper notes | [📝 论文笔记 / Paper Notes](paper-notes/README.md) |

---

## 内容结构 / Structure

```
QuantLab/
│
├── README.md                              ← 你在这里 / You are here
├── QAT_LEARNING_ROADMAP.md                ← 9 阶段完整路线图 / Full roadmap
│
├── 📖 docs/                               ← 学习文档 / Learning docs
│   │                                      （理论 + 源码深潜 + 实验）
│   │                                      (Theory + Source dive + Experiments)
│   ├── Stage0_量化基础与硬件基石.md          ✅ 浮点位布局 → 量化公式 → 四种校准器
│   │                                         → VNNI/DP4A/TensorCore
│   │                                         → Observer → FakeQuantize 状态机
│   │                                         → QuantStub/DeQuantStub 生命周期
│   │                                         → fuse_modules 全流程
│   │
│   ├── Stage1_PyTorch原生QAT三种模式.md      ✅ Eager → FX → PT2E
│   │                                         → prepare_qat_fx 图改写源码分析
│   │                                         → convert_fx 转换管线
│   │                                         → Conv+BN 融合数学推导
│   │
│   ├── Stage2_LSQ与可微量化参数.md           🔜 LSQ 从零实现 + 梯度推导
│   ├── Stage3_PTQ进阶算法.md                 📋 AdaRound → FlexRound → GPTQ
│   ├── Stage4_YOLO量化实战.md                📋 端到端 PTQ vs QAT vs QAT+LSQ
│   ├── Stage5_工业级框架.md                  📋 PPQ (商汤) + AIMET (高通)
│   ├── Stage6_大模型PTQ.md                   📋 GPTQ/AWQ/SmoothQuant/SpinQuant
│   ├── Stage7_大模型QAT.md                   📋 QLoRA/LLM-QAT/EfficientQAT
│   ├── Stage8_端侧部署全链路.md              📋 ONNX QDQ → TensorRT → QAIRT/DLC
│   └── Stage9_前沿追踪.md                   🔄 FP8/KV Cache/BitNet/MoE
│
├── 🔍 source-notes/                       ← PyTorch 源码拆解
│   └── observer.py / fake_quantize.py /
│       stubs.py / fuse_modules.py /
│       quantize_fx.py / convert.py ...
│
├── 💻 code/                               ← 独立可运行代码 / Standalone code
│   ├── calibrators/                        # MinMax / Percentile / MSE / KL
│   ├── quantizers/                         # Symmetric / Asymmetric
│   ├── lsq/                                # LSQ (Stage 2)
│   ├── adaround/                           # AdaRound (Stage 3)
│   ├── gptq/                               # GPTQ core (Stage 3)
│   ├── yolov8_qat/                         # YOLOv8 QAT project (Stage 4)
│   └── utils/                              # Layer-wise error analysis etc.
│
└── 📝 paper-notes/                        ← 论文笔记 / Paper reading notes
    ├── LSQ (ICLR 2020)
    ├── AdaRound (ICML 2020)
    ├── GPTQ (ICLR 2023)
    ├── AWQ · SmoothQuant · SpinQuant
    └── QLoRA · LLM-QAT · EfficientQAT
```

---

## 9 阶段学习路线 / 9-Stage Roadmap

| # | 阶段 / Stage | 核心内容 / Core | 难度 |
|---|------|---------|:--:|
| 0 | **量化基础 + 硬件**<br>*Quantization Basics* | 浮点位布局、量化公式推导、四种校准器手写、VNNI/DP4A/TensorCore、PyTorch Observer/FakeQuantize/Stubs/fuse_modules 源码深潜 | ⭐ |
| 1 | **PyTorch QAT 三种模式**<br>*PyTorch QAT APIs* | Eager → FX → PT2E、prepare_qat_fx/convert_fx 图改写源码、Conv+BN 融合推导 | ⭐⭐ |
| 2 | **LSQ 核心算法**<br>*Learned Step Size* | 可学习 scale、手推梯度公式、从零实现、LSQ+/PACT/DoReFa | ⭐⭐⭐ |
| 3 | **PTQ 进阶算法**<br>*Advanced PTQ* | AdaRound (Taylor→QUBO)、FlexRound、GPTQ (Hessian→Cholesky) | ⭐⭐⭐⭐ |
| 4 | **YOLO 量化实战**<br>*YOLO Quantization* | YOLOv8 PTQ vs QAT vs LSQ 消融实验、ONNX QDQ + TensorRT INT8 | ⭐⭐⭐ |
| 5 | **工业框架**<br>*Industrial Frameworks* | PPQ (IR→Pass→Backend)、AIMET (QuantSim + Range Learning QAT → DLC) | ⭐⭐⭐⭐ |
| 6 | **LLM PTQ 全家桶**<br>*LLM PTQ Stack* | GPTQ/AWQ/SmoothQuant/SpinQuant、Outlier 问题 + 旋转消除 | ⭐⭐⭐⭐ |
| 7 | **LLM QAT**<br>*LLM QAT* | QLoRA (NF4+Double Quant)、LLM-QAT (KD)、EfficientQAT (两阶段) | ⭐⭐⭐⭐⭐ |
| 8 | **端侧部署**<br>*Edge Deployment* | ONNX QDQ → TensorRT → QAIRT/DLC、逐层误差 Debug、GGUF | ⭐⭐⭐⭐ |
| 9 | **前沿追踪**<br>*Frontiers* | FP8 (E4M3/E5M2)、KV Cache 量化、BitNet b1.58、MoE 量化 | ⭐⭐⭐⭐⭐ |

---

## 三条学习路径 / Three Learning Paths

| 路径 / Path | 时间 | 适合 / For |
|------|:--:|------|
| 🟢 **快速入门**<br>*Quick Start* | 2 周 | 想快速建立量化直觉 + 跑通 PyTorch QAT<br>*Build intuition + run PyTorch QAT* |
| 🟡 **CV 落地**<br>*CV Production* | 4 周 | + LSQ + PTQ 算法 + YOLO 实战 + PPQ/AIMET<br>*+ Algorithm deep-dive + YOLO + industrial frameworks* |
| 🔴 **LLM 全栈**<br>*LLM Full Stack* | 8 周 | + 大模型 PTQ/QAT + 端侧部署 + 前沿追踪<br>*+ LLM quantization + edge deployment + frontiers* |

---

## 配套框架 / Companion Frameworks

| 框架 / Framework | 用途 / Purpose | 仓库 / Repo |
|------|------|------|
| **PyTorch Quantization** | 原生 QAT/PTQ API + 源码分析 | [pytorch.org](https://pytorch.org/docs/stable/quantization.html) |
| **AIMET** | 高通量化工具 (QuantSim + Range Learning QAT) | [quic/aimet](https://github.com/quic/aimet) |
| **PPQ** | 商汤量化编译器 (IR → 27 Pass → 多后端) | [OpenPPL/ppq](https://github.com/OpenPPL/ppq) |
| **AutoGPTQ** | LLM GPTQ 量化 (Hessian-based) | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) |
| **bitsandbytes** | QLoRA NF4 + 双重量化 | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **TensorRT** | NVIDIA INT8/FP8 推理引擎 | [NVIDIA/TensorRT](https://developer.nvidia.com/tensorrt) |

---

## 学习原则 / Principles

- **先问"为什么"，再问"怎么做" / *Ask "why" before "how"***：不理解 FakeQuantize 为什么用两个独立 flag，换个场景就懵了。
- **从 YOLO 到 LLaMA，由简到难 / *From simple to complex***：用小模型把基本功练到手熟，再上大模型。
- **源码不是"选读"，是"必修" / *Source code is mandatory***：observer.py、fake_quantize.py、stubs.py、fuse_modules.py——逐文件拆解。
- **代码和公式互相验证 / *Code verifies math***：每个概念配可运行 Python demo。
- **对比实验 = 真理 / *Ablation is truth***：PTQ vs QAT vs LSQ、MinMax vs MSE vs KL。

---

## 当前进度 / Current Progress

| 阶段 / Stage | 状态 / Status | 完成日 / Date |
|------|:--:|--------|
| Stage 0 · 量化基础 + PyTorch 源码 | ✅ 完成 / Done | 2026-07-10 |
| Stage 1 · PyTorch QAT 三种模式 | ✅ 完成 / Done | 2026-07-10 |
| Stage 2 · LSQ 核心算法 | 🔜 进行中 / Next | — |
| Stage 3 · PTQ 进阶算法 | 📋 计划 / Planned | — |
| Stage 4 · YOLO 量化实战 | 📋 计划 / Planned | — |
| Stage 5 · PPQ + AIMET | 📋 计划 / Planned | — |
| Stage 6 · 大模型 PTQ | 📋 计划 / Planned | — |
| Stage 7 · 大模型 QAT | 📋 计划 / Planned | — |
| Stage 8 · 端侧部署 | 📋 计划 / Planned | — |
| Stage 9 · 前沿追踪 | 🔄 持续 / Ongoing | — |

---

## 📜 License

<p align="center">
  <b>© 2026 <a href="https://github.com/ssG12333">ssG12333</a> — QuantLab · 模型量化深度学习笔记</b><br>
  <i>Model Quantization Deep Learning Notebook</i><br><br>
  🔥 持续施工中：从 Stage 0 到 Stage 9，一路吃透<br>
  <i>Under active development — from Stage 0 to Stage 9, mastering every step.</i><br><br>
  MIT License · 欢迎自由学习、分享、二次创作<br>
  <i>Free to learn, share, and remix — attribution appreciated.</i><br><br>
  <b>⭐ 如果帮到了你，给个 Star 就是最大的鼓励 🚀</b><br>
  <i>If this helps you, a ⭐ Star means the world to me.</i><br><br>
  <a href="https://github.com/ssG12333/QuantLab">
    <img src="https://img.shields.io/github/stars/ssG12333/QuantLab?style=social" alt="GitHub stars">
  </a>
</p>
