# QuantLab · 模型量化深度学习笔记

> 从浮点数的 32-bit 位布局，到 LLaMA 的 4-bit QLoRA 微调——把量化从"调参"变成"可控的系统工程"。
> **🔥 持续更新：从底层数学到工业落地，全程吃透。**
>
> *From IEEE 754 floating-point bit layout to LLaMA 4-bit QLoRA fine-tuning — turning model quantization from "parameter tuning" into systematic engineering.*
> ***🔥 Ongoing: from low-level math to production deployment, one bit at a time.***

[![Stages](https://img.shields.io/badge/stages-10-blue)](QAT_LEARNING_ROADMAP.md)
[![Notebooks](https://img.shields.io/badge/notebooks-4篇-green)](notebooks/)
[![Status](https://img.shields.io/badge/status-持续更新_ongoing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

<br>

---

## 这是什么？ / What is this?

一套面向 **模型量化 (Model Quantization)** 的自学笔记，适合已经知道 PTQ/QAT 大致原理、但**不满足于"能调 API"**、想下沉到框架源码级别理解的人。

*A self-study notebook on model quantization, for those who already know the basics of PTQ/QAT but want to go deeper — reading framework source code, not just calling APIs.*

我自己的背景：对深度学习基础懂个六七成（Sigmoid 查表、ONNX 图优化、端侧部署），但知识比较散。这个仓库是我**从 YOLO 练手到 LLaMA 量化**的完整学习记录。

*My background: ~70% solid on DL fundamentals (sigmoid LUT, ONNX graph optimization, edge deployment), but knowledge was fragmented. This repo is my complete learning trail — from YOLO practice to LLaMA quantization.*

<br>

---

## 快速导航 / Quick Nav

| 你想... / You want to... | 从这里开始 / Start here |
|---|---|
| 了解全局路线 | [9 阶段路线图](QAT_LEARNING_ROADMAP.md) |
| 建立量化直觉 + 啃 PyTorch 源码 | [📓 Stage 0：量化基础](notebooks/Stage0_量化基础与硬件基石.ipynb) |
| 搞懂 PyTorch QAT 三种模式 | [📓 Stage 1：PyTorch QAT 全景](notebooks/Stage1_PyTorch原生QAT三种模式.ipynb) |
| QAT 训练深度剖析 | [📓 Stage 1.5：QAT 训练深剖](notebooks/Stage1.5_QAT训练深度剖析.ipynb) |
| 学 LSQ 核心算法 | [📓 Stage 2：LSQ](notebooks/Stage2_LSQ与可微量化参数.ipynb) |
| 查 PyTorch 源码拆解笔记 | [🔍 源码分析](source-notes/README.md) |
| 跑独立可运行代码 | [💻 代码实现](code/README.md) |
| 读论文笔记 | [📝 论文笔记](paper-notes/README.md) |

<br>

---

## 9 阶段学习路线 / 9-Stage Roadmap

| # | 阶段 / Stage | 核心内容 | 难度 |
|---|------|---------|:--:|
| 0 | **量化基础 + 硬件基石**<br>*Quantization Basics* | 纯理论地基：浮点位布局 · 量化公式推导 · 四种校准器 · VNNI/DP4A/TensorCore · 手写推理引擎 | ⭐ |
| 1 | **PyTorch 量化全景**<br>*PyTorch Quant Stack* | Observer继承树 · FakeQuantize状态机 · Stubs生命周期 · fuse_modules流程 · Eager/FX/PT2E · 图改写源码 | ⭐⭐ |
| 1.5 | **QAT 训练深度剖析** 🆕<br>*QAT Deep Dive* | 手写FakeQuantize · Observer消融 · 比特崩溃点 · BN冻结原理 · 失败案例诊断 · 引出LSQ | ⭐⭐⭐ |
| 2 | **LSQ — 让 scale 活起来**<br>*Learned Step Size* | 从崩溃点自然引出 · STE深潜 · 公式(6)推导 · Gradient Scaling · 从零实现 · PyTorch对比 | ⭐⭐⭐ |
| 3 | **PTQ 进阶算法**<br>*Advanced PTQ* | AdaRound (Taylor→QUBO) · FlexRound · GPTQ (Hessian→Cholesky) | ⭐⭐⭐⭐ |
| 4 | **YOLO 量化实战**<br>*YOLO Quantization* | YOLOv8 PTQ vs QAT vs LSQ 消融实验 · ONNX QDQ · TensorRT INT8 | ⭐⭐⭐ |
| 5 | **工业框架**<br>*Industrial Frameworks* | PPQ (IR→Pass→Backend) · AIMET (QuantSim + Range Learning QAT → DLC) | ⭐⭐⭐⭐ |
| 6 | **LLM PTQ 全家桶**<br>*LLM PTQ Stack* | GPTQ/AWQ/SmoothQuant/SpinQuant · Outlier 问题 + 旋转消除 | ⭐⭐⭐⭐ |
| 7 | **LLM QAT**<br>*LLM QAT* | QLoRA (NF4+Double Quant) · LLM-QAT (KD) · EfficientQAT (两阶段) | ⭐⭐⭐⭐⭐ |
| 8 | **端侧部署**<br>*Edge Deployment* | ONNX QDQ → TensorRT → QAIRT/DLC · 逐层误差 Debug · GGUF | ⭐⭐⭐⭐ |
| 9 | **前沿追踪**<br>*Frontiers* | FP8 (E4M3/E5M2) · KV Cache 量化 · BitNet b1.58 · MoE 量化 | ⭐⭐⭐⭐⭐ |

<br>

---

## 三条学习路径 / Three Paths

| 路径 | 时间 | 适合 |
|------|:--:|------|
| 🟢 **快速入门** *Quick Start* | 2 周 | 建立量化直觉 + 跑通 PyTorch QAT |
| 🟡 **CV 落地** *CV Production* | 4 周 | + LSQ 算法 + YOLO 实战 + PPQ/AIMET |
| 🔴 **LLM 全栈** *LLM Full Stack* | 8 周 | + 大模型 PTQ/QAT + 端侧部署 + 前沿追踪 |

<br>

---

## 内容结构 / Structure

```
QuantLab/
│
├── README.md                              ← 你在这里 / You are here
├── QAT_LEARNING_ROADMAP.md                ← 9 阶段完整路线图
│
├── 📓 notebooks/                           ← Jupyter Notebook 学习文档
│   ├── Stage0_量化基础与硬件基石.ipynb        ✅ 纯理论地基
│   ├── Stage1_PyTorch原生QAT三种模式.ipynb    ✅ PyTorch 量化全景
│   ├── Stage1.5_QAT训练深度剖析.ipynb        ✅ 手写→消融→崩溃→引出LSQ
│   ├── Stage2_LSQ与可微量化参数.ipynb        ✅ LSQ 自然延伸
│   ├── Stage3_PTQ进阶算法.ipynb              📋 AdaRound → FlexRound → GPTQ
│   ├── Stage4_YOLO量化实战.ipynb             📋 PTQ vs QAT vs LSQ 消融实验
│   ├── Stage5_工业级框架.ipynb               📋 PPQ + AIMET
│   ├── Stage6_大模型PTQ.ipynb                📋 GPTQ/AWQ/SmoothQuant/SpinQuant
│   ├── Stage7_大模型QAT.ipynb                📋 QLoRA/LLM-QAT/EfficientQAT
│   ├── Stage8_端侧部署全链路.ipynb           📋 ONNX → TensorRT → QAIRT/DLC
│   └── Stage9_前沿追踪.ipynb                🔄 FP8/KV Cache/BitNet/MoE
│
├── 🔍 source-notes/                       ← PyTorch 源码拆解笔记
├── 💻 code/                               ← 独立可运行代码
└── 📝 paper-notes/                        ← 核心论文阅读笔记
```

<br>

---

## 配套框架 / Companion Frameworks

| 框架 | 用途 | 仓库 |
|------|------|------|
| **PyTorch Quantization** | 原生 QAT/PTQ API + 源码分析 | [pytorch.org](https://pytorch.org/docs/stable/quantization.html) |
| **AIMET** | 高通量化 (QuantSim + Range Learning QAT) | [quic/aimet](https://github.com/quic/aimet) |
| **PPQ** | 商汤量化编译器 (IR → 27 Pass → 多后端) | [OpenPPL/ppq](https://github.com/OpenPPL/ppq) |
| **AutoGPTQ** | LLM GPTQ 量化 (Hessian-based) | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) |
| **bitsandbytes** | QLoRA NF4 + 双重量化 | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **TensorRT** | NVIDIA INT8/FP8 推理引擎 | [NVIDIA/TensorRT](https://developer.nvidia.com/tensorrt) |

<br>

---

## 学习原则 / Principles

- **先问"为什么"，再问"怎么做"** *Ask "why" before "how"* —— 不理解 FakeQuantize 为什么用两个独立 flag，换个场景就懵了。
- **从 YOLO 到 LLaMA，由简到难** *From simple to complex* —— 用小模型把基本功练到手熟，再上大模型。
- **源码不是"选读"，是"必修"** *Source code is mandatory* —— observer.py、fake_quantize.py、stubs.py、fuse_modules.py 逐文件拆解。
- **代码和公式互相验证** *Code verifies math* —— 每个概念配可运行 Python demo。
- **对比实验 = 真理** *Ablation is truth* —— PTQ vs QAT vs LSQ、MinMax vs MSE vs KL。

<br>

---

## 当前进度 / Current Progress

| 阶段 / Stage | 状态 | 完成日 |
|------|:--:|--------|
| Stage 0 · 量化基础（纯理论） | ✅ | 2026-07-10 |
| Stage 1 · PyTorch 量化全景 | ✅ | 2026-07-13 |
| Stage 1.5 · QAT 训练深度剖析 | ✅ | 2026-07-13 |
| Stage 2 · LSQ 核心算法 | ✅ | 2026-07-13 |
| Stage 3 · PTQ 进阶算法 | 📋 | — |
| Stage 4 · YOLO 量化实战 | 📋 | — |
| Stage 5 · PPQ + AIMET | 📋 | — |
| Stage 6 · 大模型 PTQ | 📋 | — |
| Stage 7 · 大模型 QAT | 📋 | — |
| Stage 8 · 端侧部署 | 📋 | — |
| Stage 9 · 前沿追踪 | 🔄 | — |

<br>

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
    <img src="https://img.shields.io/github/stars/ssG12333/QuantLab?style=social" alt="stars">
  </a>
</p>
