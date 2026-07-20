# QuantLab · 模型量化深度学习笔记

> 从浮点数的 32-bit 位布局，到 LLaMA 的 4-bit QLoRA 微调——把量化从"调参"变成"可控的系统工程"。
> **🔥 持续更新：从底层数学到工业落地，全程吃透。**
>
> *From IEEE 754 floating-point bit layout to LLaMA 4-bit QLoRA fine-tuning — turning model quantization from "parameter tuning" into systematic engineering.*
> ***🔥 Ongoing: from low-level math to production deployment, one bit at a time.***

[![Stages](https://img.shields.io/badge/stages-10-blue)](QAT_LEARNING_ROADMAP.md)
[![Notebooks](https://img.shields.io/badge/notebooks-5篇-green)](notebooks/)
[![Status](https://img.shields.io/badge/status-持续更新_ongoing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

<br>

---

## 这是什么？ / What is this?

一套面向 **模型量化 (Model Quantization)** 的自学笔记，从量化基础理论开始（浮点数怎么存、量化公式怎么推、误差从哪来），一路深入到 PyTorch 框架源码和 LLM 量化部署。适合有深度学习基础、想系统入门模型压缩与优化的同学。这里不讲"调包即用"——每一行代码都有对应的数学推导和框架源码级拆解，目标是让你**真正看懂工具链在做什么**。

*A self-study notebook on model quantization — starting from first principles (IEEE 754, the quantization formula, error analysis) all the way to PyTorch source internals and LLM quantization deployment. For those with DL foundations who want to master model compression & optimization. No "call the API and move on" — every line of code is backed by math and framework source-level breakdown. The goal is to understand what the toolchain actually does.*

我的背景：有深度学习基础，训练过模型、做过部署，但在模型压缩这块一直停留在"知道 INT8 比 FP32 快"的层面。这个仓库就是我把它从头啃透的过程——目标是看到一个量化模型的推理链路时，能说清楚从浮点数到整数、再到硬件指令的每一步在做什么。

*My background: solid DL foundations — trained models, deployed to production. But model compression stayed at the "INT8 is faster than FP32" level. This repo is me tearing it down to first principles. The goal: trace a quantized model's inference path and explain every step — from floating-point bits to integer arithmetic to hardware instructions.*

<br>

---

## 快速导航 / Quick Nav

| 你想... / You want to... | 从这里开始 / Start here |
|---|---|
| 了解全局路线 | [9 阶段路线图](QAT_LEARNING_ROADMAP.md) |
| 建立量化直觉 + 啃 PyTorch 源码 | [📓 Stage 0：量化基础](notebooks/Stage0_量化基础与硬件基石.ipynb) |
| 搞懂 PyTorch QAT 全流程（PT2E 源码级） | [📓 Stage 1：PyTorch QAT 全景](notebooks/Stage1_PyTorch%20QAT%20PT2E%20深度拆解.ipynb) |
| QAT 训练深度剖析（手写+消融+诊断） | [📓 Stage 1.5：QAT 训练深剖](notebooks/Stage1.5_QAT训练深度剖析.ipynb) |
| 学 LSQ 核心算法（autograd 源码级） | [📓 Stage 2：LSQ](notebooks/Stage2_LSQ与可微量化参数.ipynb) |
| 学 PTQ 进阶算法 | [📓 Stage 3：PTQ 进阶](notebooks/Stage3_PTQ进阶算法.ipynb) |
| 查 PyTorch 源码拆解笔记 | [🔍 源码分析](source-notes/README.md) |
| 跑独立可运行代码 | [💻 代码实现](code/README.md) |
| 读论文笔记 | [📝 论文笔记](paper-notes/README.md) |

<br>

---

## 9 阶段学习路线 / 9-Stage Roadmap

| # | 阶段 / Stage | 核心内容 | 难度 |
|---|------|---------|:--:|
| 0 | **量化基础 + 硬件基石**<br>*Quantization Basics* | 浮点位布局(FP32/16/BF16/INT8/FP8) · 量化公式推导 · 四种校准器 · VNNI/DP4A/TensorCore指令级 · Sub-byte量化 · NF4+DoubleQuant · 误差分析 · 手写推理引擎 | ⭐ |
| 1 | **PyTorch 量化全景（PT2E 深度拆解）**<br>*PyTorch Quant Stack* | 五步管线 · ATen 图捕获 · Quantizer 标注 · FQ 插入四种情形 · 算子级分派表 · tensor 数据流追踪 · Observer/FakeQuantize 状态机 · fuse_modules · Eager/FX/PT2E 对比 | ⭐⭐ |
| 1.5 | **QAT 训练深度剖析** 🆕<br>*QAT Deep Dive* | 手写FakeQuantize · Observer消融(4种×多比特) · 比特崩溃点(SNR分析) · BN冻结原理(running_mean漂移) · 失败案例诊断手册 · 引出LSQ | ⭐⭐⭐ |
| 2 | **LSQ — 让 scale 活起来**<br>*Learned Step Size* | autograd.Function生命周期 · STE C++内核 · 公式(6)手推 · Gradient Scaling完整推导 · LSQ+完整实现 · PT2E集成 · PACT→LSQ演进 | ⭐⭐⭐ |
| 3 | **PTQ 进阶算法**<br>*Advanced PTQ* | AdaRound(Taylor→QUBO→Soft Relax) · FlexRound · CLE · Bias Correction · GPTQ(Hessian→Cholesky加速) | ⭐⭐⭐⭐ |
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
│   ├── Stage0_量化基础与硬件基石.ipynb        ✅ 量化基础 + 硬件 + NF4
│   ├── Stage1_PyTorch QAT PT2E 深度拆解.ipynb    ✅ PyTorch 量化全景（PT2E 源码级）
│   ├── Stage1.5_QAT训练深度剖析.ipynb        ✅ 手写→消融→崩溃→引出LSQ
│   ├── Stage2_LSQ与可微量化参数.ipynb        ✅ LSQ + LSQ+ + PT2E 集成
│   ├── Stage3_PTQ进阶算法.ipynb              ✅ AdaRound/FlexRound/CLE/GPTQ
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

- **FP32 → INT8 是心智锚点** *FP32→INT8 is your mental anchor* —— 全文所有例子、公式、代码默认都是 FP32→INT8。其他比特宽度（4-bit、2-bit、FP8）统一从这个基线往下推，而不是当作独立概念。先把"32 位浮点压缩到 8 位整数"刻进脑子，再学其他格式。
- **先问"为什么"，再问"怎么做"** *Ask "why" before "how"* —— 不理解 FakeQuantize 为什么用两个独立 flag，换个场景就懵了。
- **从 YOLO 到 LLaMA，由简到难** *From simple to complex* —— 用小模型把基本功练到手熟，再上大模型。
- **源码不是"选读"，是"必修"** *Source code is mandatory* —— observer.py、fake_quantize.py、stubs.py、fuse_modules.py 逐文件拆解。
- **代码和公式互相验证** *Code verifies math* —— 每个概念配可运行 Python demo。
- **对比实验 = 真理** *Ablation is truth* —— PTQ vs QAT vs LSQ、MinMax vs MSE vs KL。

<br>

---

## 当前进度 / Current Progress

> **🔧 当前阶段：前期内容重构中** —— 搭好骨架之后，逐节"填肉"。
> Stages 0-3 已完成初版，正在按统一标准（tensor 级操作、源码拆解、数学+代码互验）逐节扩充，
> 把每个概念从"能看懂"升级到"能复现每一步"。这也是为什么 Stage 0/1 的 commit 记录还在持续增长。

| 阶段 / Stage | 状态 | 说明 |
|------|:--:|------|
| Stage 0 · 量化基础 | ✅ 已重构 | 基础理论扎实 → 逐节扩充中（§3/§4 已深化） |
| Stage 1 · PyTorch 量化全景 | ✅ 已重构 | PT2E 五步管线 + Observer/Quantizer 已补齐 |
| Stage 1.5 · QAT 训练深度剖析 | ✅ 已重构 | 手写→消融→崩溃→引出LSQ 链条完整 |
| Stage 2 · LSQ 核心算法 | ✅ 已重构 | STE + 公式(6)(13) + LSQ+ 完整实现 |
| Stage 3 · PTQ 进阶算法 | ✅ 已重构 | AdaRound/FlexRound/CLE/GPTQ |
| Stage 4 · YOLO 量化实战 | 📋 待开始 | — |
| Stage 5 · PPQ + AIMET | 📋 待开始 | — |
| Stage 6 · 大模型 PTQ | 📋 待开始 | — |
| Stage 7 · 大模型 QAT | 📋 待开始 | — |
| Stage 8 · 端侧部署 | 📋 待开始 | — |
| Stage 9 · 前沿追踪 | 🔄 持续更新 | — |

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
