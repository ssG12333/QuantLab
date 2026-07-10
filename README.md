# QuantLab：我的模型量化深度学习笔记

📖 我为什么要做这个项目

我从 PTQ 的基本概念和 QAT 的大致原理起步，对深度学习基础懂个六七成——大概知道 Sigmoid 查表实现、了解 ONNX 图优化和端侧部署优化。但问题在于：**知识是散的，没有融会贯通。**

市面上有很多量化教程教你怎么调 `prepare_qat_fx` 这个 API，但几乎没有一个告诉你 PyTorch 内部是怎么把 FakeQuantize 插进去的、Observer 的继承树长什么样、`fuse_modules` 到底在模块树上做了什么操作、`QuantStub` 和 `DeQuantStub` 在整个生命周期中经历了哪些变换。

我决定从零开始，把量化这条路从底层数学一直走到大模型部署，**每一站都把源码吃透**，不满足于"能用"，目标是"能改、能设计"。这个仓库就是我的学习笔记——从 YOLO 这种小模型练手，最后过渡到 LLaMA 级别的大模型量化。

当前阶段：**Phase 1** — 先把地基打牢：PyTorch QAT 三种模式 + 底层模块源码深潜。

---

🛠️ 我的学习路线图

Phase 1: 基础夯实 — 量化数学 + PyTorch 源码深潜 [Current]

- 搞懂 FP32/FP16/BF16/INT8 的位布局，手写四种校准器，理解 VNNI/DP4A 怎么加速
- **把 PyTorch 量化模块的源码翻了个遍**：Observer 家族（从 `ObserverBase` 到 `HistogramObserver` 的完整继承链）、FakeQuantize 状态机内核、`QuantStub`/`DeQuantStub` 从出生到 `convert` 的生命周期、`fuse_modules` 如何在模块树上做钩子迁移和数学折叠
- 把 PyTorch 三代 QAT API（Eager / FX / PT2E）都跑通，尤其把 `prepare_qat_fx` 和 `convert_fx` 的图改写逻辑理解到源码级别

Phase 2: QAT 核心算法 — 可微量化参数 [Next]

- **LSQ**：我下一个要攻克的目标——能把 scale 作为可学习参数的梯度公式从零推出来
- LSQ+ / PACT / DoReFa-Net：把可微量化参数这个方向系统地串一遍

Phase 3: PTQ 进阶算法 — 打通 PTQ 和 QAT 的联系 [Planned]

- AdaRound 的 Taylor 展开 → QUBO → Soft Relaxation 推导链
- FlexRound、GPTQ 的 Hessian-based 误差补偿机制
- 理解 Cross-Layer Equalization 为什么对 MobileNet 有用

Phase 4: YOLO 量化实战 [Planned]

- 在 YOLOv8/v11 上跑通 PTQ vs QAT vs QAT+LSQ 的完整对比
- 理解检测头（regression）的量化敏感性
- 导出 ONNX QDQ → TensorRT INT8 推理

Phase 5: 工业级框架 [Planned]

- **PPQ**（商汤）：学它的量化编译器设计——IR → Pass → Backend
- **AIMET**（高通）：学它的 QuantSim + Range Learning QAT → QAIRT/DLC 链路

Phase 6: 大模型 PTQ 全家桶 [Planned]

- GPTQ / AWQ / SmoothQuant / SpinQuant：逐个啃论文 + 复现核心逻辑

Phase 7: 大模型 QAT [Planned]

- QLoRA（NF4 + Double Quantization）、LLM-QAT（数据蒸馏 + KD）、EfficientQAT（两阶段策略）

Phase 8: 端侧部署全链路 [Planned]

- ONNX QDQ → TensorRT → 高通 QAIRT/DLC，画完整数据流图

Phase 9: 持续追踪 [Ongoing]

- FP8、KV Cache 量化、BitNet b1.58、MoE 量化等前沿方向

---

📂 仓库目录

```
QuantLab/
│
├── README.md                              # 你在看的这一页
├── QAT_LEARNING_ROADMAP.md                # 9 阶段完整路线图（全局导航）
│
├── Stage0_量化基础与硬件基石.md            # 浮点数的位布局、量化公式手推、
│   │                                        四种校准器手写、硬件为什么能加速
│   │                                      # 🔍 源码深潜: Observer 继承树、
│   │                                        FakeQuantize 状态机内核、
│   │                                        QuantStub/DeQuantStub 生命周期、
│   │                                        fuse_modules 内部全流程、
│   │                                        prepare_qat/convert 管线
│   │
├── Stage1_PyTorch原生QAT三种模式.md        # Eager/FX/PT2E 三代 API 逐个走、
│   │                                        prepare_qat_fx 图改写源码分析、
│   │                                        convert_fx 源码分析、
│   │                                        Conv+BN 融合数学推导、
│   │                                        QConfigMapping 优先级系统
│   │
├── Stage2_LSQ与可微量化参数.md             # [Next] 从零实现 LSQ
├── Stage3_PTQ进阶算法.md                   # [Planned] AdaRound→GPTQ
├── Stage4_YOLO量化实战.md                  # [Planned]
├── Stage5_工业级框架.md                    # [Planned] PPQ + AIMET
├── Stage6_大模型PTQ.md                     # [Planned]
├── Stage7_大模型QAT.md                     # [Planned]
├── Stage8_端侧部署全链路.md                # [Planned]
└── Stage9_前沿追踪.md                     # [Ongoing]
```

---

🚀 如果你也想跟着学

1. **先看全局**：[QAT_LEARNING_ROADMAP.md](./QAT_LEARNING_ROADMAP.md) 是整个学习路径的导航图，先过一遍知道 9 个阶段的全貌。

2. **按顺序来**：
   ```
   Stage0 → 建立量化直觉 + 吃透 PyTorch 核心模块源码
       ↓
   Stage1 → 理解三种 QAT 模式和底层图改写机制
       ↓
   Stage2 → QAT 核心算法（LSQ）← 我接下来要做的
   ```

3. **每个 Stage 怎么学**：
   - 先读"开篇"建立直觉，再读正文
   - 每个代码块都拷到自己环境里跑一遍
   - 把文档末尾的实验做完
   - 用 Checklist 逐项自检（打勾确认自己真的懂了）

4. **不满足于"能跑通"**：每份文档里都有"源码深潜"章节——我花大量时间把 PyTorch 源码的对应部分啃下来，写成简化版但保留核心逻辑和设计决策。目标是"能改 PyTorch 量化代码"而不只是"能调 API"。

---

📊 当前进度

| 阶段 | 内容 | 状态 | 完成日 |
|------|------|------|--------|
| Stage 0 | 量化数学 + PyTorch 核心模块源码 | ✅ 完成 | 2026-07-10 |
| Stage 1 | PyTorch QAT 三种模式 | ✅ 完成 | 2026-07-10 |
| Stage 2 | LSQ 与可微量化参数 | 🔜 接下来 | — |
| Stage 3 | PTQ 进阶算法 | 📋 计划中 | — |
| Stage 4 | YOLO 量化实战 | 📋 计划中 | — |
| Stage 5 | PPQ + AIMET | 📋 计划中 | — |
| Stage 6 | 大模型 PTQ | 📋 计划中 | — |
| Stage 7 | 大模型 QAT | 📋 计划中 | — |
| Stage 8 | 端侧部署 | 📋 计划中 | — |
| Stage 9 | 前沿追踪 | 🔄 持续 | — |

---

⚠️ 我给自己定的几条原则

- **先问"为什么"，再问"怎么做"**：如果不理解 FakeQuantize 为什么要用两个独立 flag 而不是一个 `mode` 枚举，那就算代码能跑，换个场景就懵了。每篇文档在给出代码之前，先把设计动机讲清楚。
- **从 YOLO 到 LLaMA，由简到难**：不要上来就搞大模型量化。先用小模型把 QAT 的训练循环、observer 调度、BN 冻结这些基本功练到手熟。
- **源码不是"选读"，是"必修"**：Stage 0 里我把 PyTorch 的 `observer.py`、`fake_quantize.py`、`stubs.py`、`fuse_modules.py` 的源码逻辑都拆解了一遍。理解源码是从"API 调用者"到"框架级工程师"的必经之路。
- **代码和公式互相验证**：每个核心概念都有可运行的 Python demo。看完公式马上跑代码，跑完代码回头验证公式——两条腿走路。
- **对比实验 = 真理**：PTQ vs QAT vs QAT+LSQ、MinMax vs MSE vs KL——不做对比实验的概念理解是假的。

---

🔗 我的主要参考源

- 📖 必读论文：[AdaRound (ICML 2020)](https://arxiv.org/abs/2004.10568) · [LSQ (ICLR 2020)](https://arxiv.org/abs/1902.08153) · [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323) · [SmoothQuant (ICML 2023)](https://arxiv.org/abs/2211.10438) · [AWQ (MLSys 2024)](https://arxiv.org/abs/2306.00978) · [SpinQuant (ICLR 2025)](https://arxiv.org/abs/2405.16406) · [QLoRA (NeurIPS 2023)](https://arxiv.org/abs/2305.14314)
- 🛠 框架：[PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) · [AIMET (Qualcomm)](https://github.com/quic/aimet) · [PPQ (OpenPPL)](https://github.com/OpenPPL/ppq) · [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) · [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- 🎓 课程：[MIT 6.S191: TinyML & Efficient DL](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [MIT 6.5940: EfficientML](https://efficientml.ai/)
- 📊 追踪：[HuggingFace Daily Papers](https://huggingface.co/papers) · [Awesome Low-Precision Training](https://github.com/Hao840/Awesome-Low-Precision-Training)

---

> 🤖 [ssG12333](https://github.com/ssG12333) 的学习笔记，持续施工中 🚧
