# 📖 学习文档

这里放的是每个阶段的学习正文——理论讲解 + 动手实验。每份文档覆盖一个完整的知识模块。

## 文档结构

每个 Stage 文档内部按以下标签组织内容：

- **📖 理论阅读**：数学推导、概念解释、设计动机
- **🔍 源码理解**：PyTorch 对应模块的源码拆解
- **💻 实现代码**：可运行的核心算法 Python 实现
- **🧪 动手实验**：验证理解的实验任务

## 文档列表

| 文件 | 内容 | 状态 |
|------|------|------|
| [Stage0_量化基础与硬件基石.md](./Stage0_量化基础与硬件基石.md) | 量化基础 + 硬件基石：FP32/16/BF16/INT8/FP8位布局 · 量化公式 · 四种校准器 · VNNI/DP4A/TensorCore指令级 · Sub-byte量化 · NF4+DoubleQuant · 误差分析 · 手写推理引擎 | ✅ |
| [Stage1_PyTorch QAT PT2E 深度拆解.md](./Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md) | PT2E 深度拆解：ATen 图捕获 → Quantizer 标注 → FQ 插入四情形 → 四种Spec完整拆解 → tensor级C++内核 → 算子级分派(7类op) → tensor 数据流追踪 → Observer/FakeQuantize 状态机 → Eager/FX/PT2E 对比 | ✅ |
| [Stage1.5_QAT训练深度剖析.md](./Stage1.5_QAT训练深度剖析.md) | 🆕 手写 QAT → 4种Observer×多比特消融 → 比特崩溃点(SNR公式) → BN冻结(running_mean漂移追踪) → 失败诊断手册 → 引出 LSQ | ✅ |
| [Stage2_LSQ与可微量化参数.md](./Stage2_LSQ与可微量化参数.md) | LSQ：从 Stage 1.5 的崩溃点自然引出 → STE 深潜 → 公式推导 → 从零实现 | ✅ |
| Stage3_PTQ进阶算法.md | AdaRound / FlexRound / GPTQ / CLE / Bias Correction | ✅ |
| Stage4_YOLO量化实战.md | YOLOv8 端到端 QAT | 📋 |
