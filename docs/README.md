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
| [Stage0_量化基础与硬件基石.md](./Stage0_量化基础与硬件基石.md) | 纯理论地基：IEEE754 · 量化公式 · 四种校准器 · 硬件原理 · 手写推理引擎 | ✅ |
| [Stage1_PyTorch原生QAT三种模式.md](./Stage1_PyTorch原生QAT三种模式.md) | PyTorch 量化全景：Observer/FakeQuantize/Stubs/fuse_modules 源码深潜 + Eager/FX/PT2E 三种 API | ✅ |
| [Stage1.5_QAT训练深度剖析.md](./Stage1.5_QAT训练深度剖析.md) | 🆕 手写 QAT → Observer 消融 → 比特崩溃点 → BN 冻结原理 → 失败案例 → 引出 LSQ | ✅ |
| [Stage2_LSQ与可微量化参数.md](./Stage2_LSQ与可微量化参数.md) | LSQ：从 Stage 1.5 的崩溃点自然引出 → STE 深潜 → 公式推导 → 从零实现 | ✅ |
| Stage3_PTQ进阶算法.md | AdaRound / FlexRound / GPTQ / CLE / Bias Correction | 📋 |
| Stage4_YOLO量化实战.md | YOLOv8 端到端 QAT | 📋 |
