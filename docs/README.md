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
| [Stage0_量化基础与硬件基石.md](./Stage0_量化基础与硬件基石.md) | 浮点数位布局、量化公式手推、四种校准器、硬件加速原理 + PyTorch Observer/FakeQuantize/QuantStub/fuse_modules 源码深潜 | ✅ |
| [Stage1_PyTorch原生QAT三种模式.md](./Stage1_PyTorch原生QAT三种模式.md) | Eager/FX/PT2E 三代 API + prepare_qat_fx/convert_fx 图改写源码分析 | ✅ |
| Stage2_LSQ与可微量化参数.md | LSQ 从零实现 + 梯度公式推导 | 🔜 |
