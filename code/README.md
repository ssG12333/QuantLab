# 💻 代码实现

从 Stage 文档中抽取的独立可运行代码，按模块分类。

## 目录规划

```
code/
├── calibrators/          # 四种校准器手写实现 (MinMax/Percentile/MSE/KL)
├── quantizers/           # 从零实现的对称/非对称量化器
├── lsq/                  # LSQ 从零实现 (Stage 2)
├── adaround/             # AdaRound 复现 (Stage 3)
├── gptq/                 # GPTQ 核心逻辑简化实现 (Stage 3)
├── yolov8_qat/           # YOLOv8 QAT 完整项目 (Stage 4)
└── utils/                # 公共工具 (layerwise_error_analysis 等)
```

## 当前状态

代码目前嵌入在各 Stage 文档的"💻 实现代码"小节中。随着文档推进，将逐步抽取到这里，做成独立的可运行模块。
