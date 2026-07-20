# 🔍 PyTorch 源码分析笔记

这里放的是 PyTorch 量化模块的源码拆解——把官方仓库中对应文件的逻辑抽取出来，用简化的代码保留核心设计决策。**目标是理解"PyTorch 为什么这样设计"，而不只是知道 API 怎么调。**

## 已拆解的模块

| 模块 | PyTorch 源文件 | 笔记位置 |
|------|---------------|----------|
| Observer 家族 | `torch/ao/quantization/observer.py` | 已整合在 [Stage0](../docs/Stage0_量化基础与硬件基石.md) §8 |
| FakeQuantize 内核 | `torch/ao/quantization/fake_quantize.py` | 已整合在 [Stage0](../docs/Stage0_量化基础与硬件基石.md) §9 和 [Stage1](../docs/Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md) §1 |
| QuantStub / DeQuantStub | `torch/ao/quantization/stubs.py` | 已整合在 [Stage0](../docs/Stage0_量化基础与硬件基石.md) §10 |
| fuse_modules 内部流程 | `torch/ao/quantization/fuse_modules.py` | 已整合在 [Stage0](../docs/Stage0_量化基础与硬件基石.md) §11 |
| prepare_qat_fx 图改写 | `torch/ao/quantization/quantize_fx.py` | 已整合在 [Stage1](../docs/Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md) §5 |
| convert_fx 转换管线 | `torch/ao/quantization/fx/convert.py` | 已整合在 [Stage1](../docs/Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md) §6 |
| QConfigMapping 匹配系统 | `torch/ao/quantization/qconfig.py` | 已整合在 [Stage1](../docs/Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md) §8 |

## 待拆解的模块

- `torch/ao/quantization/pt2e/` — PT2E QAT 的完整管线
- `torch/ao/quantization/fx/prepare.py` — FX prepare 的详细实现
- `torch/ao/quantization/backend_config/` — BackendConfig 的内部结构
- `torch/ao/quantization/fx/fusion_patterns.py` — 融合模式匹配引擎
