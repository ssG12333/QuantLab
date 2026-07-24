# FAQ · 问题速查

> 在学习过程中向 Claude 问过的问题集合。每个问题都附了简短答案和对应学习章节的交叉引用。
> 这些问题不写在 Stage 正文里——正文是体系化的，这里是**碰到的坑、闪过的疑问、断点处的追问**。

---

## 基础概念

### Q1：量化公式里那个 "0" 到底是什么？

**答**：`q = round(r/S + Z)` 里的 Z 就是 zero_point——**INT8 整数 0 对应实数域的哪个位置**。对称量化 Z=0，INT8 的 0 = 实数 0，网格以实数 0 为中心对称。非对称量化 Z≠0，INT8 的 0 ≠ 实数 0，网格"偏移"以对齐数据的 min。

→ 详见 [Stage 0 §3.2.1](Stage0_量化基础与硬件基石.md)

---

### Q2：对称 [0,127] vs 非对称 [0,255] 到底差在哪？

**答**：步长差一倍。同区间 [0, 3.0]：对称 S=3.0/127≈0.02362（128 个刻度），非对称 S=3.0/255≈0.01176（256 个刻度）。S 差一倍 → S² 差四倍 → **MSE 差四倍**。本质：非对称把全部 256 个值压到有效区间，不浪费任何刻度在数据永远不会出现的负半轴。

→ 详见 [Stage 0 §3.2.2](Stage0_量化基础与硬件基石.md)

---

### Q3：scale 是"选大的"吗？

**答**：不是选，是被数据范围**决定**的。`S = r_max / 127`。数据范围大 → S 必须大，否则大值被 clamp 截断。但 S 大意味着刻度粗。Per-Tensor 的问题就是 1 个 scale 被 outlier 通道"绑架"→ 小通道被碾碎。Per-Channel 每个通道用自己的 r_max 算自己的 S——大范围用大 S，小范围用小 S。

→ 详见 [Stage 0 §4.1-4.2](Stage0_量化基础与硬件基石.md)

---

### Q4：Per-Tensor、Per-Channel、Per-Group 的大小关系？

**答**：
- 每个 scale 管的值数量：Per-Tensor > Per-Channel > Per-Group
- scale 的数量：Per-Tensor < Per-Channel < Per-Group
- 精度：Per-Tensor < Per-Channel < Per-Group

"粒度细"说的是每个 scale 管的值少，不是 scale 数量多。

→ 详见 [Stage 0 §4.0](Stage0_量化基础与硬件基石.md)

---

### Q5：`axis` 是什么？

**答**：在 `quantize_per_channel` 里，`axis` = scale 沿 tensor 的哪一维变化。对 Conv2d weight `[64,3,3,3]`，axis=0 表示 64 个 scale 沿 out_ch 维——通道 0 的 27 个值共用 S_0。ONNX QuantizeLinear 也有 axis 属性，含义一样。

→ 详见 [Stage 0 §4.0](Stage0_量化基础与硬件基石.md)（axis 提前认领框）

---

### Q6：`unsqueeze` 的广播在广播什么？

**答**：`quantize_per_channel` 里 `[64,27] / [64,1]` —— `[64,1]` 沿 dim=1 自动"铺开"27 次变成 `[64,27]`。本质是 PyTorch 帮你把每通道的 1 个 scale 复制到该通道的 27 个权重上，省了一个 for 循环。

→ 详见 [Stage 0 §4.2](Stage0_量化基础与硬件基石.md)

---

### Q7：权重量化和激活量化是一回事吗？

**答**：不是。一个 Conv 有**两条独立量化线**——权重走对称 Per-Channel，激活走非对称 Per-Tensor。各用各的 scale、各用各的 zero_point、各用各的粒度。唯一共享的：公式都是 `q = round(r/S + Z)`。

→ 详见 [Stage 0 §3.4](Stage0_量化基础与硬件基石.md)

---

### Q8：为什么训练时要把 FP32 → INT8 当成心智锚点？

**答**：INT8 是工业默认起点——256 个等级、对称 [-128,127]、非对称 [0,255]。所有其他比特宽度（4-bit、2-bit、FP8）都是从这个基线往下推。每个例子里的 `/127`、`clamp(-128,127)` 都是 INT8 的具体体现，不要把它读成"某通用量化操作"。

→ 详见 [Stage 0 §2 心智锚点声明](Stage0_量化基础与硬件基石.md)

---

## PTQ 与校准

### Q9：四种校准器就是 PTQ 的校准方法吗？

**答**：对。这四种（MinMax / Percentile / MSE / KL Divergence）就是 PTQ 的标准校准管线：

```
校准数据(无标签) → Forward收集激活分布 → 校准器算S,Z → 量化权重+激活 → 完事
```

校准只针对**激活**（权重的值训练完就固定了）。PTQ 的 S 算完就冻，QAT 的 S 算完还要训练。

→ 详见 [Stage 0 §5.5.1](Stage0_量化基础与硬件基石.md)

---

## QAT / PT2E / 量化规则

### Q10：PT2E 的 QAT 不对 bias 做量化吗？

**答**：对，bias 保持 FP32/INT32。三个原因：① bias 只有每通道 1 个值，量化收益 ≈ 0；② INT8×INT8 累加到 INT32，bias 保持 INT32 零额外开销；③ bias 的 scale 是 Derived 的（S_bias = S_w × S_a），不需要 Observer/FQ。

→ 详见 [Stage 1 §4](Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md)

---

### Q11：在 Torch 层面能改 Quantizer 对 bias 做量化吗？

**答**：能。把 bias 的 `DerivedQuantizationSpec` 改成普通 `QuantizationSpec(dtype=torch.int8)`，走完整 Observer→FQ 管线。但没有硬件支持（INT8 bias 加进 INT32 累加器反而多一步反量化），所以没人做。

---

### Q12：我有自己的量化规则，能拿预训练模型自动插 QDQ 吗？

**答**：能，这就是 PT2E 的核心设计目的。把你的规则写成 Quantizer（一张"什么算子插什么 QDQ"的表）→ `torch.export.export` 抓图 → `quantizer.annotate()` 标注 → `prepare_pt2e` 自动插 QDQ → `convert_pt2e` 固化。只要你的规则合理（粒度、对称方式、跳过哪些算子），QDQ 插入就是机械操作。

→ 详见 [Stage 1 §11](Stage1_PyTorch%20QAT%20PT2E%20深度拆解.md)

---

## 硬件 & 推理底层

### Q13：INT8×INT8→INT32→INT8 这个"中间态"是干嘛的？

**答**：这是 requantize——量化推理的心跳。INT8×INT8 乘加进 INT32 累加器（不溢出），然后 `acc × (S_a×S_w/S_out)` 把 INT32 "压"回 INT8 给下一层。没有 requantize，每层输出都是 INT32 → 必须转 FP32 → 下一层再转 INT8 → 来回转换吃掉所有加速。

→ 详见 [Stage 0 §7.5](Stage0_量化基础与硬件基石.md)

---

### Q14：Sigmoid/Tanh 这些非线性算子没有 INT8 指令，怎么量化推理？

**答**：两种方案。方案 A——FP32 fallback（Dequantize→FP32 算→Quantize 回 INT8，简单但慢）。方案 B——**LUT 查表**（预计算 256 种输入对应 256 种输出，推理时 O(1) 数组索引）。INT8 只有 256 个值，任何一元函数都可以变成 256 条目的查表，"函数计算"变成"数组索引"。

→ 详见 [Stage 0 §7.6](Stage0_量化基础与硬件基石.md)

---

### Q15：Q/DQ 节点插在什么位置？哪些算子需要，哪些不需要？

**核心判据一句话**："这个算子会改变 tensor 里每个元素的值吗？"

- **值变** → 需要 Q/DQ（确保输入输出在正确的数值域）
- **形变/搬运/选择** → 不需要 Q/DQ（INT8 值不动，照搬）

具体分为五大类：

**类别 A：INT8 kernel 算子 → 全 Q/DQ**

| 算子 | 权重 Q | 输入 Q | 输出 DQ |
|------|:--:|:--:|:--:|
| Conv1d/2d/3d | ✅ per-channel | ✅ per-tensor | ✅ |
| ConvTranspose | ✅ per-channel | ✅ per-tensor | ✅ |
| Linear | ✅ per-channel | ✅ per-tensor | ✅ |

**类别 B：多输入算子 → 共享 Q + DQ**

Add / Sub / Mul / Concat / Cat。两个输入必须用同一个 scale 和 zero_point，否则整数运算无意义（`q₁=100, S₁=0.02` = 2.0，`q₂=40, S₂=0.05` 也 = 2.0 → 100+40=140 没意义）。

**类别 C：没有 INT8 kernel → FP32 fallback → DQ + Q**

Sigmoid / Tanh / Softmax / LayerNorm / GELU（无 LUT 时）。先 DQ 回 FP32 → FP32 计算 → Q 回 INT8。但如果用 LUT 查表（§7.6），Sigmoid/Tanh/GELU 可以走类别 D。

**类别 D：纯搬运 → 不插任何 Q/DQ**

Reshape / View / Flatten / Transpose / Permute / Squeeze / Unsqueeze / Slice / Split / Chunk / Pad / MaxPool2d / AvgPool2d / Dropout / Identity / ReLU（LUT 查表）。

Pool 不需要 Q/DQ 的原因：Max 只比大小，INT8 比大小 = FP32 比大小（同一 scale 下是单调映射）；Avg 在 INT32 累加后取整即可。

**类别 E：特殊**

| 组件 | Q/DQ | 原因 |
|------|:--:|------|
| Bias | Derived | `S_bias = S_w × S_a`，不插独立 Q |
| BatchNorm | ❌ | fuse 阶段已吸收进 Conv，图中不存在 |
| 模型输入 | Q | 首层之前把 FP32 输入量化 |
| 模型输出 | DQ | logits/bbox 需要 FP32 精度 |

**常见误判速记**：
- Reshape/Transpose 不需要 Q/DQ → 值没变，只是 view 变了
- MaxPool 不需要 Q/DQ → 比大小不碰值
- Concat 本身不插 Q → Q/DQ 插在输入分支上（共享 scale），拼完再 DQ

**完整速查表**（30 种算子）：

```
算子             权重Q  输入Q  输出DQ   备注
──────────────────────────────────────────────
Add                    共享Q    ✅     类别B
AvgPool2d                ❌     ❌     类别D
Bias            Derived  —     —     类别E
Cat                     共享Q    ✅     类别B
Conv1d/2d/3d      ✅     ✅     ✅     类别A
Concat                  共享Q    ✅     类别B
Dropout                  ❌     ❌     类别D
Flatten                  ❌     ❌     类别D
GELU                    DQ     Q     类别C (有LUT则D)
LayerNorm               DQ     Q     类别C
Linear             ✅     ✅     ✅     类别A
MaxPool2d                ❌     ❌     类别D
Pad                      ❌     ❌     类别D
ReLU                     ❌     ❌     类别D (LUT)
Reshape                  ❌     ❌     类别D
Sigmoid                 DQ     Q     类别C (有LUT则D)
Softmax                 DQ    Q/FP32 类别C
Transpose                ❌     ❌     类别D
```

→ 完整决策树 + 每个类别的 ASCII 数据流图见 [Stage 0 §7.8](Stage0_量化基础与硬件基石.md)

---

### Q15.5：Server 和 Edge 的 Q/DQ 规则一样吗？QAT 仿真应该按哪个来？

**答：不一样。QAT 的意义就是仿真目标硬件——硬件上怎么做量化，QAT 就怎么插 FQ。仿真越像，梯度越接近部署时的真实误差。**

Q/DQ 规则不由"PT2E 默认策略"决定，而由目标硬件的真实量化行为决定：

```
Server（GPU，有 FP32 ALU，fallback 几乎零开销）:
  AvgPool → 不插 FQ（掉回 FP32 算，省事）
  MaxPool → 不插 FQ（INT8 比大小等价 FP32）
  Reshape/Transpose → 不插 FQ（纯搬运，值不变）

Edge（MCU/NPU/DSP，没有 FP32 ALU 或 FP32 慢 100×）:
  AvgPool → 必须插 output FQ（全程 INT8，没有 FP32 走廊可掉）
  MaxPool → 不插 FQ（同上，INT8 比大小能直接跑）
  Reshape/Transpose → 不插 FQ（同上，纯搬运不碰值）

关键差异只在 AvgPool：
  Server: Conv → DQ → AvgPool(FP32) → Q → Conv  （AvgPool 透明）
  Edge:   Conv_INT8 → requantize → AvgPool(INT8) → requantize → Conv
                                ↑ 必须产出 INT8 → 必须有 output QSpec
```

**AvgPool 在端侧怎么算？** 没有 FP32 除法指令，靠 requantize 吸收 `÷kernel_size`：

```
AvgPool 2×2 = (q1+q2+q3+q4) / 4

INT8 域改造:
  1. 4 个 INT8 相加 → INT32 sum
  2. q_out = round(sum × (S_in / 4×S_out) + Z_out)  ← "÷4" 吸进 scale 乘法
  3. S_out 由端侧校准数据单独跑一次确定（逐层校准）
```

**核心原则**：

```
QAT 的 FQ 插入规则 ≠ PT2E 默认策略
QAT 的 FQ 插入规则 = 目标硬件每层在哪个域跑 × 那里有没有 requantize

确定流程:
  1. 拿到目标芯片的算子支持列表 → 哪些 op 有 INT8 kernel？
  2. 有 → 插 FQ（模拟 INT8 噪声）
  3. 没有 → 芯片怎么处理？FP32 fallback → 不插 FQ
                        INT8 变通（如 AvgPool 的 sum+requantize）→ 插 FQ
  4. 把上面的规则写进 Quantizer.annotate()
```

**一句话**：QAT 不是为了"让 PT2E 跑通"，是为了"让训练时的梯度见过部署时会发生的量化噪声"。芯片在哪些地方发生量化，QAT 就在哪些地方插 FQ——不多插、不少插、不错插。

---

## PYTHON / PYTORCH 导出坑

### Q16：Torch 2.6 PT2E 底层用的是 FX 图吗？

**答**：是。PT2E 底层是 `torch.export` 产的**严格版 FX 图**（不是旧版 `symbolic_trace`）。Dynamo trace 一遍 → 产出 FX Graph → Quantizer 在 graph node 上标注 → prepare 插节点 → convert 生成 q/dq。

---

### Q17：PT2E 量化后导出 ONNX 一直 Dynamo 报错，怎么办？

**答**：根因通常是 Torch 2.6 Dynamo-ONNX 对 `batch_norm` 家族有翻译 bug（`'tuple' object has no attribute 'dtype'`，卡在 `_native_batch_norm_legit_functional`）。三个方案：

1. **改 BN training 字面量**：在 FX graph 上把 BN 节点的 `training=True` 改成 `False`（但实测 no_training 变体也有 bug）
2. **走 legacy 导出器**：注册 4 个 `quantized_decomposed` op 的 symbolic → ONNX QDQ，legacy 路径原生支持 BN
3. **手动 fold BN**：如果只是看 Q/DQ 插入位置，手动把 BN fold 进 Conv 再导出

Legacy 注册 symbolic 是最干净的——旧导出器 BN 翻译是成熟的，缺的只是 PT2E 量化 op 映射。

---

### Q18：BN 是什么？在我的图里干什么？

**答**：BN（Batch Normalization）是预训练模型自带的归一化层。训练时用 batch 统计量（μ_B, σ²_B），推理时用运行统计量（running_mean, running_var）。你的 FX 图里它是 `_native_batch_norm_legit_functional(training=True)`——training 标志被 `model.train()` 烤进了图里，导致 ONNX 翻译失败。BN 本身跟量化无关，它只是挡在 Q/DQ 导出路上的一个 op。

---

### Q19：有什么隐藏概念在正文里被当成了"你显然知道"？

**答**：三个核心的：
- **INT32 累加器 + requantize**：量化推理的完整数据循环，不只是 INT8 乘法
- **LUT 查表**：非线性算子在量化推理中怎么解决
- **qscheme**：PyTorch 把"对称+粒度"打包成的 enum

→ 补在了 [Stage 0 §7.5-7.8](Stage0_量化基础与硬件基石.md)

---

*持续更新中。每次遇到新问题、踩到新坑，都往这里加一条。*
