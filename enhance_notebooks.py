"""Insert narrative explanations into existing notebooks at key transition points."""
import json, copy
from pathlib import Path

def md_cell(text: str) -> dict:
    """Create a markdown cell from text."""
    lines = text.strip().split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in lines[:-1]] + [lines[-1] + '\n'],
    }

def insert_after_cell(cells: list, index: int, text: str):
    """Insert a markdown cell after the cell at given index."""
    cells.insert(index + 1, md_cell(text))

def enhance_stage0(nb_path: str):
    """Add narrative bridges to Stage 0 notebook."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    # Find key insertion points by scanning cell content
    insertions = []  # (index, text) tuples

    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])

        # After "开篇" — add concrete example before jumping into FP32
        if '## 开篇' in src and '## 知识总览' in src:
            insertions.append((i, """### 在进入公式之前，先建立直觉

为什么 256 个值能替代 42 亿个值？我们来算一笔账。

一个 ResNet50 有约 2500 万个参数。FP32 下每个参数 4 字节，总共约 100MB。
训练时这些参数"自由"地在 FP32 空间移动。但训练完后，绝大部分参数的\"自由度\"大幅下降——它们被训练数据"锁定"在某个很小的范围内。

**打个比方**：你在学投篮。刚开始练习时，手臂的角度、力量、
出手点都在大范围调整（训练阶段的高方差）。但练了 1000 次后，
你的动作收敛到一个很小的范围——偏差 2cm 就会投偏。
这 2cm 的范围，256 个等级就够了。后面 99.9% 的精度是\"练习过程中\"才需要的——推理时不需要。

这就解释了为什么训练用 FP32 但推理可以用 INT8：**训练需要搜索空间，推理只需要执行路径。**

下面是实际的数学和代码。"""))

        # After seeing FP32/FP16/BF16 bit layout — bridge to quantization formula
        if '### 1.2 动手验证' in src and i+1 < len(cells) and cells[i+1]['cell_type'] == 'code':
            insertions.append((i+1, """### 1.3 从这里到量化

现在你看到了 FP32 的内部结构：1 bit 符号 + 8 bits 指数 + 23 bits 尾数。
总共 32 bits ≈ 42.9 亿个可能值。

**量化的任务**：用 INT8 的 256 个值来近似 FP32 的 42.9 亿个值。
这听起来不可能，但关键在\"scale\"——不是均匀地选 256 个值，
而是根据数据的范围**自适应地**选 256 个值。

比如激活值全在 [0, 3.0] 区间——用 256 个值去覆盖 [0, 3.0]，
每个值的间距是 3.0/255 ≈ 0.012。只要你的数据不需要比 0.012 更细的精度，
这个量化就是\"无损\"的。

**接下来的第 2-6 节就是在回答：怎么找到这个最佳的 scale？**"""))

        # Before section 5 (calibration algorithms) — bridge
        if '## 5. 校准算法' in src:
            insertions.append((i, """### 4.3 从粒度到校准：现在的问题是——"怎么找到 scale"

确定了量化粒度（per-tensor vs per-channel）后，
下一个问题是：**怎么计算 scale 和 zero_point？**

你需要用到校准数据——从训练集里抽几百张图（不需要标签），
跑一遍模型，记录每层激活值的分布，然后据此计算 scale。

**这就是校准（Calibration）**。不同的"据此计算"方式，
衍生出不同的校准器。下面四种校准器从简到繁，
你会发现它们本质上是在做同一个权衡：**覆盖范围 vs 分辨率。**"""))

        # After calibration code — guidance on which to choose
        if '## 6. 舍入策略' in src and i > 0:
            insertions.append((i-1, """### 5.3 四种校准器——什么时候用哪个？

看完上面的代码和对比实验，你可能想问：\"到底应该用哪个？\"

- **MinMax**：最快，适合 outlier 少的网络（ResNet、VGG）。如果数据已知很干净，MinMax 就够了。
- **Percentile**：当数据有已知 outlier 但你知道它们是噪声时（比如传感器坏点）。
- **MSE**：当你追求\"在 MSE 意义上最优\"但不在乎分布一致性时。
- **KL Divergence**：工业标准，TensorRT 的默认选择。适合大多数场景，
  尤其当你不确定数据分布时。KL 的代价是校准更慢（需要更多样本建直方图）。

**一个常见的坑**：对 LLM 的激活值不要用 Percentile——LLM 的 outlier 不是噪声，
而是模型\"精心\"学到的注意力聚焦信号。用 Percentile 删掉它们 = 删掉了模型的核心能力。
这就是 Stage 6 要讲的 SmoothQuant 和 AWQ 的动机。"""))

        # Before handwritten engine — motivation bridge
        if '## 8. 从零手写' in src:
            insertions.append((i, """### 7.4 从理论到实践：把所有知识串起来

前 7 节建立了量化推理的完整理论基础。现在把这些知识串成一条线：
**手写一个不依赖任何框架的 INT8 推理引擎，在 MNIST 上验证。**

这个实验会逼你面对所有细节：
- 权重的 scale 怎么算？（对称，per-tensor）
- 激活的 scale 怎么算？（需要 calibration——用校准数据跑一遍前向）
- bias 怎么处理？（保持 FP32——它的量级太小，量化会丢失精度）
- 推理时每一层的数据流是怎样的？（INT8 输入 → 反量化 → matmul → 量化输出 → 下一层）

做出这个之后，进入 Stage 1 学 PyTorch 的量化 API 时你会发现：
PyTorch 的 Observer、FakeQuantize、QuantStub——
每一个抽象层都对应着你手写代码中的某个功能。你不再是在"学 API"，
而是在看\"别人怎么把我手写的逻辑工程化了\"。"""))

    # Apply insertions in reverse order (to preserve indices)
    for idx, text in sorted(insertions, reverse=True):
        insert_after_cell(cells, idx, text)

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Enhanced {Path(nb_path).name}: {len(cells)} cells (added {len(insertions)} narrative bridges)")


def enhance_stage1(nb_path: str):
    """Add narrative bridges to Stage 1 notebook."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    insertions = []

    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])

        # After intro — explain the "六层抽象" motivation
        if '## 知识总览' in src and '六层抽象' in src:
            insertions.append((i, """### 为什么要分六层？

你可能想问：不就是量化吗？为什么 PyTorch 要搞六层抽象、
上万行代码，而不是一个简单的 `quantize(model)` 函数？

**因为量化不是一个操作，而是一个管线。** 不同后端（fbgemm/qnnpack/TensorRT）、
不同精度（8-bit/4-bit）、不同策略（PTQ/QAT）对每一层的需求都不同。

这六层抽象的设计原则是 **"关注点分离"**：
- 统计收集（Observer）和量化模拟（FakeQuantize）分离
- 图预处理（fuse_modules）和量化标注（QConfigMapping）分离
- API 模式（Eager/FX/PT2E）和底层机制分离

这样当你需要自定义一个 Observer 时，不需要改动 FakeQuantize 的代码。
当你需要适配新后端时，只需要换一个 BackendConfig。

**接下来我们从最底层（Observer）开始，一层层往上爬。**"""))

        # Between Observer and FakeQuantize — bridge
        if '## 2. FakeQuantize 状态机' in src:
            insertions.append((i-1, """### 1.6 从 Observer 到 FakeQuantize

Observer 只是"看"。它收集 min/max，算出 scale/Z，但**不做任何量化操作**。

**谁说"看"完要"做"？这就是 FakeQuantize 的职责。**

FakeQuantize 继承自 Observer——所以它"会看"。同时它额外加了一个能力：
在 forward 中执行量化-反量化操作（所以叫 "Fake"——量化是真的，但数据类型还是 float）。

**这个继承关系是 PyTorch 量化设计中最精妙的部分之一。**
它意味着一个 FakeQuantize 模块同时是：
1. 一个 Observer（有 min_val, max_val, calculate_qparams）
2. 一个量化模拟器（forward 中做 round+clamp+dequant）

下面我们拆开它的状态机，看它是如何用两个 flag 驱动 QAT 三阶段的。"""))

        # Before fuse_modules — bridge from stubs
        if '## 4. fuse_modules' in src:
            insertions.append((i-1, """### 3.3 Stubs 和融合的关系

QuantStub/DeQuantStub 告诉框架"在哪里量化"。
但还有一个重要的问题没解决：**量化之前的预处理。**

训练好的模型推理时，Conv+BN 可以折叠成一个 Conv（Stage 0 学了数学原理）。
如果在折叠之前插入 FakeQuantize——你的量化节点会插在 Conv 和 BN **之间**，
这会导致 BN 的输入是量化后的值，和训练时的分布完全不同。

**所以 fuse_modules 必须在 prepare_qat 之前执行。**
把 Conv+BN+ReLU 先融合成单个模块，然后再在融合模块的输入输出处插入 FakeQuantize。

下面我们拆开 fuse_modules 的完整流程。"""))

        # Before prepare_qat_fx summary
        if '## 9. 源码汇总' in src:
            insertions.append((i-1, """### 8.3 从配置到执行：QConfigMapping 是"策略层"，prepare_qat_fx 是"执行层"

现在你已经学完了所有独立模块。把它们串起来的，是 `prepare_qat_fx` 和 `convert_fx`。

`prepare_qat_fx` 像一个"总调度"：它读取 QConfigMapping（每层用什么策略），
遍历计算图，在对应的位置插入对应的 Observer/FakeQuantize 模块。
`convert_fx` 是它的反向操作：QAT 训练完后，
把 FakeQuantize 替换为真正的量化/反量化节点。

下面我们就来看这两个函数内部的完整流程。"""))

    for idx, text in sorted(insertions, reverse=True):
        insert_after_cell(cells, idx, text)

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Enhanced {Path(nb_path).name}: {len(cells)} cells (added {len(insertions)} narrative bridges)")


def enhance_stage15(nb_path: str):
    """Add narrative bridges to Stage 1.5 notebook."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    insertions = []

    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])

        if '## 1. 手写 QAT' in src and '脱离 PyTorch' in src:
            insertions.append((i, """### 为什么要手写？

Stage 1 教你用 PyTorch 的 `prepare_qat_fx` 一键完成 QAT 准备。
这很方便——但也让你对 QAT 内部到底发生了什么"无感"。

**手写的目的不是替代 PyTorch，而是拆开黑箱。**
当你手动管理 observer_enabled/fake_quant_enabled 的切换、
手动在每个 epoch 控制 scale 的冻结时机、手动处理 BN 的 eval 模式——
你会发现很多"最佳实践"（Observer 开 2-3 epoch、BN 必须冻结）
不再是需要背诵的咒语，而是可以推导出来的结论。

下面我们从头写一个不依赖 torch.ao 的 QAT 训练循环。"""))

        if '## 2. Observer 消融' in src:
            insertions.append((i-1, """### 1.3 从手写到理解：Observer 为什么重要

手写 QAT 训练循环跑通后，你可能会注意到一个细节：
我们用的是一个最朴素的 MinMax Observer——直接取输入数据的 min 和 max。

**但"最优"的 min/max 不在于覆盖所有数据——而在于平衡"覆盖范围"和"分辨率"。**

下面的消融实验会给你一个量化的答案：
不同的 Observer 在同一模型、同一比特下能差多少？
这个差距在低比特下是放大还是缩小？

**做完这个实验后，你对 PyTorch 为什么默认用 MovingAverageMinMaxObserver
（而不是最简单的 MinMaxObserver）会有一个基于数据的理解——而不是"文档这么写的"。**"""))

        if '## 3. 比特宽度系统性实验' in src:
            insertions.append((i-1, """### 2.3 从 Observer 消融到比特实验：串起来

Observer 消融告诉你"选对 Observer 能差 2-3 个点"。
但还有一个更根本的问题：**不同比特数下，QAT 的表现如何？**

你会发现 8-bit 几乎无损、4-bit 开始降、2-bit 直接崩。
这个"崩溃点"不是魔法——它是固定 scale QAT 的硬上限。
**而这个上限，正是 Stage 2（LSQ）要解决的问题。**

所以这一节的实验不是"跑一个 benchmark"——
它是在为 Stage 2 建立"问题意识"。
做实验时留心观察：在哪个比特数下 scale 的精确性开始变得"生死攸关"？"""))

        if '## 6. 固定 scale QAT 的能力边界' in src:
            insertions.append((i-1, """### 5.3 从失败案例到能力边界：把零散的"坑"串成系统性的理解

上面四个失败案例看起来是独立的"坑"——但它们的根因是同一个：
**固定 scale QAT 假设 scale 在校准阶段就能被正确确定，
而且这个"正确"的 scale 在训练过程中不会过时。**

在 8-bit 下这个假设成立（256 个等级足够宽容）。
在 4-bit 下它开始动摇（16 个等级，scale 的偏差被放大）。
在 2-bit 下它完全崩塌（4 个等级，scale 几乎不可能被正确设定）。

**这个渐进式的崩溃，是 LSQ 存在的理由。**
进入 Stage 2 时，你不是在学一个"凭空出现的技巧"——
你是在学一个"我已经用实验证明了必要性的方案"。"""))

    for idx, text in sorted(insertions, reverse=True):
        insert_after_cell(cells, idx, text)

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Enhanced {Path(nb_path).name}: {len(cells)} cells (added {len(insertions)} narrative bridges)")


def enhance_stage2(nb_path: str):
    """Add narrative bridges to Stage 2 notebook."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    insertions = []

    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])

        if '## 开篇' in src and 'Stage 1.5' in src:
            insertions.append((i, """### 回顾：Stage 1.5 给了我们什么

在进入 LSQ 的数学之前，先明确我们要解决什么问题。

Stage 1.5 的比特宽度实验给你展示了一组数据：
- 8-bit QAT: 精度接近 FP32（256 个等级，scale 不太精确也能凑合）
- 4-bit QAT: 开始下降（16 个等级，scale 偏差开始显现）
- 2-bit QAT: 直接崩溃（4 个等级，scale 必须极度精确）

**崩溃的根因不是"权重量化太粗"——是 scale 不够精确。**

在 2-bit 下只有 4 个量化等级。如果 scale 偏了 20%，
量化网格就会完全"对不准"权重的最优值。
而固定 scale 只靠前 2-3 个 epoch 的 Observer 决定——
在训练初期权重还在快速变化时做出的决定，到训练后期可能是错的。

**LSQ 的回答：让 scale 变成可学习参数，跟着 weight 一起被梯度优化。**

下面我们一步步实现这个想法。"""))

        if '## 2. LSQ 梯度推导' in src:
            insertions.append((i-1, """### 1.4 从 STE 到 LSQ：就差一步

STE 解决了一个问题：如何让梯度穿过 `round()`。
但它留下了一个盲区：**scale 参数没有梯度。**

在普通 QAT 中这不是问题——因为 scale 是 buffer，本来就不参与训练。
但如果我们要让 scale 可学习——就需要手动给它设计梯度。

**LSQ 的公式 (6) 就是在回答：如果 scale 是可学习参数，它的梯度应该等于多少？**

下面的推导从乘法法则开始，一步步走到最终形式。
推导本身不难（只需要基础微积分+STE 的假设），
但每一步的**物理含义**才是重点——不要只盯着公式看，读每步后面的直觉解释。"""))

        if '## 4. 从零实现 LSQ' in src:
            insertions.append((i-1, """### 3.3 从梯度推导到代码：把公式变成 autograd.Function

公式 (6) 和 (13) 推导完后，下一步是把它们变成可运行的代码。

PyTorch 提供 `torch.autograd.Function` 让你自定义前向和反向逻辑。
LSQ 的 forward 和普通 FakeQuantize 完全一样——量化-反量化。
不同的只有 backward：手动计算 `grad_scale`（公式 6）+ Gradient Scaling（公式 13）。

**注意**：对输入 x 的梯度还是用 STE（和普通 FakeQuantize 一样）。
LSQ 只改变了 scale 的梯度——其他一切不变。"""))

        if '检验标准' in src and i == len(cells)-1:
            insertions.append((i-1, """### 7.3 LSQ 之后——回头看 Stage 1.5 的数据

学完 LSQ 的实现后，回顾 Stage 1.5 的比特宽度实验。

你会意识到：**LSQ 不是"一个更复杂的 QAT 方法"——它是"固定 scale QAT 在低比特下的必然出路"。**

固定 scale QAT 的能力边界在 4-bit 附近。LSQ 把这个边界推进到了 2-bit。
这不是"优化"——是"范式转换"：从"靠 Observer 猜一个 scale"到"让梯度自己找到最优 scale"。

这个范式转换的思想蔓延到了整个量化领域：
- AIMET 的 Range Learning QAT = LSQ 的工程化实现
- QLoRA 的 NF4 = LSQ 思想在非均匀量化上的应用
- BitNet 的三值化 = LSQ 在 1.58-bit 的极限探索

**你刚学完的不是一个"技巧"——是 QAT 领域最重要的理论基石。**"""))

    for idx, text in sorted(insertions, reverse=True):
        insert_after_cell(cells, idx, text)

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Enhanced {Path(nb_path).name}: {len(cells)} cells (added {len(insertions)} narrative bridges)")


if __name__ == '__main__':
    base = Path(r'c:\Users\weijiashengs\Desktop\量化学习o\notebooks')

    enhance_stage0(str(base / 'Stage0_量化基础与硬件基石.ipynb'))
    enhance_stage1(str(base / 'Stage1_PyTorch QAT PT2E 深度拆解.ipynb'))
    enhance_stage15(str(base / 'Stage1.5_QAT训练深度剖析.ipynb'))
    enhance_stage2(str(base / 'Stage2_LSQ与可微量化参数.ipynb'))

    print("\nDone. All notebooks enhanced with narrative bridges.")
