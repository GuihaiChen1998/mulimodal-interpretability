# 第 1 周：工程冒烟测试（实验动机与实验设计）

> 截至 2026-09-26。依据 `HANDOFF.md` 第 5–6 节与 `record.md` 第 3、9、10 次讨论。
> 本周**只打通工程链路**，不产出论文结论；结论性的比较（M0 vs M2-A vs M2-B）在第 2 周。

---

## 1. 实验动机

### 1.1 要回答的科学问题
从文本学来的 Steerling 概念空间（33,732 个已命名 known 概念 + 101,196 个 discovered 概念 + 残差 ε），能否成为**视觉输入的内生可解释接口**？视觉信息进入模型后，是走已命名概念通路，还是漏进 ε？

### 1.2 为什么值得做
- **空白**：19 篇精读文献里，没有一篇同时做到"内生可解释 + 生成式 + 多模态"。CBGM 做的是图像生成，CB-LLM / Steerling 只有文本，MM-CBM 是判别式图文、不能生成（`lit_review.md` 近邻位置表）。
- **事后解释的问题**：TAM、VPS、SAE-V 等事后方法无法验证是否忠实。内生模型的概念贡献是精确已知的（known + unknown + ε 严格等于隐状态），这正是卖点，也可以用来做忠实性测试床（M5）。
- **风险有依据，也有应对**：IMA、SAE-V、Group-sparse SAE 都发现视觉表示和文本表示天然分离；Basu et al. 发现 Steerling 原版的概念 steering 在临床任务上与随机扰动没有差异。但 IMA、Group-sparse SAE、USAE 也表明这种分离可以靠训练纠正。M2 的思路是把视觉信息**在结构上**先翻译成已命名概念 z = Σ aₖeₖ，再送进 LLM。

### 1.3 为什么第 1 周先做工程
整个方案建立在几个**尚未实测**的工程前提上，任何一个不成立都会迫使方法改道：
1. 官方仓库只提供推理包，**没有训练代码**。masked diffusion（MDM）训练损失要自己实现（参考 LLaDA-V）。
2. `forward` 支持 `input_embeds`，这是读源码得出的，还没有实际跑过。
3. 块因果注意力用 flex_attention 实现，要求 Python ≥ 3.13、CUDA 12.8，在目标 GPU 上能否编译还不知道。
4. 576 个图像 token 恰好是 9 个 64-token 块，文本必须从块边界开始，块掩码是否按预期对齐需要实测。
5. M2 要用到视觉概念子集 V_vis 和 COCO 类别到概念 ID 的映射，这两样目前还不存在。

第 2 周的试点（10 万样本上比较 M0 / M2-A / M2-B）要求这 5 项全部打通。

---

## 2. 实验设计

### 2.0 默认设定（待用户确认的项已标出）

| 项 | 取值 | 状态 |
|---|---|---|
| 骨干 | `steerling-8b-instruct`，冻结；概念头与概念嵌入全程冻结 | instruct 为建议值，**待确认** |
| 视觉编码器 | CLIP ViT-L/14-336，取倒数第二层，去掉 CLS，得到 576 个 patch token | 建议值，**待确认**；代码保持与编码器无关 |
| 投影器（本周） | M0：2 层 MLP（LLaVA-1.5 式）。M2 的概念投影器在第 2 周实现 | — |
| 序列布局 | `[576 个图像 token（9 个块，不 mask）][文本从第 576 位开始]` | — |
| 算力 | Runpod 1 × A100 SXM 80GB（Ampere sm80，与原计划的 A6000 同架构） | 替代原先的 4 × A6000 |
| 数据 | LLaVA-Pretrain（558K）中取 1K 条做过拟合；COCO 2017 val 标注用于 V_vis 映射 | — |

### 2.1 实验列表与验收标准

**E1 环境**
- 用 uv 装 Python 3.13，安装 PyTorch（CUDA 12.8 构建）和 `steerling` 包。
- 单独测试 flex_attention 在 A100 上的 `torch.compile` 编译：用 64-token 块因果掩码，与朴素掩码注意力比较输出。
- ✅ 验收：编译成功；与朴素实现的最大绝对误差 < 1e-2（bf16）；记录显存和吞吐。

**E2 骨干加载与分解校验**
- 加载 `steerling-8b-instruct`，用几个文本 prompt 生成，确认输出正常。
- 取最后一层隐状态 h，校验 known + unknown + ε 与 h 的差异。
- ✅ 验收：生成文本正常；分解误差在数值精度范围内；bf16 推理显存约 18GB（与文档一致）。

**E3 `input_embeds` 通路**
- 同一段文本分别用 `input_ids` 和 `input_embeds = embed(input_ids)` 前向，比较 logits。
- 在前面拼接 576 个随机图像向量，检查文本第一个 token 位于第 576 位（第 10 个块的起点），并检查块掩码：图像块内部双向、块间因果、文本能看到全部图像块。
- ✅ 验收：两条通路的 logits 一致（bf16 容差内）；掩码可视化与设计一致。

**E4 视觉通路（M0）**
- CLIP ViT-L/14-336 → 576 × 1024 → MLP → 576 × 4096，送入 `input_embeds`。
- ✅ 验收：端到端前向和反向跑通；只有投影器有梯度，Steerling 参数的 `requires_grad=False`。

**E5 MDM 损失**
- 参考 LLaDA-V：采样 t ~ U(0, 1]，答案 token 以概率 t 替换为 mask token，只在被 mask 的位置算交叉熵，并按 1/t 加权；图像前缀和问题部分不 mask。
- 需要确认 Steerling 的 mask token id 与噪声调度（读源码）。
- ✅ 验收：单元测试中 mask 比例符合 t；只对 mask 位置计损失；loss 为有限值。

**E6 1K 样本过拟合**
- 从 LLaVA-558K 取 1K 条图文对，只训 M0 投影器（lr 1e-3，若干 epoch）。
- ✅ 验收：训练 loss 明显下降并接近饱和；对训练图片生成的描述能复现训练文本的主要内容。不通过就回查 E3–E5。

**E7 V_vis 与 COCO 映射**
- 输入：`known_concepts.csv`（latin-1 编码，**本会话尚未上传，需要用户提供**）。
- 先用规则剔除 LaTeX、代码、单字母、纯语法类概念，再用 CLIP 文本编码器计算概念名/描述与视觉相关的程度，得到 V_vis 候选集。
- COCO 80 个类别 → 概念 ID：用名称匹配加 CLIP 文本相似度取 top-k，并人工抽查。
- ✅ 验收：输出 `v_vis.json`（概念 ID 列表与筛选理由）和 `coco2concept.json`；报告 V_vis 的规模，以及 80 个 COCO 类别中有多少能映射到概念。

### 2.2 本周产出
- 代码：环境脚本、Steerling 加载/分解校验、`input_embeds` 与块对齐测试、M0 投影器、MDM 损失、过拟合训练脚本、V_vis 构建脚本。
- 记录：每项实验的实测数字（显存、吞吐、误差、loss 曲线）写回 `record.md`。
- 结论：第 2 周试点是否可以开始（go / no-go）。

### 2.3 存储估算（决定 Pod 磁盘和 volume 大小）
| 内容 | 大小（约） |
|---|---|
| steerling-8b-instruct 权重（bf16） | 16–20 GB |
| CLIP ViT-L/14-336 | 1.7 GB |
| Python 环境 + PyTorch + 编译缓存 | 10–15 GB |
| LLaVA-558K 图像（第 2 周全量） | 约 27 GB |
| COCO 2017（val 标注 + 图像） | 约 1–20 GB |
| 检查点（投影器很小，LoRA 阶段增多） | 10–30 GB |

本周 100GB 的 container disk 足够。实际方案：US-WA-1 不支持 network volume，所以改用 **200GB Pod 持久盘（`/workspace`）**，权重、数据、检查点都放在这里。

### 2.4 风险与备选
| 风险 | 备选方案 |
|---|---|
| flex_attention 在 A100 上编译失败 | 退回用 SDPA 加显式块掩码（速度较慢，但正确性不受影响） |
| `input_embeds` 与内部位置编码或掩码冲突 | 在包外包装一层 forward，自己构造位置索引和块掩码 |
| Steerling 的 mask token 或噪声调度与 LLaDA 不同 | 以 Steerling 源码为准，改写损失 |
| V_vis 覆盖不到空间关系、计数、OCR | 如实记录，在第 2 周相应限定评测任务 |
