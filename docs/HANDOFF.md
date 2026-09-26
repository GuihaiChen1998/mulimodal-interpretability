# 续接文档（HANDOFF）：基于 Steerling 的内生可解释多模态模型

> 新会话请先完整阅读本文档。详细历史见 `record.md`，19 篇文献的精读笔记见 `lit_review.md`。
> 截至：2026-09-26，第 10 次讨论之后。

---

## 0. 给新会话中 Claude 的工作约定

1. **每次讨论结束后，在 `record.md` 末尾追加一节**（日期 · 第 N 次讨论：用户提供了什么、结论、待决定）。Claude 无法写回项目文件，需在输出目录生成更新版，由用户替换。
2. **读文献必须读全文（含附录）**，读完一篇立即写该篇笔记（核心问题、方法、实验、关键数字、优点、局限、未来工作〔区分"原文"与"推断"〕、与本项目关系），不要只读摘要。无法逐行读完的部分（如长篇数学推导）需明确说明。
3. 用中文交流；对事实做核实，不确定的要标明。
4. 用户算力：**4 × A6000（48GB/张，Ampere，无 FP8，PCIe 互联）**。

---

## 1. 项目目标

- 基于 **Steerling-8B**（Guide Labs 的内生可解释 causal diffusion LM）提出**内生可解释的多模态大模型**。
- 第一篇：**通用场景的图像+文本**（不做全模态）。
- 医学/抑郁（用户已有 DAIC-WOZ）作为后续第二篇或下游应用，不在第一篇范围内。

## 2. 已做出的决定

| 决定 | 内容 |
|---|---|
| 场景 | 先通用图文，后音频，抑郁后置 |
| 骨干 | Steerling-8B，冻结；Stage 2 加 LoRA；**概念头与概念嵌入全程冻结**（保证概念语义固定）。建议用 instruct 版（待确认） |
| 主方法 | **M2：概念化视觉输入**（见第 5 节） |
| 论文组合 | M0（动机/基线）→ M2 + M3（泄漏控制）→ M5（忠实性测试床，第二贡献）；M4（中层对齐）视 M0 结果再定 |
| 视觉编码器 | Claude 建议 **CLIP ViT-L/14-336**（快速迭代），SigLIP2 作为后期消融——**用户尚未确认** |

## 3. Steerling 关键事实（已核实，含源码）

- 64-token 块内双向、块间因果的 masked diffusion；上下文 4096；GQA 32 头 / 4 KV 头；bf16 推理约 18GB。
- 有 `steerling-8b` 与 `steerling-8b-instruct` 两版。官方仓库（github.com/guidelabs/steerling）**是推理包，无训练代码**；要求 Python ≥3.13、CUDA 12.8；注意力用 flex_attention（需确认在 A6000 上可编译）。
- `forward` 支持 `input_embeds`。
- 已知概念头：线性预测器 4096→33,732 + sigmoid + top-32；特征 = Σ 权重 × 概念嵌入；**概念嵌入为稠密矩阵 33,732 × 4096，位于最后一层隐空间**。发现概念 101,196 个，低秩分解（rank 256，top-128）。
- ε 修正使 known + unknown + ε 严格等于隐状态（分解无损）。
- **自带 steering 注入：在第 ≥16 层把向量加到残差流**（`inject_layer=16`，`position_injection`）→ 概念嵌入在中后层残差流中可直接起作用。这是 M2-B 的依据。
- 官方 33K 概念表（用户已上传过 `known_concepts.csv`，**latin-1 编码**）：99.7% 的描述以 "Tokens ..." 开头，即按"促进哪些输出 token"定义的词汇/话题簇；可 steer 的 16,327 个；tone 441、alignment 1,020、demographic 1,347；有大量视觉话题与声音事件概念，**没有副语言/韵律/面部行为概念**；临床状态概念缺失（这是抑郁方向后置的原因）。

## 4. 文献调研结论（详见 `lit_review.md`）

- 已精读 19 篇：用户给的 15 篇（IMA、CoX-LMM、Same Task、HEIE、VPS、Info Flow、InterSHAP、SAE-V、USAE、TAM、Group-sparse SAE、MICLIP、PID、Emotion Neurons、Inside-Out）+ 4 篇近邻（MM-CBM、CB-LLM、CBGM、SIM）。
- **研究空白 1（主线）已确认**：尚无"内生可解释 + 生成式 + 多模态"兼具的工作。脉络：CBGM（图像生成，Steerling 作者前作）→ CB-LLM / Steerling（文本生成）→ MM-CBM（判别式图文，不能生成）→ 本工作。
- 证据链（核心风险及其可解性）：
  - 视觉/文本表示天然分离：IMA、SAE-V（文本训练的字典解释不了视觉）、Group-sparse SAE（标准训练学成分裂字典）。
  - 但可训练纠正：IMA、Group-sparse SAE、USAE、CoX-LMM 附录 G。
  - 视觉作用集中在中低层、对齐偏晚：Info Flow、Same Task、PID。
  - 融合能力在指令微调阶段才学到：PID → **诊断要在 Stage 2 后下结论**。
  - MLLM 事后解释无法验证忠实性：TAM、VPS、Emotion Neurons → 内生模型的卖点。
- 关于 Steerling 本身的两篇第三方证据：**Basu et al. arXiv:2603.18353**（临床分诊中概念 steering 与随机无差异）；**SIM arXiv:2606.12289**（预测主要由约 500 个已监督概念中介）。二者需同时引用。
- 其他研究空白：忠实性金标准测试床；概念级（而非模态级）跨模态分解；带随机对照的概念干预评测；情感/临床场景（后续）。

## 5. 主方法 M2 设计 v0

每个图像块：视觉编码器 → **概念投影器** → 在视觉概念子集 V_vis 上的 TopK 稀疏权重 a → **z = Σ a_k e_k**（e_k 为冻结的 Steerling 概念嵌入）+ 有预算的残差 r。

两个变体（第 2 周试点决定）：
- **M2-A 输入层**：视觉 token = W·z + r（W 为共享线性映射，每个概念对应一个固定输入向量）。
- **M2-B 中层注入**：图像位置输入一个可学习占位向量，自第 L 层起把 α·z 加到残差流，复用 Steerling 自己的 steering 通道。代价：第 L 层前文本看不到图像内容。

损失：
1. 掩码扩散损失（只 mask 答案文本；图像为不 mask 的前缀，对齐到块边界）。
2. **文本→视觉概念蒸馏**：冻结 Steerling 读描述得到概念激活作为图像概念目标，用**排序损失**（依据 SIM 对称性 I），无需人工标注。
3. 残差预算（并画"可解释性–性能"曲线）。
4. 对抗解耦（CB-LLM）：探针从 r / 发现概念通道预测图中物体，残差通道对抗使其失败。
5. 正交约束（CBGM）。
6. 可选：OWLv2 检测器弱标签（约 20% 样本即可，依据 CBGM）；需检查标签排序一致性（依据 SIM）。

训练：Stage 1 只训概念投影器（LLaVA-558K，估计约 1 天）；Stage 2 投影器 + LoRA（LLaVA-665K，估计 2–4 天）。时间均为 FLOPs 粗估，需实测。训练代码参考 **LLaDA-V**（CVPR 2026，LLaDA-8B + SigLIP2 + MLP，masked diffusion 视觉指令微调，开源）。

应用亮点：**按构造检测幻觉**（输出了"狗"但输入概念中没有狗 → 来自语言先验），在 POPE 上验证。

评测套件：SIM 对称性 II 雅可比子空间检验（**分别在文本输入与图像输入上测**）；已知/发现/ε 的贡献占比；概念–物体 AUROC（COCO）；Group-sparse SAE 的 MMS 与双模态概念数；带随机概念对照的增强/移除干预；POPE；TAM 的 F1-IoU 协议；Same Task 的成对图文任务一致性；M5：用精确概念贡献评估 TAM/VPS/SAE。

主要风险：视觉信息压进概念空间导致性能下降（用残差预算曲线应对）；概念覆盖不到空间关系、计数、OCR（如实报告并限定任务）；flex_attention 在 A6000 的兼容性；无官方训练代码。

## 6. 下一步（立即可做）

**第 1 周：工程**
1. 环境：Python ≥3.13、CUDA 12.8，验证 flex_attention 在 A6000 上可用。
2. 加载 steerling-8b-instruct，打通 `input_embeds` + 576 图像 token 对齐 9 个块。
3. 参考 LLaDA-V 实现 MDM 损失；1K 样本过拟合测试。
4. 构建 V_vis（去掉 LaTeX/代码等非视觉概念，可用 CLIP 文本编码器算概念名与图像的相关性）与 COCO 类别→概念 ID 映射。

**第 2 周：试点与选型**
在同一 10 万样本上跑 M0（普通 MLP 投影）、M2-A、M2-B，比较概念占比、对称性 II、概念–物体 AUROC，选定变体；M0 结果即论文动机图。

## 7. 待决定事项

- [ ] 视觉编码器：确认 CLIP ViT-L/14-336（Claude 建议）还是 SigLIP2。
- [ ] 骨干 base 还是 instruct（建议 instruct）。
- [ ] 是否加入 M4 中层对齐（视 M0 结果）。
- [ ] 目标会场与时间线（尚未讨论）。

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `HANDOFF.md` | 本文档，新会话入口 |
| `record.md` | 全部讨论的逐次记录（第 1–11 次） |
| `lit_review.md` | 19 篇文献的逐篇精读笔记 + 总览表 + 研究空白 + 方法启示 |
| `known_concepts.csv` | Steerling 官方 33K 概念表（latin-1 编码），做 V_vis 时需要 |
| 与 Grok 的讨论记录 | 早期背景，已被本文档与 record.md 覆盖，可不再上传 |
