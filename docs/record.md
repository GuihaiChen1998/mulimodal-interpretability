# 项目讨论记录：基于 Steerling 的内生可解释多模态模型

> 每次讨论后追加一节。最新的在最下面。

## 项目目标（当前版本）

- 基于 Steerling-8B（Guide Labs 的内生可解释 causal diffusion LM）做改进，提出内生可解释的多模态 / 全模态模型。
- 备选落地场景：医学，尤其抑郁症评估（访谈音频 + 转写文本 + 面部）。
- 已有数据：DAIC-WOZ。
- 考虑方向：用 adapter / OPD（on-policy distillation）做 1B–4B 规模的模型。

---

## 2026-09-25 · 第 1 次讨论（审阅与 Grok 的对话记录）

### 与 Grok 讨论的要点（用户提供的记录）

1. 整理了 2024–2026 顶会/顶刊的 MLLM 工作；2026 年顶会里"提出通用新基座"的论文明显变少，主流转向统一理解+生成、omni、领域适配。（注：2026 表格中部分条目 Grok 自己标了"需核对"，引用前需逐条核实。）
2. Grok 对 Steerling 的判断：不要做"LLaVA 配方接 Steerling 冲 CVPR"；Steerling 语言能力偏弱（MMLU 46.4）、4K 上下文、块内双向扩散、无官方训练代码。
3. 医学方向：建议收窄为 PHQ-8 分项对齐的概念瓶颈 + 音频（或面部），而不是三模态基座；CVPR 不是合适会场。
4. 全模态：DAIC-WOZ 上图像/视频本质是同一张脸的时间序列，不宜算两个模态；189 场访谈撑不起基座。
5. Adapter vs OPD：adapter 不能缩小推理模型；OPD 才能得到真正 1–4B 的学生。Grok 最终建议 Qwen2.5-3B + 模态 adapter + 概念 adapter + OPD，Steerling 仅作参考。

### Claude 的审阅意见

- **已核实**：Steerling-8B 是 causal diffusion LM，64-token 块内双向、块间因果；known ~33K / discovered ~100K / residual 三路分解；有 base 与 instruct 两个版本（steerling-8b / steerling-8b-instruct，第 3 次讨论更正）；技术报告基准表数字与 Grok 所述一致。MLlm-DR、HiMA-MDD 均真实存在。
- **Grok 漏掉的关键文献**：Basu et al., arXiv:2603.18353, *Interpretability without actionability*。在 400 个临床分诊案例上，Steerling 的概念 steering 与随机概念扰动无显著差异（p=0.84），99.92% 概念激活 < 0.01。
  - 风险：原版 Steerling 的通用概念在临床任务上"可看不可控"。
  - 机会：可作为论文动机——让临床概念真正可因果操作，并扩展到多模态。
  - 注意：该文用的是 base model 自由生成 + 关键词解析，概念是事后从通用 atlas 挑的，不等于 Steerling 路线被否定。
- **Grok 最终方案的问题**：小 LLM + 蒸馏 + 音视频 query 模块 ≈ MLlm-DR（ACM TOMM）；再加概念瓶颈会撞 CB-LLM（Sun et al., ICLR 2025）。且方案已脱离"基于 Steerling"。唯一站得住的贡献点是"跨模态、可因果操作的概念瓶颈"，需正面回应 Basu et al.。
- **Steerling 在论文中的角色需先定**：(A) 冻结 backbone；(B) 架构原则（k+u+ε 线性通路）移植到小模型；(C) 概念教师。建议先用诊断实验决定。
- **DAIC-WOZ 注意事项**：train/dev 提供 PHQ-8 分项分数（可直接作八个概念的监督），test 只有总分/二分类；有工作指出模型会利用访谈者 Ellie 的提问作捷径，建议只用被试话语。数据协议下不要把转写发给商业 API 当教师，用本地开源权重模型。
- **OPD 的叙事风险**：如果分数由黑箱教师蒸馏得到，会削弱"内生可解释"；蒸馏应只服务于理由文本流畅度，分数必须走概念通路。

### 待决定

- [ ] Steerling 当 backbone（A）还是架构原则（B）？
- [ ] 第一篇的模态组合：文本+音频 还是 文本+面部？
- [ ] 目标会场与时间线。

### 下一步（建议的 1–2 周诊断实验，纯推理不训练）

1. 用 `steerling` 包对 DAIC 被试话语（按问答轮次切分，适配 4K 上下文）抽取 known / discovered / residual 表征。
2. 检查：33K known 概念中是否存在与 PHQ-8 八项相关且激活不接近零的概念。
3. 分别用 known 概念、discovered 概念、residual 线性回归 PHQ-8 分项，看信息主要在哪一路。
4. 复刻 Basu et al. 的对照：选中概念 steering vs 随机概念 steering。
5. 判定：信息主要在 residual 且 steering ≈ 随机 → 走方案 B（训练任务专属临床概念层）；否则可走方案 A。

---

## 2026-09-25 · 第 2 次讨论（官方 33K 概念集分析 + 场景选择）

### 用户提供
- Steerling-8B 官方 known 概念表 `known_concepts.csv`（33,732 条；字段：concept_name / description / group / is_steerable / is_tone / is_alignment / is_demographic）。注意：文件非 UTF-8，需用 latin-1 读取。
- 问题：在常规场景做全模态/三模态好，还是在医学或其他场景做好？

### 概念集分析结果
- 33,732 个概念，约 19K 个组；可 steer 的 16,327 个（约 48%）；tone 441、alignment 1,020、demographic 1,347。
- 99.7% 的描述以 "Tokens ..." 开头：概念是按**它促进哪些输出 token** 定义的词汇/话题簇（含大量 LaTeX、代码、单字母等），不是"说话人处于何种状态"。
- PHQ-8 覆盖：睡眠、疲劳、食欲、羞耻/内疚、绝望、自杀有**话题型**概念；兴趣丧失几乎没有（只有一个 "Mood Disorder Symptom Checklists"）；精神运动性改变基本没有。
- 最直接的临床概念（Depression Screening Instruments、Mood Disorder Symptom Checklists、Suicide and Self-Harm 等）多为 is_steerable=False 且被标为 alignment。
- 模态：有大量视觉话题（imagery、photo）、声音事件/音乐话题概念，但**没有副语言/韵律/面部行为**类概念。

### 由此得出的判断
- 话题 ≠ 状态：DAIC 中 Ellie 会问每个人睡眠，"睡眠"概念会对所有人激活，不编码严重程度。这也部分解释了 Basu et al. 的负结果。
- 现有概念空间天然适合图像/声音事件的**语义内容**，不适合临床**副语言状态**。

### 场景建议（Claude）
- 第一篇：常规场景，**先做图像+文本**（不是全模态），核心问题是"文本概念空间能否成为视觉的内生可解释接口"。
  - 工程上可行：冻结 Steerling，LLaVA-1.5 式 576 个视觉 token = 恰好 9 个 64-token 块；图像作为不 mask 的前缀，只对文本做 MDM。
  - 评测：概念激活 vs COCO/VG 物体属性标签、概念 steering 改变描述、POPE 幻觉定位、concept 占比 vs residual。
- 第二步：加音频（AudioSet 类声音事件，与已有概念对得上）。
- 医学/抑郁：作为下游应用或第二篇；可用 CMU-MOSEI / MELD 等大规模三模态情感数据作为中间桥梁，需要自建临床"状态"概念层。
- 其他场景备选：带概念标注的医学影像（皮肤镜 Derm7pt / SkinCon、胸片报告）。

### 待决定（更新）
- [ ] 是否接受"通用图文先行，抑郁后置"的路线？
- [ ] 可用 GPU 资源（决定能否跑 LLaVA 规模的对齐）。

---

## 2026-09-25 · 第 3 次讨论（算力：4×A6000）

### 用户提供
- 算力：4 张 A6000（每张 48GB，共 192GB，Ampere 架构）。

### 新核实的事实
- 官方仓库：**不提供微调代码**；有 **steerling-8b-instruct** 指令版本（更正第 1 次记录中"仅 base"的说法）。
- bf16 推理约 18GB 显存；上下文 4096；GQA 32 头 / 4 KV 头；known 取 top-k 概念嵌入加权。
- LLaDA-V（CVPR 2026）：LLaDA-8B（扩散 LM）+ SigLIP2 + MLP，用 masked diffusion 做视觉指令微调，**有训练代码**，可作为改 Steerling 训练目标的参考实现。其 Stage 2 用了约 10M 图文数据，远超本项目算力。

### 可行性判断（Claude，基于 FLOPs 的粗估，实际需实测）
- 可行：冻结 Steerling，Stage 1 只训投影层（LLaVA-558K），估计约 1 天；Stage 2 投影层 + LoRA（LLaVA-665K），估计 2–4 天。骨干冻结时多卡只同步少量梯度，PCIe 互联不是瓶颈。
- 不可行：8B 全参微调、LLaDA-V 规模的千万级数据、any-res 高分辨率、视频/全模态。
- 全参微调本来也不需要：会破坏预训练好的概念结构，与"内生可解释"立论冲突。

### 建议的第一阶段计划
- 第 1 周：工程冒烟测试——确认 HF 模型能否接收 inputs_embeds 与自定义块注意力掩码；实现 MDM 损失（参考 LLaDA-V）；图像 576 token = 9 个完整块，文本从块边界开始；1K 样本过拟合测试。
- 第 2 周：Stage 1 小规模（~100K 样本）→ **go/no-go**：图像 token 的 known 概念激活是否与 COCO 物体标签对应；视觉相关 token 的 logit 有多少来自 known/discovered vs epsilon。
- 通过后：全量 Stage 1 → Stage 2（LoRA）→ 概念对齐损失（方法贡献）→ POPE / 概念 steering / 归因评测。

### 待决定（更新）
- [ ] 视觉编码器：CLIP ViT-L/336（LLaVA-1.5 对照最干净）还是 SigLIP2（与 LLaDA-V 对照）。
- [ ] 骨干用 base 还是 instruct。

---

## 2026-09-25 · 第 4 次讨论（澄清当前 idea）

### 当前 idea 的一句话定义
把图像接进 Steerling，并让视觉信息也必须经过它的概念瓶颈（k + u + ε），使模型对图像问题的每个回答都能精确分解为"哪些概念、由图像哪些区域激活"，且可在推理时开关这些概念来因果地改变回答。

### 核心科学问题
从文本学来的概念空间，能否成为视觉输入的内生可解释接口？视觉信息会走概念通路，还是漏进残差 ε？

### 预期贡献
1. 第一个内生可解释（非事后解释）的视觉-语言模型，基于 Steerling。
2. 方法：让视觉信息进入概念通路的训练机制（如概念对齐损失、残差约束）。
3. 评测协议：概念占比、概念-物体对应、概念 steering 的因果效果、幻觉定位。

### 与普通"LLaVA 换骨干"的区别
不比 MMBench 分数，比"解释是否忠实、是否可操作"。

---

## 2026-09-25 · 第 5 次讨论（15 篇相关文献梳理 + 研究空白）

### 用户提出的质疑
"这种 idea 不是已经有很多工作了吗？"——给出 15 篇文献，要求读完后做优缺点/未来工作表，再找空白。
（Claude 阅读深度：15 篇均读了摘要与关键段落，未逐篇精读全文。）

### 文献表（按类型分组）

| # | 论文 | 类型 | 做了什么 | 优点 | 局限 | 未来工作（原文或推断） |
|---|---|---|---|---|---|---|
| 1 | IMA (NeurIPS'24) | 机制分析 | 冻结 LLM 接图/视频/音频；感知 token 与文本 token 表示不同但隐式对齐 | 覆盖多模态；对齐度与幻觉相关 | ≤7B、仅输入拼接式架构 | 原文：更大模型与其他架构 |
| 2 | Concept-Based Explainability for LMMs (NeurIPS'24) | 事后字典 | 对 token 表示做字典学习，得到视觉+文本双向 grounded 的"多模态概念" | 概念可双模态落地 | 事后；不保证概念因果决定输出 | 推断：因果验证、干预 |
| 3 | Same Task, Different Circuits (NeurIPS'25) | 机制分析 | 视觉/文本同任务电路基本不重叠；视觉表示晚层才对齐文本；回贴补掉约 1/3 差距 | 因果电路 + 免训练修复 | 简单任务；逐任务分析 | 推断：训练时让视觉早对齐 |
| 4 | HEIE (CVPR'25) | 输出层解释 | MLLM 输出热力图+分数+CoT 文字解释评 AIGC 瑕疵 | 应用落地 + 数据集 | "解释"是生成文本，非模型机制 | 与本项目相关度低 |
| 5 | VPS (CVPR'25) | 事后归因 | 子区域搜索做 Grounding DINO/Florence-2 归因 | 忠实度大幅提升，有理论 | 绕开内部参数；检测模型非 MLLM | 推断：扩到生成式 MLLM |
| 6 | Cross-modal Information Flow (CVPR'25) | 机制分析 | attention knockout：低层整图→问题 token，中层物体级→问题 token，高层汇到末位 | 阶段清晰 | 仅 LLaVA 系 AR 模型 | 推断：其他架构 |
| 7 | InterSHAP (AAAI'25) | 事后归因 | Shapley 交互指数分离模态贡献与交互，>2 模态，医学数据 | 模型无关、局部解释 | 只到模态级，无概念 | 推断：细粒度 |
| 8 | SAE-V (ICML'25) | 事后字典 | SAE 扩到 MLLM，跨模态特征权重做数据过滤 | 解释→实际收益 | 事后；算力大；大量特征未激活 | 推断：更多模型 |
| 9 | Universal SAE (ICML'25) | 事后字典 | 一个 SAE 同时重建多个视觉模型激活，得通用概念 | 跨模型概念对齐 | 仅视觉模型；事后 | 推断：扩到 MLLM |
| 10 | TAM (ICCV'25 Oral) | 事后归因 | 估计因果推断去除上下文 token 干扰 + 秩高斯滤波，逐 token 视觉解释 | 逐 token 热力图 | 只给"在哪"不给"什么概念"；MLLM 忠实度难评 | 推断：忠实度评测 |
| 11 | Group-sparse SAE (ICLR'26) | 事后字典 | CLIP/CLAP 上 SAE 易学成"分裂字典"；跨模态随机 mask + 组稀疏得多模态字典 | 直面模态分裂，含音频 | 双塔嵌入空间，非生成式 | 推断：扩到 MLLM |
| 12 | MICLIP (ICLR'26) | 事后字典 | 对比学习把内部单元与输入概念、输出语义对齐 | 考虑因果与输出 | 视觉模型；事后 | 推断：扩到 MLLM |
| 13 | PID 分析 (ICLR'26) | 机制分析 | PID 分解冗余/独有/协同信息，26 个 LVLM；视觉指令微调是学会融合的关键阶段 | 规模大 | 输出层信息量，非概念机制 | 推断：指导设计 |
| 14 | Emotion/Rhetoric Neurons (ACL'26) | 神经元干预 | 识别 6 类情绪 + 4 类修辞神经元，自适应 mask 做 steering | 与情感相关、可操控 | 纯文本；神经元多义 | 推断：多模态情感 |
| 15 | Inside-Out (CVPR'26) | 机制分析 | 用电路做无标签泛化度量 | 解释→实用指标 | 仅 ViT | 相关度低 |

### 核心判断
- 15 篇全部是**事后分析/归因**或**输出层文字解释**，没有一篇构建**内生可解释的生成式多模态大模型**。
- 列表外的真正近邻（需纳入相关工作）：CB-LLM（ICLR'25，文本）、MM-CBM（NeurIPS'25 Workshop，CLIP 双 CBL，只做分类/检索）、Concept Bottleneck Generative Models（Ismail et al.，Steerling 团队前作）、Basu et al.（Steerling 临床 steering 失败）、SIM（Barbiero et al. 2026，证实 Steerling 预测确实由少量概念中介且可稀疏化）。

### 研究空白
1. **（主空白）内生可解释的生成式 VLM**：文本有 CB-LLM/Steerling，CLIP 有 MM-CBM，生成式 MLLM 无。
2. **事后方法缺金标准**：内生模型已知真实概念贡献，可作为测试床评估 TAM、SAE-V、字典学习等事后方法能否还原真相。
3. **模态分裂在内生模型里如何解决**：IMA/Same Task/Group-sparse SAE 都发现视觉与文本表示分离、晚对齐或分裂字典——这正是本项目最大风险，也是方法贡献点（训练期强制共享概念空间）。
4. **多模态概念级因果干预评测**：SAE-V、情绪神经元、MICLIP 做了 steering，但都在事后特征上；Basu 显示 Steerling 原版 steering 在临床失败。
5. **（后续）情感/临床的概念级跨模态解释**：InterSHAP 只到模态级，情绪神经元只有文本。

### 待决定
- [ ] 主线选空白 1+3（模型+方法），还是加上空白 2（测试床）作为第二贡献？

---

## 2026-09-25/26 · 第 6 次讨论（15 篇文献逐篇全文精读）

### 用户要求
不能只读摘要和关键段落，必须全文读完；读完一篇写一篇的优缺点和未来工作。

### 完成情况
- 15 篇全部下载全文（含附录；HEIE、Inside-Out 另取 CVPR 补充材料），逐篇精读并立即记录。详细笔记见 `lit_review.md`（每篇：核心问题、方法、实验设置、关键数字、优点、局限、未来工作〔区分原文/推断〕、与本项目的关系），文末附总览对照表、跨论文发现和研究空白。

### 结论摘要
- 15 篇中 0 篇构建内生可解释的多模态模型；13 篇完全不改动模型。
- 证据链：视觉/文本表示天然分离（IMA、SAE-V、Group-sparse SAE）→ 但可被训练纠正（IMA、Group-sparse SAE、USAE、CoX-LMM 附录 G）→ 视觉作用集中在中低层、对齐偏晚（Info Flow、Same Task、PID）→ 融合在指令微调阶段才学到（PID）→ 事后解释无法验证忠实性（TAM、VPS、Emotion Neurons）。
- 研究空白修订为 6 条：内生生成式 MLLM（主线）；在内生模型中解决模态分裂（方法）；忠实性金标准测试床；概念级跨模态分解；带随机对照的概念干预评测；情感/临床场景（后续）。

### 对方法的新增启示
- 图像 token 与配对描述 token 的概念激活施加组稀疏约束 + 共享掩码。
- 输入端 + 输出端双重对齐。
- 考虑中层概念读出/对齐，缓解"概念头只在最后一层"与"视觉作用在中低层"的张力。
- go/no-go 诊断放在 Stage 2 之后。

### 待决定（更新）
- [ ] 第一篇的贡献组合：方法（空白 1+2）为主，是否加入测试床（空白 3）和干预评测（空白 5）？
- [ ] 是否在中层加概念读出（会改动 Steerling 结构，影响"基于 Steerling"的叙事）。

---

## 2026-09-26 · 第 7 次讨论（补读 4 篇近邻工作）

### 完成情况
- 全文精读 MM-CBM、CB-LLM、Concept Bottleneck Generative Models（CBGM）、SIM，笔记已追加到 `lit_review.md` 第 16–19 篇，并附"近邻位置表"和新增方法启示。（SIM 附录 A–E 的逐步数学推导核对了结构与结论，未逐行复算。）

### 关键结论
- 四者确认空白 1：尚无"内生可解释 + 生成式 + 多模态"兼具的工作。脉络：CBGM（图像生成，Steerling 作者前作）→ CB-LLM / Steerling（文本生成）→ MM-CBM（判别式图文）→ 本工作（生成式多模态）。
- CB-LLM 生成版只有 4–14 个类别级概念，Steerling 有 33K 已命名概念 → 选择 Steerling 的理由。
- SIM 独立验证：Steerling 的预测主要由约 500 个已监督概念中介（与 Basu et al. 的 steering 失败结果互补）。
- 可借鉴：CB-LLM 的对抗解耦、CBGM 的正交约束、MM-CBM 的检测器自动概念标签（CBGM：约 20% 标注即可）、SIM 的对称性 II 雅可比子空间检验（分别在文本和图像输入上测）。

---

## 2026-09-26 · 第 8 次讨论（候选方法）

### 用户问题
基于文献调研，现在能提出哪些候选方法？

### Claude 提出的候选（待用户选择）
- **M0 基线 + 诊断**：冻结 Steerling + 投影层（→ LoRA），用 SIM 对称性 II 检验、残差能量占比、概念-物体 AUROC 量化"视觉信息有多少漏进 ε"。作为动机实验与基线，不单独成文。
- **M1 输出侧视觉概念对齐**：对图像 token 的概念激活施加弱监督（检测器标签映射到概念 ID）+ 与配对描述文本概念激活的组稀疏对齐。风险低、增量小；隐患：最后一层的视觉 token 表示可能已不承载信息（Info Flow）。
- **M2 概念化视觉输入（主推）**：投影层把每个图像块表示为"已命名视觉概念嵌入的稀疏组合 + 有预算的残差"，v = W·Σ a_k e_k + r（TopK 稀疏，W 为共享线性映射解决输入/输出空间不匹配）。视觉信息进入 LLM 前即被翻译为已命名概念，结构上保证（SIM：架构编译优于优化）。训练信号：MDM 描述损失 + **文本→视觉概念蒸馏**（冻结 Steerling 读描述得到的概念集合作为图像概念目标，无需人工标注）+ 可选检测器弱标签。应用亮点：幻觉可按构造检测（输出了"狗"但输入概念中没有狗）。
- **M3 泄漏控制组件**：对残差 r / 发现概念通道做对抗解耦（CB-LLM）+ 正交约束（CBGM），并设残差预算，画"可解释性-性能"曲线。与 M1/M2 组合使用。
- **M4 中层概念对齐组件**：在中间层加概念读出/对齐（依据 Same Task、Info Flow、PID），可能同时提升性能；代价是改动 Steerling 结构。
- **M5 忠实性测试床（第二贡献）**：用内生模型的精确概念贡献评估 TAM、VPS、SAE 等事后方法。

### 建议组合
M0（动机）→ M2 + M3（方法主体）→ M5（第二贡献）；M4 视 M0 结果决定是否加入。

### 待决定
- [ ] 主方法选 M2（新颖、风险高）还是 M1（稳妥、增量）？
- [ ] 视觉概念子集的构建方式（33K 中筛出视觉相关概念）。

---

## 2026-09-26 · 第 9 次讨论（确定主方法 M2，细化设计）

### 决定
- **主方法采用 M2（概念化视觉输入）**。组合：M0（动机/基线）→ M2 + M3（泄漏控制）→ M5（忠实性测试床）；M4 视 M0 结果再定。

### 查阅 Steerling 源码后确认的事实（github.com/guidelabs/steerling）
- `forward` 支持 `input_embeds` → 可直接喂入视觉 token。
- 块因果注意力用 flex_attention 实现，块大小 64；576 个图像 token = 9 个完整块。
- 已知概念头：线性预测器（4096 → 33,732）+ sigmoid + top-32，特征 = Σ 权重 × 概念嵌入；**概念嵌入是稠密矩阵 33,732 × 4096，位于最后一层隐空间**；发现概念头为低秩分解（rank 256，top-128）。
- ε 修正使 composed 精确等于 hidden（分解无损），known_features 在计算 unk 时 detach。
- **自带"位置注入"式 steering：在第 ≥16 层把向量加到残差流**（inject_layer=16）→ 概念嵌入在中后层残差流中可直接起作用。
- 仓库为推理包，无训练代码。

### M2 设计（v0）
- 视觉编码器 → 每个图像块 → 概念投影器 → 在视觉概念子集 V_vis 上的 TopK 稀疏权重 a → z = Σ a_k e_k（e_k 为冻结的 Steerling 概念嵌入）；另有受预算约束的残差 r。
- 两个变体，由试点实验决定：
  - **M2-A 输入层**：视觉 token = W·z + r（W 为共享线性映射，每个概念对应一个固定输入向量）。
  - **M2-B 中层注入**：复用 Steerling 自己的 steering 通道，在第 ≥L 层把 α·z 加到图像位置的残差流（与 Info Flow / Same Task 的中层结论一致）。
- 损失：掩码扩散（答案文本）+ 文本→视觉概念蒸馏（冻结 Steerling 读描述得到的概念作为目标，采用排序损失以符合 SIM 对称性 I）+ 残差预算 + 对抗解耦（CB-LLM）+ 正交约束（CBGM）+ 可选检测器弱标签（约 20%）。
- 冻结 Steerling 的概念头与概念嵌入（保证概念语义固定）；Stage 1 只训投影器，Stage 2 加 LoRA。

### 前两周计划
- 第 1 周：环境（Python ≥3.13、CUDA 12.8、flex_attention 在 A6000 上可用性）；加载 steerling-8b-instruct，打通 input_embeds + 块对齐；实现 MDM 损失（参考 LLaDA-V）；1K 样本过拟合；构建 V_vis 与 COCO 类别→概念 ID 映射。
- 第 2 周：同一 10 万样本上跑 M0 基线与 M2-A/M2-B 试点，比较概念占比、对称性 II 检验、概念–物体 AUROC，选定变体。

### 待决定
- [ ] 视觉编码器：CLIP ViT-L/336 还是 SigLIP2。
- [ ] 骨干：base 还是 instruct（建议 instruct）。

---

## 2026-09-26 · 第 10 次讨论（视觉编码器选择）

- 用户问：从快速迭代角度倾向哪个编码器？
- Claude 建议：**先用 CLIP ViT-L/14-336**，代码保持编码器无关，SigLIP2 作为后期消融。理由：LLaVA-1.5 配方与超参最成熟、排错面最小；576 token 恰好 9 个 64-token 块；IMA、Info Flow、TAM、CoX-LMM 等分析工作均基于 CLIP 版 LLaVA，可直接复用协议与对照；CLIP 文本编码器与图像编码器同空间，可快速用于筛选视觉概念子集、生成图像块级概念弱标签、初始化概念投影器。SigLIP2 的优势（更强的密集特征与定位）留到方法跑通后再验证。
- 待用户确认。

---

## 2026-09-26 · 第 11 次讨论（会话交接）

- 用户因上下文长度限制准备新开会话，要求写续接文档，并询问是否需要把 record.md 传给新会话。
- Claude 生成 `HANDOFF.md`（新会话入口：工作约定、目标、已定决策、Steerling 源码事实、文献结论、M2 设计 v0、下一步、待决定、文件清单）。
- 建议：把 HANDOFF.md、record.md、lit_review.md（以及做 V_vis 时需要的 known_concepts.csv）上传到本 Project 的项目文件中，新会话会自动读取；此后每次会话结束用新版 record.md 替换旧版。

---

## 2026-09-26 · 第 12 次讨论（第 1 周工程启动：实验设计 + Runpod 资源）

### 用户提供 / 要求
- 在新会话（Claude Code 云端，已接入 Runpod MCP）上传了 `HANDOFF.md`、`record.md`、`lit_review.md`，要求开始第 1 周工程搭建：描述实验动机与实验设计；如需 GPU，创建 Runpod Pod（1 × A100 SXM；区域 US-CA-2 / US-WA-1 / US-WA-2；Temporary storage 100GB；挂载 global volume；暴露 8888 端口）；更新 record。
- 算力由原先的 4 × A6000 改为 **Runpod 云 GPU**（A100 与 A6000 同为 Ampere 架构，flex_attention 的兼容性问题相同）。

### 完成情况
- 项目文档已放进仓库 `docs/`：`HANDOFF.md`、`record.md`、`lit_review.md`。此后 Claude 直接在仓库里更新 record，不再需要用户手动替换。
- 新建 `docs/week1_plan.md`：实验动机，以及 E1–E7 七项冒烟实验（环境 / flex_attention、骨干加载与分解校验、input_embeds 与块对齐、M0 视觉通路、MDM 损失、1K 过拟合、V_vis 与 COCO 映射），每项都有验收标准，另附存储估算与风险备选。
- 本周默认设定：steerling-8b-instruct（冻结）+ CLIP ViT-L/14-336 + M0 MLP 投影器。编码器和 base/instruct 两项仍是**建议值、待用户确认**，代码会保持与编码器无关。

### Runpod 实测情况（2026-09-26 查询）
- 账号下没有任何 network volume，也没有 Pod。
- A100 SXM 80GB（`NVIDIA A100-SXM4-80GB`）：secure $1.59/h，community $1.39/h；美国区有货的数据中心只有 US-KS-2、US-MD-1、US-WA-1（均为 LOW）。
- Network volume 只能在支持它的数据中心创建，而且 Pod 必须和 volume 在同一个数据中心：
  - US-CA-2：支持 volume（仅 HIGH_PERFORMANCE），但**当前没有 A100 SXM 库存**。
  - US-WA-1、US-WA-2：**不支持 network volume**。
  - 有 A100 SXM 的 US-KS-2、US-MD-1 也不支持 network volume。
- 结论：在指定区域内，"A100 SXM + 挂载 network volume"**目前无法同时满足**。

### 决定与已创建的资源
- 用户选择：**US-WA-1 的 A100 SXM，不挂 network volume**，改用 Pod 自带的持久盘。
- 已创建 Pod `steerling-mm-week1`（id `de3m2w1c58cwx1`）：
  - 1 × A100-SXM4-80GB，secure，US-WA-1，主机 CUDA 12.8，32 vCPU，250GB 内存。
  - 镜像 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`。
  - Container disk 100GB（临时，重启后清空）；持久盘 200GB，挂载在 `/workspace`。
  - 端口：8888/http（Jupyter），22/tcp（SSH）。
  - 价格：GPU $1.59/h；持久盘另按容量计费，Pod 停止后仍收费。
- 访问方式：Jupyter 地址 `https://de3m2w1c58cwx1-8888.proxy.runpod.net`（密码见 Runpod 控制台里该 Pod 的环境变量 `JUPYTER_PASSWORD`，不写进仓库）；SSH 用 `ssh de3m2w1c58cwx1-6441125f@ssh.runpod.io`。
- 注意：持久盘绑定在这台机器上。Pod 停止后再启动，如果这台机器的 GPU 被别人占用，就可能无法启动。权重、数据、检查点要定期同步到别处（如 HF Hub 私有仓库）。

### 待决定
- [x] `known_concepts.csv` 已由用户上传到仓库 `datasets/known_concepts(1).zip`（见下一节）。
- [ ] 视觉编码器与骨干版本的确认（沿用第 10 次讨论的待决事项）。

---

## 2026-09-26 · 第 13 次讨论（known_concepts.csv 入库核对）

### 用户提供
- 已把 `known_concepts.csv` 的压缩包手动提交到仓库：`datasets/known_concepts(1).zip`（压缩后 5.3MB，解压后 15.3MB，内含一个文件 `known_concepts(1).csv`）。

### 核对结果（Claude 解压后逐项统计）
- 33,732 行，`concept_id` 为 0–33731，没有重复；`head` 全部为 known。
- 编码：不是 UTF-8（第 2255 字节起出错），用 latin-1 能正常读取，与第 2 次讨论的记录一致。
- 实际字段：`concept_id`、`concept_name`、`concept_description`、`head`、`public_group_id`、`group_name`、`is_steerable`、`is_tone`、`is_alignment`、`is_demographic`。第 2 次讨论里记的字段名（description / group）是简写，以这里为准。
- 统计与第 2 次讨论完全一致：可 steer 16,327，tone 441，alignment 1,020，demographic 1,347；组数 19,366（记录中写的"约 19K"）；99.7% 的描述以 "Tokens" 开头。
- 结论：确认是同一份官方概念表，可直接用于 E7（构建 V_vis 与 COCO 映射）。

### 待确认
- [ ] E2 中核对 `concept_id` 是否对应已知概念头输出的下标。

---

## 2026-09-26 · 第 14 次讨论（第 1 周工程：E1–E7 在 Pod 上完成）

### 用户决定
- 视觉编码器用 **CLIP ViT-L/14-336**，骨干用 **steerling-8b-instruct**（采纳 Claude 的建议）。
- 要求立即在 Pod 上完成 E1–E7。

### 工程过程中遇到的问题
- 本容器到 Pod 的 SSH（22 端口）被网络策略挡住，改用 Pod 上 Jupyter（8888 端口）的 kernel API 作为控制通道（`tools/pod/podexec.py`，Jupyter 密码从环境变量读取，不入库）。Runpod 代理会拦截 Python 默认的 User-Agent，需要改成 curl 的。
- `/workspace` 是 MooseFS 网络盘，在上面建 venv 会报 Stale file handle。Python 环境改装到容器盘 `/opt`，Pod 重启后重跑 `tools/pod/setup_env.sh`。
- 不锁版本时，依赖会把 torch 升到 2.14+cu130，Pod 驱动 570（CUDA 12.8）用不了。已锁定 torch==2.8.0+cu128，CLIP 改用 transformers 实现。

### 结果（详见 `docs/week1_results.md`，原始数据在 `results/week1/`）
- **E1 ✅**：flex_attention 在 A100 上可用，前向误差 ≤ 0.0078，反向相对误差 0.32%，比 SDPA 回退快 13–29 倍。已知限制：序列长度在 64–128 之间且不是 64 的倍数时（如 106），Triton 共享内存超限；我们的序列都 ≥ 640，不受影响。
- **E2 ✅**：显存 16.8GB。分解 known + unk_hat + ε = hidden，bf16 下相对误差 ≤ 0.32%，fp32 重算完全精确。范数占比：known 0.18、unk_hat 0.98、ε 0.32。**concept_id 就是概念头的行下标**（top token 命中率 75.5%，错位 1 后 4.8%）。chat 模板每轮以 `<|endofchunk|><|eot_id|>` 结束。
- **E3 ✅**：`input_ids` 与 `input_embeds` 两条通路的 logits 完全一致；文本从第 576 位（第 9 块）开始；块内双向、块间因果、文本能看到全部图像块，均已验证。注意：9 个图像块之间也是因果的。
- **E4 ✅**：只有投影器可训练（2098 万参数）；B=8、T=640 时峰值 55.8GB。
- **E5 ✅**：实现了**按块的 MDM 损失**（每个样本选一个答案块 b，之前的块保持干净，b 块内按 t 做 mask，之后的内容丢弃，损失 = Σ CE/t/n_b），与 Steerling 逐块生成的方式一致。这和 LLaDA-V 整段 mask 的做法不同，是一个设计决定。6 个单元测试全部通过。
- **E6 ⚠️ 流程已验证（go），严格过拟合未达到**：
  - 1K 样本 10 个 epoch，t=1 评估损失 9.25→5.32，生成 F1 0.04→0.21。
  - **E6b**：配对正确与打乱配对的损失差，训练集 2.34 nats，未见过的样本 1.44 nats，说明视觉通路带进了图像特有的信息，并且能泛化。
  - 64 样本恒定学习率的记忆测试退化成重复词（贪心并行解码的问题），无法作为判据。
- **E7 ⚠️ 初版可用**：
  - 用概念名称加 CLIP 打分判断"是否视觉概念"不可靠（对照词上 AUROC 0.877）。
  - **新发现：概念必须在被 mask 的位置上读出。**在干净 token 上读，会被代码和字母片段类概念占满，因为 MDM 只在 mask 位置训练。
  - 按 mask 位置读出：V_vis = 4,374 个概念，一共只有 6,093 个（18%）被激活过，有 14 个"枢纽概念"。
  - COCO 映射改用特异性排序（激活率 × IDF）后，约 36 类正确、29 类部分正确、15 类错误（Claude 粗略人工判定）。
  - 对 M2 的启示：文本→视觉概念蒸馏的目标，要用"mask 位置读出 + 特异性"的协议来构造。

### 资源
- Pod `de3m2w1c58cwx1` 仍在运行（$1.59/h）。持久盘 `/workspace` 上有：权重（steerling、CLIP）、LLaVA-558K（zip）、COCO 标注、1K 和 5K 两个子集、`projector_m0.pt`。

### 待决定
- [ ] 人工核对 COCO→概念映射，确定 V_vis v1。
- [ ] 是否把"图像前缀块内全双向注意力"作为 M2 的消融。
- [ ] 第 2 周开始前 Pod 是否先停机（停机后 GPU 不计费，持久盘仍计费）。

---

## 2026-09-26 · 第 15 次讨论（COCO→概念映射合并版 v1）

### 用户要求
- COCO 映射由 Claude 先合并一版，再交给用户审阅。

### 完成情况
- `scripts/e7c_merge_coco_map.py`：对每类的候选池，用三种信号打分：
  - Steerling 在 mask 位置的读出特异性（补充了同义词，如 purse、hair dryer、sofa）；
  - CLIP 文本相似度；
  - 词面匹配（名称命中、LM head token 命中）。

  自动结果中 high confidence 为 47/80。
- `scripts/e7d_review_coco_map.py`：Claude 逐类判定，并修改 10 类的主概念（person、airplane、zebra、skateboard、cup、broccoli、cake、refrigerator、teddy bear、kite）。
- 结果写在 `docs/coco_concept_map_v1.md` 和 `results/week1/e7/coco2concept_v1_reviewed.json`：**✅ 精确 37 / 🟡 相关或上位 42 / ❌ 无 1（bench）**。

### 发现
- 80 类只对应到 62 个不同的概念，有 4 组类别共用一个概念（包、Sk- 运动、刀叉勺、微波炉/烤箱/烤面包机）。概念–物体 AUROC 应按概念合并类别来评测。
- 名称最贴切的概念常常在 mask 位置上从不激活（如 Kite Flying、Toilets and Sanitation、Typing and Keyboards），模型实际用的是另一个近义概念。M2 的蒸馏目标应优先选"模型真正会用的"概念。
- 餐具器皿、运动器材、路边设施在 33K 概念中没有物体级条目，属于概念覆盖的局限。

### 待决定
- [ ] 用户审阅 v1，重点看 42 个 🟡 类别和 10 处修改；kite 用 "Kite Flying"（名称精确，但从不激活）还是 "Drones and UAVs"（会激活，但语义偏）。
- [ ] 共用概念的类别在评测时是否合并。

---

## 2026-09-26 · 第 16 次讨论（回顾 M0–M5；M0 诊断工具与试跑）

### 用户提供 / 要求
- 回顾了 M0–M5 的设计，问：第 1 周的实验算不算 M0 动机实验？能不能可视化？能不能说明动机？
- Claude 的回答：**不算。**第 1 周是工程冒烟测试。现有结果只能说明"管线可行"，说明不了"直接接入会破坏可解释性"。另外，E2 的"known 占 0.18"是在未 mask 的位置上测的，不能当基线。
- 用户要求先做：(1) M0 的测量工具；(2) 用 1K 投影器试跑验证。需要 GPU 就保留 Pod。

### 完成情况（详见 `docs/m0_pilot.md`）
- 工具：对数值精确分解（known、discovered、ε 三项之和等于 logit）；SIM 对称性 II 违反度（对上下文输入嵌入求梯度，看落在前 K 个已命名概念梯度子空间里的比例，并用随机 K 个概念作对照）。都在被 mask 的位置上测。
- 设计：COCO val2017，同一个被 mask 的物体词，信息来源分三种：IMG（图像）、TXT（同一图的另一条描述）、NONE（语言先验）。200 个样本，其中 40 个做 SIM。
- 结果：
  - 已命名概念占 \|logit\|：IMG 78.3%、TXT 81.1%、NONE 75.1%（IMG−TXT = −2.9 个百分点 [−4.3, −1.5]）。
  - log p：IMG −2.97、TXT −1.38、NONE −2.39，**1K 投影器的图像反而拉低了预测**。
  - SIM@128：IMG 0.179、TXT 0.035，但随机对照分别为 0.294 和 0.057，**相对机会水平的改善几乎一样**（39% 对 37%）。

### 结论
- 工具可用，区分度足够；纯文本条件下的 SIM 数值与原文一致。
- 试跑中的图像条件等于噪声，**不能用来讲动机**。
- **SIM 的原始违反度会被上下文维度混杂**，正式 M0 必须用相对机会水平的归一化指标，或匹配上下文规模。
- 正式 M0 需要：Stage 1（10 万样本），并确认 IMG 的 log p 高于 NONE；之后做 Stage 2 LoRA；再用归一化 SIM 和对数值分解来测。

### 资源
- 两项工作都做完后，Pod 停机。数据保留在 200GB 持久盘 `/workspace` 上：权重、LLaVA-558K、COCO val2017 图像和标注、`projector_m0.pt`、全部结果。Python 环境在容器盘上，重启后需要重跑 `tools/pod/setup_env.sh`。

### 待决定
- [ ] 是否开始 Stage 1（10 万样本，约 3–4 小时 A100）。

---

## 2026-09-26 · 第 17 次讨论（核心问题讲解；审稿风险：为什么用 LLaVA 范式）

### 用户提出的问题
1. 目标、期望的假设、现有结果分别是什么？
2. "视觉信息会走已命名概念，还是漏进 discovered 和 ε"具体是什么意思？
3. 审稿人会不会质疑为什么一定用 LLaVA 范式，毕竟这两年出现了不少多模态训练范式？

### Claude 的回答要点
- 目标、假设与判定标准：H1（直接外接后视觉信息绕开已命名概念，作为动机）、H2（M2 恢复概念依赖，作为方法）、H3（按构造检测幻觉，作为应用）。H1 可能不成立，两种叙事都已准备。现有结果**还没有回答 H1**：只说明了工程可行，以及测量方法该怎么定。
- 核心问题的完整讲解已整理成 **`docs/core_question.md`**（Steerling 的三路分解、概念只见过文本、ε 后门、为什么重要、两个测量工具、假设与判定标准、范式选择）。
- **范式质疑：会被问，本质是"漏出是否是 LLaVA 特有的现象"。**
  - 选择范围本身有原则性约束：只能"外接"。原生多模态预训练、无编码器、统一理解与生成等范式都要重训 LLM，会毁掉概念头；也没有训练代码和相应算力。
  - 扩散 LM 的同类工作 LLaDA-V 也用 MLP 投影层；分析类文献（IMA、Info Flow、TAM、CoX-LMM）也基于 LLaVA 系，便于比较。
  - **实验上的应对：M0 至少覆盖两种外接方式**：MLP 投影层（576 个 token）加查询重采样器（64 个查询，正好一个扩散块）。门控交叉注意力作为可选。如果都出现漏出，结论就升级为"事后外接视觉的通用问题"。
  - **M2 定位为接入层的设计原则**（输入端概念瓶颈），可以叠加在任何外接方式上；做一组"M2 + 重采样器"证明两者正交。
  - 视觉编码器 CLIP 与 SigLIP2 的消融保持原计划；局限部分讨论原生范式，并把"原生预训练加概念监督"列为未来工作。
  - 具体文献（Chameleon、Emu3、Transfusion、Fuyu、EVE、Janus、Show-o、Llama 3.2 Vision 等）在引用前逐篇核实。

### 完成情况
- 新增 `docs/core_question.md`。
- 代码：`src/smm/vlm.py` 新增 `QueryResampler`（Perceiver/Q-Former 式：64 个可学习查询，2 层交叉注意力加 FFN，宽度 1024，输出 4096）和 `build_connector('mlp' | 'resampler' | 'resampler<N>')`（N 必须是 64 的倍数，以保持块对齐）。`e4_e6_overfit.py`、`e6b_image_dependence.py`、`m0_diagnostics.py` 都加了接入方式参数。`tests/test_connectors.py` 覆盖形状、块对齐、梯度和 MLP 的逐块独立性；全部 12 个测试通过。

### 计划变更
- 正式 M0 = {MLP 投影层, 查询重采样器} × Stage 1（10 万样本）+ Stage 2（LoRA）× 诊断（对数值分解 + 归一化 SIM + 映射概念命中率）。Stage 1 先训 MLP，再训重采样器；预计总时间比单一接法多 50–100%。

### 待决定
- [ ] 何时开 Pod 开始 Stage 1。
- [ ] 门控交叉注意力是否纳入（视前两种接法的结果）。

---

## 2026-09-27 · 第 18 次讨论（如何读 SIM 图；改进画法）

### 用户问题
- 如何理解 SIM symmetry-II violation 那张图？

### Claude 的解释要点
- **违反度的含义：**预测对上下文的梯度中，不能写成"前 K 个激活概念的梯度组合"的比例。0 表示完全经过已命名概念，1 表示完全绕开。
- **曲线随 K 下降是必然的**（子空间嵌套），本身不是发现。
- **随机概念也能解释很多：**所有概念梯度都共享同一个雅可比矩阵，挤在少数主导方向上，上下文越小越明显。所以必须和随机对照比。
- **试跑的读数：**相对机会水平的改善 IMG 39%、TXT 37%，几乎相同。蓝橙两线的原始差距主要来自上下文规模。TXT 在 K=256 时 0.016，与 SIM 原文一致。IMG 一侧因投影器等于噪声而不可信。结论：工具可用，并暴露了混杂，**还没有回答 H1**。

### 完成情况（用户要求：改，同时保留原来的画法）
- `src/smm/diagnostics.py`：每个 K 都有随机对照（嵌套 256 个随机概念）；新增规模匹配对照 `matched_m`（取 ‖g_f‖ 最大的前 m 个上下文 token，m = 描述 B 的 token 数），并记录 `gf_share`。
- `scripts/m0_diagnostics.py`：新增 `--ks` 参数；图像条件自动传入 m。
- `scripts/m0_analyze.py`：保留原图 `m0_pilot.png`，新增 `m0_sim_normalized.png`（(a) 实线加虚线机会水平；(b) 相对机会水平的改善，含规模匹配；(c) IMG − TXT 配对差，H1 预测小于 0）。兼容旧记录。
- `tests/test_sim.py`（玩具模型），加上原有测试共 15 个，全部通过。新图已用合成数据检查过排版，正式数字要等 Stage 1 之后重跑。

### 待决定
- [ ] 正式 M0 的 K 是否加到 512（与 SIM 原文对齐，时间约翻倍）。

---

## 2026-09-27 · 第 19 次讨论（M0 的 K 维持不变；全模态数据；训练加速与 Stage 1 启动）

### 用户决定 / 提出的问题
1. **M0 的 K 先不加到 512**，维持 (8, 32, 128, 256)。
2. 为以后扩展到全模态（文本、图像、音频、视频）做准备：收集音频、视频概念数据（例如 VGGSound）。
3. 开始下一步实验。
4. 训练过程中陆续问到能否用 Unsloth、DeepSpeed、XTuner、PiSSA/GaLore 加速；能否用 AudioCLIP、VideoCLIP 这类模型。

### 加速方案的结论（均有实测或论证）
- **Unsloth / XTuner：**只支持它们适配过的标准架构（自回归因果 LM），Steerling 是自定义的块因果扩散模型，用不上。XTuner 的打包在我们的场景下收益也很小（样本长度整齐）。
- **DeepSpeed ZeRO-1/2 + Offload：**确实能在单卡上放大显存，但它只卸载**可训练**参数的优化器状态和梯度。我们只训练约 2100 万参数的接入层（约 0.34–0.5GB），显存大头是冻结权重（16.8GB）和激活值（约 39GB），所以几乎省不下。
- **PiSSA / GaLore / DoRA / VeRA：**作用对象是 LLM 的权重，Stage 1 不训练 LLM，所以不适用；到 Stage 2 可以考虑 PiSSA（收敛更快）。但它会调整权重的主方向，可能扰动概念结构，要用 M0 指标和 LoRA 对比。
- **实测一：跳过概念头的快速损失**（composed = hidden，所以数学上等价）。损失一致（差异 0.04%），梯度相差约 1%（bf16），只快约 10%，瓶颈在 8B 骨干本身（`check_fast_loss.json`）。
- **实测二：batch 与激活值检查点**（`bench_batch.json`）：
  - MLP：B=8 为 131 ms/样本，B=12 为 134 ms，开检查点后为 185 ms，所以选 B=8、累积 8 步；
  - 重采样器：B=8 为 46 ms，**B=32 为 40 ms**，B=64 为 40 ms 但要 70GB，所以选 B=32、累积 2 步；
  - 检查点在两种接入方式下都更慢。
- **实测三（关键发现）：flex_attention 的动态形状重编译让训练慢 2.4–2.9 倍。**一出现第二种序列长度，PyTorch 就自动切换到动态形状内核（640 个 token 从 1.00 秒变成 2.39 秒）。改为按长度静态编译（`use_static_flex`，加载时默认启用）后恢复全速（`bench_flex_*.json`）。第一次启动的训练因此重启。

### 全模态数据（详见 `docs/omni_concept_data.md`）
- 已下载：
  - 音频：AudioSet 本体（527 类）、ESC-50、AudioCaps 与 Clotho 的描述文本；
  - 视频：VGGSound 标签表（约 310 类）、Kinetics-700 标注、VATEX 描述、MSR-VTT（视频加描述）。
- FSD50K：HF 镜像对 Pod IP 限流，改从 Zenodo 下载（进行中）。
- 采纳用户建议，文本侧相似度改用模态专用编码器：**CLAP**（`laion/larger_clap_general`，音频；AudioCLIP 在 HF 上没有维护良好的版本）和 **X-CLIP**（`microsoft/xclip-base-patch32`，视频；在 Kinetics-400 上训练，有偏乐观的风险）。LanguageBind 留作将来的全模态统一编码器。
- `scripts/omni_concept_map.py`：沿用 COCO 的协议（mask 位置读出 + 特异性 + 模态文本相似度 + 词面匹配），输出 V_audio、V_video 和各词表的映射。VGGSound 归入音频模态。小规模试跑通过：试跑中 V_audio 632 个、V_video 668 个概念（只用了 128 条描述，正式数字以完整运行为准）。

### 实验进度
- **Stage 1（MLP）已开始**：10 万样本，B=8 × 累积 8，lr 1e-3，约 3.7 小时。
  - 第 0 步：COCO 探针上 IMG −3.21 对 NONE −2.20（差 −1.01，只有 28% 的样本图像更好），这是训练要扭转的起点。
- **Pod 流水线**（`tools/pod/pipeline_after_mlp.sh`）：MLP 完成后，依次运行音频、视频概念映射 → 重采样器 Stage 1 → 两种接入方式的 M0 诊断。
- 新增：`src/smm/coco_probe.py`（诊断和训练评估共用）、`scripts/stage1_train.py`、`enable_block_checkpointing`、`use_static_flex`、`mdm_loss_fast`。

### 待决定
- [ ] Stage 2 用 LoRA 还是 PiSSA（或两者都做并比较对概念结构的影响）。
- [ ] 是否提供 HF_TOKEN（避免 Pod IP 被 Hugging Face 限流）。
- [ ] VGGSound / Kinetics 原始片段何时下载（需要的时候再按分片取）。

---

## 2026-09-27 · 第 20 次讨论（AVoCaDO；HF token）

### 用户提出
1. 如果 X-CLIP 不好用，就换成 AVoCaDO（*An Audiovisual Video Captioner Driven by Temporal Orchestration*，ICLR 2026）。
2. 提供了临时的 Hugging Face token（已设为环境变量）。

### 核实与处理
- **AVoCaDO：**
  - 权重在 `AVoCaDO-Captioner/AVoCaDO`，17.9GB，Apache-2.0，基于 Qwen2.5-Omni-7B 微调。
  - 训练集在 `AVoCaDO-Captioner/training_set`：描述 jsonl 362MB，共 106,959 段；视频 568GB。
  - 项目页 avocado-captioner.github.io，arXiv 2510.10395。
- **它是生成式描述模型，不是对齐嵌入模型**，所以不能在原位置替换 X-CLIP（给不出文本相似度）。定位改为：
  1. **现在：**把训练集描述切句后取 6 万句，并入音频、视频两边的描述语料（已实现，`--av_sentences`）。弥补原有语料"视频描述不写声音、音频描述不写画面"的缺口。
  2. **之后：**用模型给 VGGSound、Kinetics、FSD50K、MSR-VTT 片段生成音视频描述，得到片段级概念，也作为音频、视频版 M2 的蒸馏目标。模型权重正在后台下载。
- **X-CLIP 的去留：**完整映射跑完后做"有 X-CLIP 项"和"无 X-CLIP 项"的对比与人工抽查，再决定。
- **HF token：**当前会话读不到新设的环境变量（需要新会话），所以在 Pod 上用 `hf auth login` 登录。token 存在 Pod 持久盘的 HF 缓存目录里，**没有写进仓库**。用户说明这是临时 token，项目结束后建议吊销。

### 训练进度（MLP Stage 1）
- 第 200 步：留出集上配对与打乱配对的损失差从 0.03 扩大到 0.37；COCO 上 IMG − NONE 从 −1.01 缩小到 −0.40（图像更好的样本比例从 28% 升到 38%）。方向正确，但还没有转正。

---

## 2026-09-27 · 第 21 次讨论（视频模态用 MSR-VTT + MSVD？）

### 用户提供
- Grok 整理的短音频、短视频数据集表格（`Multimodal-Short-Audio-Video-Datasets.md`）。问：考虑到数据集大小和可行性，视频模态用 MSR-VTT 和 MSVD 怎么样？

### 核实
- MSVD：`friedrichor/MSVD`（1.85GB，1,970 个视频，与 MSR-VTT 镜像出自同一维护者），已下载到 Pod。
- **用 ffprobe 抽查，两个镜像的视频都不含音轨**（MSR-VTT 0/12，MSVD 0/12）。
- Grok 表格的修正：MSR-VTT 镜像实际 2.2GB（表中写 6–8GB）；AudioSet 实际标注 527 类（本体有 632 个条目）。

### 结论（Claude）
- **赞成**把 MSR-VTT + MSVD 作为视频模态的**核心概念集与评测集**：可直接下载、合计约 4GB、描述密集（20 条或约 40 条/片段）、是标准基准。已把 MSVD 描述加入视频映射的描述语料。
- **三个局限：**
  1. 没有音轨，只能作为视觉-时序模态；音视频配对另用 VGGSound 或 AVoCaDO 训练集视频分片；
  2. 规模不够训练视频接入层，训练数据另选（AVoCaDO 分片、VATEX、OpenVid 子集）；
  3. 没有细粒度标签，标签→概念评测仍用 Kinetics-700 词表。
- 详见 `docs/omni_concept_data.md` 第 7 节。

### 训练进度（MLP Stage 1）
| 步数 | 留出集损失差（打乱 − 配对） | COCO IMG − NONE | 图像更好的比例 |
|---|---|---|---|
| 0 | 0.03 | −1.01 | 28% |
| 200 | 0.37 | −0.40 | 38% |
| 400 | 0.97 | −0.21 | 42% |

- AVoCaDO 模型（17.9GB）已下载完成；FSD50K 仍在从 Zenodo 下载。
