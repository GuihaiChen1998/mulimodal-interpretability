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
