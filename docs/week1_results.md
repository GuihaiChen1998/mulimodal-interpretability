# 第 1 周结果：工程冒烟测试 E1–E7

> 2026-09-26，Runpod Pod `steerling-mm-week1`（1 × A100-SXM4-80GB，US-WA-1）。
> 计划见 `week1_plan.md`；原始结果 JSON 在 Pod 的 `/workspace/mm/results/week1/`，其中较小的已复制到仓库 `results/week1/`（投影器权重 `projector_m0.pt` 只存放在 Pod 上）。
> 代码：`src/smm/`（模型接线、MDM、生成），`scripts/e*.py`（各项实验），`tests/test_mdm.py`，`tools/pod/`（Pod 环境与控制脚本）。

## 总览

| 实验 | 结论 | 关键数字 |
|---|---|---|
| E1 环境 / flex_attention | ✅ 通过（有一个已知限制） | Python 3.13.8、torch 2.8.0+cu128；flex 与 fp32 参考的前向误差 ≤ 0.0078，反向相对误差 0.32%；前向比 SDPA 回退快 13–29 倍 |
| E2 骨干加载与分解校验 | ✅ 通过 | 显存 16.8GB；known + unk_hat + ε 与 hidden 的相对误差 ≤ 0.32%（bf16），用 fp32 重算为 0；concept_id = 概念头行下标 |
| E3 input_embeds 与块对齐 | ✅ 通过 | 两条通路 logits 完全一致（差 0.0）；文本从第 576 位（第 9 块）开始；块可见性全部符合设计 |
| E4 视觉通路（M0） | ✅ 通过 | 只有投影器可训练（2098 万参数）；B=8、T=640 时前向加反向峰值 55.8GB |
| E5 MDM 损失 | ✅ 通过 | 6 个单元测试在本地和 Pod 上均通过 |
| E6 1K 过拟合 | ⚠️ 流程已验证（go）；严格的"过拟合"标准未达到 | 10 个 epoch：t=1 评估损失 9.25→5.32，生成 F1 0.04→0.21；配对图像与打乱图像的损失差：训练集 2.34 nats，未见过的样本 1.44 nats |
| E7 V_vis 与 COCO 映射 | ⚠️ 初版可用，需人工修订 | 按描述文本读出的 V_vis = 4,374 个概念；COCO 80 类中约 36 类映射正确，29 类部分正确，15 类错误（Claude 粗略人工判定） |

---

## E1 环境与 flex_attention
- **环境装在哪里：**`/workspace` 是网络文件系统（MooseFS），在上面建 venv 会报 `Stale file handle`，所以 Python 环境装在容器盘 `/opt`，Pod 重启后需要重跑 `tools/pod/setup_env.sh`（约 3 分钟）。权重和数据放在 `/workspace`。
- **torch 必须锁版本：**不锁的话，open_clip 等依赖会把 torch 升级到 2.14+cu130，而 Pod 的驱动是 570（CUDA 12.8），这个版本跑不起来。现在用 constraints 锁定 torch==2.8.0，CLIP 改用 transformers 自带的实现。
- **flex_attention 的正确性与速度**（`e1_flex_attention.json`，64-token 块因果、GQA 32/4、head_dim 128）：

| B×T | 前向最大误差 flex / SDPA | 反向相对误差 | flex 前向耗时 | SDPA 前向耗时 |
|---|---|---|---|---|
| 2×640 | 0.0078 / 0.0020 | 0.32% | 0.14 ms | 1.87 ms |
| 2×704 | 0.0039 / 0.0020 | 0.32% | 0.18 ms | 2.46 ms |
| 1×1024 | 0.0039 / 0.0020 | 0.32% | 0.20 ms | 2.41 ms |
| 1×4096 | 0.0078 / 0.0020 | 0.32% | 1.03 ms | 29.5 ms |

- **已知限制：**序列长度在 64 到 128 之间、且不是 64 的倍数时（例如 106），Triton 需要 272KB 共享内存，超过 A100 的 163KB 上限，编译失败（`scripts/e1b_flex_lengths.py`）。42、64、128、192、640、700、704、768 都正常。
  - 影响：Steerling 自带的生成器处理短文本 prompt 时会碰到这个问题。我们的序列都 ≥ 640（576 个图像 token 加文本），不受影响。
  - 做法：训练用 flex，短文本分析用 SDPA。
  - Steerling 默认不开 flex（要设 `STEERLING_USE_FLEX_ATTN=1`）。

## E2 骨干加载与分解校验
- steerling-8b-instruct 加载耗时 37 秒，显存 16.8GB，短序列前向峰值 17.2GB。
- chat 模板格式：`<|start_header_id|>user<|end_header_id|>\n\n{内容}<|endofchunk|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`。每一轮以 `<|endofchunk|><|eot_id|>` 结束，训练时答案的结尾直接从模板推出（`answer_ids`）。
- **分解是否无损**（`e2b_decomp_relerr.json`，3 条文本）：composed 与 hidden 的逐位置最大相对误差为 0.26%–0.32%，是 bf16 舍入造成的（隐状态最大值约 21.6，对应 bf16 的 ulp 约 0.084）。用 fp32 重算 known + unk_hat + ε 时误差为 0，两种方式 argmax 100% 一致。
- **三部分的范数占比**（相对 ‖h‖ 的平均值）：known 0.18，unk_hat 0.98，ε 0.32。known 通路在范数上只占一小部分，这正是 M0 诊断要量化的"信息主要走哪一路"的基线。
- **concept_id 对齐**：随机抽 40 个概念，看概念嵌入经 LM head 最促进的前 15 个 token 在概念名称和描述里出现的比例。同一 id 为 75.5%，把 id 错位 1 后只有 4.8%。结论：CSV 的 `concept_id` 就是概念头的行下标。
- **flex 与 SDPA 的一致性**：同一输入 logits 的最大差为 0.75（logits 最大值约 21.6），argmax 100% 一致。
- **生成**：用 SDPA 能正常作答，例如 "The capital of France is Paris"。

## E3 input_embeds 通路与块对齐
- `input_ids` 和 `input_embeds = tok_emb(input_ids)` 两条通路的 logits 完全相同（最大差 0.0）。
- 576 个图像 token 加 17 个文本 token，文本从第 576 位开始，正好是第 9 块的起点。
- 块可见性（对嵌入加扰动，看隐状态怎么变）：
  - 扰动第 8 个图像块，前 8 块的变化为 0（块间因果）；
  - 扰动第 3 块里的一个 token，同块其他 token 都有变化（最小 0.125，块内双向），前 3 块变化为 0；
  - 扰动第 0 块或第 8 块，所有文本位置都有变化（最小 0.5 和 0.75），说明文本能看到全部图像块。
- 设计上的一点提醒：图像块之间也是**因果**的，第 0 块看不到第 8 块。这和 LLaVA 自回归时的情况一样，但在 M2 里可能值得做"图像前缀块内全双向"的消融。

## E4 视觉通路（M0）
- CLIP ViT-L/14-336 取倒数第二层，去掉 CLS，得到 576×1024；再经 2 层 GELU MLP（fp32 主权重）映射到 576×4096。
- 可训练参数只有投影器的 20,979,712 个，Steerling 的全部参数 `requires_grad=False`。
- 第一个 batch：loss 8.45，序列长 640，投影器各层梯度范数都是有限的非零值。
- 显存：B=8、T=640 时前向加反向峰值 55.8GB。Steerling 不支持 gradient checkpointing，第 2 周如果要更大 batch 或更长序列，需要自己加。

## E5 MDM 损失（`src/smm/mdm.py`，`tests/test_mdm.py`）
- **和 LLaDA-V 的差别（设计决定）：**Steerling 是块因果的，而且逐块生成，去噪一个块时它前面的块都已经是干净的。所以训练时每个样本随机选一个答案块 b：
  - b 之前的答案 token 保持干净；
  - b 块里的答案 token 以概率 t ~ U(0.001, 1] 替换为 mask；
  - b 之后的内容全部丢弃；
  - 损失 = Σ_masked CE / t / n_b（n_b 是 b 块中答案 token 的个数）。
  - 这与推理时的逐块生成一致。LLaDA-V 的做法（整段答案一起按 t mask）适用于全双向模型。
- 单元测试（6 个，全部通过）覆盖：
  - 序列长度是 64 的倍数，图像前缀不计损失；
  - 只有所选块内的答案位置计损失，而且该块是最后一块；
  - 更早的答案块保持干净；
  - mask 比例与 t 一致（误差 < 0.03）；
  - 至少 mask 一个 token，权重等于 1/(t·n_b)；
  - 完美 logits 时损失为 0，随机 logits 时为有限正数。

## E6 1K 过拟合
**主实验**（`e4_e6/`）
- 设置：1K 条 LLaVA-558K 样本，只训 M0 投影器；AdamW，lr 1e-3，3% warmup 后余弦衰减到 0；B=8，10 个 epoch，共 1250 步；flex attention。
- 速度与显存：1.67 秒/步（期间 E7 同时占用 GPU），显存峰值 56.0GB。

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t=1 评估损失（前 256 条训练样本，第一个答案块全 mask） | 9.25 | 7.05 | 6.59 | 6.42 | 6.29 | 5.93 | 5.75 | 5.52 | 5.39 | 5.32 | 5.32 |
| 生成 token-F1（16 条训练样本） | 0.04 | 0.09 | | | | 0.16 | | | | | 0.21 |

- 训练损失（随机 t，每 125 步的均值）：8.78 → 8.51 → 7.22 → … → 3.91。
- 生成样例（第 10 个 epoch，见 `gens_ep10.json`）：图像内容开始起作用。例如参考 "a map of the district of elgin county with areas in blue"，生成 "a map of the city of the island, in the province of …"；参考 "a patio furniture set sitting on a patio floor"，生成 "a set of 3 dining chairs … for the living"。
- 小问题：第 1 轮评估时解码只在 `<|eot_id|>` 停止，所以输出里混进了 `<|endofchunk|>`。已修正为遇到 endofchunk、eot、endoftext 任一即停止，并跳过特殊 token 解码。第 1 轮的 F1 因此略被低估。

**严格过拟合测试**（`e6_mem64/`，64 条样本，恒定 lr 1e-3，计划 60 个 epoch，跑到第 34 个 epoch 时中止）
- t=1 评估损失停在 3.8 左右，训练损失均值 11.2→5.3。
- 第 20 个 epoch 的生成退化成重复词，例如 "of of of …"、"company company …"。
- 判断：恒定高学习率加上我写的贪心并行解码（每一步都取置信度最高的位置、不加重复惩罚），容易在多个位置提交同一个 token。这个测试**无法区分是流程有 bug 还是能力不够**，所以改用 E6b。

**E6b：图像依赖性检验**（`e6b_image_dependence.json`，更直接地判断流程是否正确）
- 做法：用训练好的投影器，计算 t=1 评估损失，比较"配对正确的图像"和"打乱配对的图像"。

| 数据 | 配对正确 | 打乱配对 | 差距 |
|---|---|---|---|
| 训练集 256 条 | 5.32 | 7.65 | **2.34** |
| 未见过的样本 256 条 | 6.19 | 7.62 | **1.44** |

- 结论：投影器把**这张图特有**的信息送进了 Steerling，而且能泛化到没见过的图。"图像 → 投影器 → input_embeds → 块因果注意力 → MDM 损失"这条链路是通的，**第 2 周试点可以开始（go）**。
- 没有达到"复现训练描述"的原因主要是容量和步数：只训 2100 万参数的投影器，Steerling 完全冻结，10 个 epoch 且学习率衰减到 0。另外 LLaVA-558K 的描述噪声很大，比如 "yusuft husaih was the new face of chennai's ip league squad" 这种。第 2 周的 Stage 1 本来就要用 10 万样本，不再追求 1K 过拟合。
- 第 2 周要改进的解码：参考 Steerling 官方生成器，使用熵采样、top-p 和重复惩罚。

## E7 V_vis 与 COCO 映射
一共尝试了两种方法。第二种是本周的主要结论。

**E7a：CLIP 打分（`e7_build_vvis.py`）**
- 做法：先用关键词规则过滤，保留 28,515 个概念；再用 "a photo of {概念名}" 在 5,000 张 LLaVA 图上检索，得到"视觉程度"分数。
- 问题：在对照词上的 AUROC 只有 0.877，以视觉对照词第 10 百分位为阈值时，有 32.5% 的非视觉对照词也会通过，最后得到 19,201 个概念，过于宽松。
- 分数最高的反而是 "Microcontroller GPIO Pin Control" 这类名称很具体的概念。结论：**概念名称加 CLIP 不是好的"视觉程度"信号**。
- 这一版的 COCO 映射还依赖 token 字面匹配，多义词错配很多：tie 对到 "Economic Interdependence"，mouse 对到 "Letter J"，sink 对到 "Carbon sinks"。

**E7b：让 Steerling 自己读描述（`e7b_caption_concepts.py`）**
- **读出位置很关键（新发现）：**起初在**干净**的描述 token 上读概念，结果被代码语法、字母片段这类概念占满。例如 person 对到 "Code Syntax and Numerical Computation"，见 `e7b_clean_readout_summary.json`。
  - 原因：Steerling 是掩码扩散模型，只在**被 mask 的位置**上学预测，干净位置的概念读出没有意义。
  - 改为先 mask 目标 token，再读取模型用来预测它的概念。
  - **这对 M2 很重要：**"文本→视觉概念蒸馏"的目标必须在 mask 位置上读，不能直接读干净描述。
- **V_vis（按描述读出）：**
  - 用 15,000 条描述（COCO val 10K + LLaVA 5K），每条随机 mask 30% 的 token，做两轮。
  - 统计 mask 位置上 sigmoid 权重 ≥ 0.1 的 top-32 概念，出现在 ≥ 5 条描述中、且不在一半以上描述中都出现的，计入 V_vis。
  - 结果：**4,374 个概念**，其中可 steer 的 2,589 个，与 E7a 重叠 2,727 个。
  - 一共被激活过的概念只有 6,093 个，只占 33,732 个的 18%。在一半以上描述中都出现的"枢纽概念"有 14 个，例如 "Animal Species"、"Image Caption Scene Descriptions"、"Color names and shades"。
  - 高频概念里仍有大量语法和标点类概念，比如 "Past-tense -ed verb endings" 和 "Punctuation and Delimiter Symbols"。第 2 周要把 E7a 的规则过滤叠加上去，或者只统计名词位置。
- **COCO 映射（按描述读出）：**
  - 做法：对每个类别取 60 条包含该词的 COCO 描述，mask 掉这个词，读出该位置的概念。
  - 如果按激活总量排序，枢纽概念会胜出，例如 toilet、keyboard 都对到 "Animal Species"。改用**特异性**排序（该类别下的激活率 × IDF）。
  - 结果：79/80 类有映射（sports ball 在描述里没有出现），主概念在该类描述中的平均出现率为 0.70。
  - Claude 粗略人工判定：约 36 类正确，例如 mouse→"Computer mouse and pointers"、sink→"Bathrooms and Bathing Fixtures"、toothbrush→"Toothbrushing and Flossing Technique"、cell phone→"Phones and Personal Devices"。约 29 类部分正确（上位或相关概念），约 15 类错误，例如 train→"Training (human and ML)"、dining table→"SQL and database programming"、bicycle→"Bus (vehicle and software)"、handbag 和 hair drier 只有 1 条描述。
- **第 2 周前需要人工核对 `coco2concept_caption.json`**（每类给出了前 5 个候选），也可以把 E7a 的 CLIP 相似度作为第二信号合并使用。

**对研究的启示**
- 在 mask 位置上，known 概念的激活被少数枢纽概念主导，只有 18% 的概念会被激活过。这和 Basu et al. 报告的"99.92% 的概念激活 < 0.01"、以及 SIM 报告的"约 500 个概念起中介作用"是一致的。
- 类别特异的物体概念大多存在，但需要用对比性的排序才能把它们找出来。M2 的蒸馏目标和概念–物体 AUROC 评测都应该采用"mask 位置读出 + 特异性"这套协议。

---

## 第 2 周前需要处理的事项
1. 人工核对 COCO 映射，确定 V_vis v1（规则过滤 ∩ 按描述读出）。
2. 为 Steerling 加 gradient checkpointing（HF 封装里 `supports_gradient_checkpointing=False`），为更大 batch 做准备。
3. 解压 LLaVA-558K 全量（27GB 的 zip，目前只抽取了 6K 张），准备 10 万样本的试点。
4. 消融候选：图像前缀块内全双向（目前 9 个图像块之间是因果的）。
