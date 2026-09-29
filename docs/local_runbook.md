# 本地服务器运行手册（4×A6000）：M0 v2 并行诊断

> 2026-09-27。**先说明：M0 v2 不需要重新训练。**接入层（Stage 1 MLP）只训练一次，已经训好。换目标词（颜色 / 数量 / 空间关系 / 属性）只改变"诊断时 mask 哪个词"，每种类型是一次独立的推理加梯度计算（每个约 1 小时）。这 5 个任务互不依赖，每张卡跑一个即可并行。
> 以后如果要在本地做训练类任务（例如 Stage 2 的不同变体），可以用同一套调度器，只需换一个任务清单。

## 0. 总体流程
```
git pull → setup（一次）→ 下载模型/数据（一次）→ check_env
→ run_jobs（指定显卡，并行）→ m0_v2_report → git push → 在对话里告诉 Claude "结果已推送"
```

## 1. 拉代码
```bash
git clone https://github.com/guihaichen1998/mulimodal-interpretability.git   # 已有仓库就 git pull
cd mulimodal-interpretability
git checkout claude/dazzling-lovelace-7jtf9k
```

## 2. 环境与数据（只做一次）
先修改 `tools/local/local_env.sh` 中的 `MM_DATA`（默认 `~/mm_data`，需要约 25GB）。
```bash
bash tools/local/setup_local.sh      # uv + Steerling 依赖 + torch 2.8.0+cu126（适配 CUDA 12.4 驱动）
bash tools/local/download_local.sh   # Steerling-8B-instruct、CLIP-L/336、COCO val2017
```
- **setup 之后 `torch.cuda.is_available()` 为 False：**说明驱动太旧，改用 torch 2.6：`TORCH_CU=cu124 bash tools/local/setup_local.sh`。M0 只用 SDPA 注意力，2.6 足够。
- **内网访问不了外网时：**
  - Hugging Face：`export HF_ENDPOINT=https://hf-mirror.com`，再运行 `download_local.sh`；
  - PyPI 和 uv：`export UV_DEFAULT_INDEX=<内网 PyPI 镜像>`；
  - PyTorch 官方 wheel 源访问不了时：在能上网的机器下载 whl 后拷过来，用 `uv pip install xxx.whl` 安装。
- 这套环境**不用** vLLM 或 SGLang，和服务器上已有的版本互不影响（独立的 venv，放在 `$MM_DATA/steerling/.venv`）。

## 3. 接入层权重（已在仓库里）
`git pull` 之后就有，不需要额外操作：
- `weights/connector_stage1_mlp.pt`：MLP 接入层，fp32，84MB，md5 `639bb885…`，M0 v2 用这个。`local_env.sh` 默认就指向它。
- `weights/connector_stage1_resampler_v2_bf16.pt`：重采样器 v2（64 个查询，学习率 2e-4），bf16，62MB，以后需要时用。

## 4. 环境检查（每次换机器或换卡时做）
```bash
source tools/local/local_env.sh
python tools/local/check_env.py            # 每张卡做一次 bf16 矩阵乘和 SDPA，报告剩余显存；结果写入 logs/check_env.txt
```
最后一行应该是 `ENV_OK`。

**冒烟测试**（约 3 分钟，只用 1 张卡，确认整条链路能跑通）：
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash -c 'source tools/local/local_env.sh && \
  python scripts/m0_diagnostics.py $MLP_CONNECTOR $COCO_DIR results/week1/e7/coco2concept_v1_reviewed.json \
  /tmp/m0_smoke --n 2 --n_sim 1 --target_type color'
```

## 5. 并行运行：一张卡一个任务，由你指定用哪几张卡
```bash
source tools/local/local_env.sh
python tools/local/run_jobs.py tools/local/jobs/m0_v2.txt --gpus 0,1,2,3 --dry_run   # 先看会怎么分配
nohup python tools/local/run_jobs.py tools/local/jobs/m0_v2.txt --gpus 0,1,2,3 --min_free_gb 40 \
  > logs/run_m0_v2.out 2>&1 &
```
- **`--gpus`：**填 nvidia-smi 里的卡号，例如只用空闲的 1 号和 3 号就写 `--gpus 1,3`；写 `auto` 表示所有卡。
- **卡被别人占用：**每个任务启动前都会重新检查剩余显存，低于 `--min_free_gb` 的卡跳过、等它空出来。任务数多于卡数时自动排队（5 个任务配 4 张卡时，第 5 个等第一张空出来的卡）。
- **卡号对应：**每个任务只看到自己那张卡，卡号和 nvidia-smi 一致（`CUDA_DEVICE_ORDER=PCI_BUS_ID`）。
- **中断后重跑：**同一条命令即可，已成功的任务会跳过；加 `--force` 全部重跑，加 `--only color,count` 只跑指定的任务。
- **看进度：**
  - `cat logs/run_m0_v2.out`
  - `cat logs/m0_v2/status.json`
  - `tail -f logs/m0_v2/color.log`（日志里每 10 个样本打印一行）

**任务清单 `tools/local/jobs/m0_v2.txt`：**object、color、count、spatial、attribute 各 200 个样本，其中 40 个做 SIM 检验。
- object 重复一遍 A100 上的实验，用来确认本地环境得到的数值一致；
- 其余四个是新的目标词类型，词表见 `src/smm/coco_probe.py` 中的 `LEXICON`。

## 6. 汇总并把结果交给 Claude
```bash
source tools/local/local_env.sh
python scripts/m0_v2_report.py       # 生成 results/m0_v2/REPORT.md、summary.json、m0_v2_overview.png、_run_info.md
git add results/m0_v2 && git commit -m "M0 v2 results (local 4xA6000)" && git push
```
然后在对话里说一句"M0 v2 结果已推送"。我会拉取后分析，并更新 docs 和 record。
- 结果都是小文件：每种类型约 300KB 的逐样本记录，加上图和汇总。
- **任务失败时也照样运行 report 并 push。**`_run_info.md` 会自动收集环境检查、任务状态和每个日志的最后 25 行，我据此排查，不需要你手动贴日志。
- **只想快速问一句时：**把 `results/m0_v2/REPORT.md` 的表格直接贴到对话里也可以。

## 预计耗时与资源
- A100 上每个任务约 40 分钟；A6000 预计 60–80 分钟。
- 4 张卡并行时，5 个任务约 2–2.5 小时跑完。
- **显存：**Steerling 8B（bf16，约 16GB）加上 SIM 的反向传播，单卡峰值预计不超过 40GB。所以默认 `--min_free_gb 40`；如果某个任务 OOM，`_run_info.md` 里会看到。
