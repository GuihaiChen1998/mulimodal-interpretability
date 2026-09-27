# Resampler v2: the ablation picked 64 queries at lr 2e-4 (lr 1e-3 collapses; 128 queries fail the COCO probe).
# Full Stage-1 run (1 epoch, same data as the MLP) -> M0 diagnostics -> AVoCaDO window-caption pilot (50 clips).
source /workspace/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/mm
L=/workspace/logs
STEERLING_USE_FLEX_ATTN=1 python scripts/stage1_train.py /workspace/data/llava_100k/data.json /workspace/data/llava_pool5k/data.json \
  results/stage1_resampler_v2 --connector resampler --bs 32 --accum 2 --lr 2e-4 --eval_every 200 --workers 8 \
  --coco_dir /workspace/data/coco --coco_map results/week1/e7/coco2concept_v1_reviewed.json > $L/stage1_resampler_v2.log 2>&1
echo "[$(date -u +%H:%M)] resampler v2 stage 1 rc=$?"
python scripts/m0_diagnostics.py results/stage1_resampler_v2/connector_stage1_resampler.pt /workspace/data/coco \
  results/week1/e7/coco2concept_v1_reviewed.json results/m0_stage1_resampler_v2 --connector resampler --n 200 --n_sim 40 \
  > $L/m0_resampler_v2.log 2>&1
echo "[$(date -u +%H:%M)] m0 resampler v2 rc=$?"
python scripts/avocado_window_captions.py /workspace/data/omni/av/avocado/videos results/avocado_pilot/windows_6s.jsonl \
  --n_clips 50 --win 6 --hop 3 > $L/avocado_pilot.log 2>&1
echo "[$(date -u +%H:%M)] avocado pilot rc=$?"
echo V2_DONE
