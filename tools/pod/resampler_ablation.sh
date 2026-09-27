# Stage-1 resampler did not make IMG beat NONE on the COCO probe (IMG-NONE -0.25 after 1 epoch, lr 1e-3).
# Short ablation (400 optimizer steps = 25.6K samples, same effective batch 64) over learning rate and
# number of queries, then the AVoCaDO window-caption pilot. Waits for the M0 pipeline to free the GPU.
source /workspace/env.sh
export HF_HUB_OFFLINE=1 STEERLING_USE_FLEX_ATTN=1
cd /workspace/mm
L=/workspace/logs
until grep -q PIPELINE_DONE $L/pipeline.log; do sleep 120; done
python -c "import json;d=json.load(open('/workspace/data/llava_100k/data.json'));json.dump(d[:25600],open('/workspace/data/llava_100k/sub25k.json','w'))"
for cfg in "resampler 1e-3" "resampler 2e-4" "resampler128 1e-3" "resampler128 2e-4"; do
  set -- $cfg; out=results/abl_resampler/$1_lr$2
  python scripts/stage1_train.py /workspace/data/llava_100k/sub25k.json /workspace/data/llava_pool5k/data.json \
    $out --connector $1 --bs 32 --accum 2 --lr $2 --eval_every 200 --workers 8 \
    --coco_dir /workspace/data/coco --coco_map results/week1/e7/coco2concept_v1_reviewed.json > $L/abl_$1_lr$2.log 2>&1
  echo "[$(date -u +%H:%M)] ablation $1 lr $2 rc=$?"
done
echo ABLATION_DONE
python scripts/avocado_window_captions.py /workspace/data/omni/av/avocado/videos results/avocado_pilot/windows_6s.jsonl \
  --n_clips 50 --win 6 --hop 3 > $L/avocado_pilot.log 2>&1
echo "[$(date -u +%H:%M)] avocado pilot rc=$?"
echo ALL_DONE
