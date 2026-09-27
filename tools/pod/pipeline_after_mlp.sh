# Runs after Stage-1 MLP training: omni concept maps -> Stage-1 resampler -> M0 diagnostics (both connectors).
source /workspace/env.sh
export HF_HUB_OFFLINE=1  # all models are cached; the pod IP gets rate-limited by HF for anonymous traffic
cd /workspace/mm
L=/workspace/logs
until grep -q "^DONE" $L/stage1_mlp.log || grep -q Traceback $L/stage1_mlp.log; do sleep 60; done
echo "[$(date -u +%H:%M)] MLP stage 1 finished"
python scripts/omni_concept_map.py audio /workspace/data/omni results/omni > $L/omni_audio.log 2>&1; echo "[$(date -u +%H:%M)] omni audio rc=$?"
python scripts/omni_concept_map.py video /workspace/data/omni results/omni > $L/omni_video.log 2>&1; echo "[$(date -u +%H:%M)] omni video rc=$?"
STEERLING_USE_FLEX_ATTN=1 python scripts/stage1_train.py /workspace/data/llava_100k/data.json /workspace/data/llava_pool5k/data.json \
  results/stage1_resampler --connector resampler --bs 32 --accum 2 --lr 1e-3 --eval_every 200 --workers 8 \
  --coco_dir /workspace/data/coco --coco_map results/week1/e7/coco2concept_v1_reviewed.json > $L/stage1_resampler.log 2>&1
echo "[$(date -u +%H:%M)] resampler stage 1 rc=$?"
for c in mlp resampler; do
  python scripts/m0_diagnostics.py results/stage1_$c/connector_stage1_$c.pt /workspace/data/coco \
    results/week1/e7/coco2concept_v1_reviewed.json results/m0_stage1_$c --connector $c --n 200 --n_sim 40 > $L/m0_$c.log 2>&1
  echo "[$(date -u +%H:%M)] m0 $c rc=$?"
done
echo PIPELINE_DONE
