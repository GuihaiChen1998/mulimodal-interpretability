# Replaces the tail of pipeline_resampler_v2.sh without the AVoCaDO pilot (dropped: user chose the smaller
# VALOR-32K / AVE substitutes). Waits for the running resampler-v2 training, then runs its M0 diagnostics.
source /workspace/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/mm
L=/workspace/logs
until grep -q "^DONE" $L/stage1_resampler_v2.log || grep -q Traceback $L/stage1_resampler_v2.log; do sleep 60; done
echo "[$(date -u +%H:%M)] resampler v2 stage 1 finished"
python scripts/m0_diagnostics.py results/stage1_resampler_v2/connector_stage1_resampler.pt /workspace/data/coco \
  results/week1/e7/coco2concept_v1_reviewed.json results/m0_stage1_resampler_v2 --connector resampler --n 200 --n_sim 40 \
  > $L/m0_resampler_v2.log 2>&1
echo "[$(date -u +%H:%M)] m0 resampler v2 rc=$?"
echo V2_DONE
