# After the Stage-1 / M0 pipeline: AVoCaDO window-captioning pilot on 50 clips of shard 000 (timing measurement).
source /workspace/env.sh
cd /workspace/mm
L=/workspace/logs
until grep -q PIPELINE_DONE $L/pipeline.log; do sleep 120; done
until grep -q SHARD_DONE $L/avocado_shard.log; do sleep 60; done
python scripts/avocado_window_captions.py /workspace/data/omni/av/avocado/videos results/avocado_pilot/windows_6s.jsonl \
  --n_clips 50 --win 6 --hop 3 > $L/avocado_pilot.log 2>&1
echo "[$(date -u +%H:%M)] avocado pilot rc=$?"
echo ALL_DONE
