#!/usr/bin/env bash
# Models (Steerling-8B-instruct ~16 GB, CLIP-L/336 ~1.7 GB), COCO val2017 (images 0.8 GB + annotations).
# The trained MLP connector is fetched separately (see docs/local_runbook.md, step 3).
set -e
source "$(dirname "$0")/local_env.sh"
export HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$COCO_DIR" "$MM_DATA/weights"
hf download guidelabs/steerling-8b-instruct > /dev/null && echo "ok steerling-8b-instruct"
hf download openai/clip-vit-large-patch14-336 > /dev/null && echo "ok clip-vit-large-patch14-336"
cd "$COCO_DIR"
[ -d val2017 ] || { wget -q http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
[ -f annotations/captions_val2017.json ] || { wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip; }
echo "COCO: $(ls val2017 | wc -l) images"
echo DOWNLOAD_DONE
