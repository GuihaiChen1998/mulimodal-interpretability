# Paths for the local server. Edit once; every job sources this file.
export MM_DATA=${MM_DATA:-$HOME/mm_data}                       # models cache + COCO
export HF_HOME=${HF_HOME:-$MM_DATA/hf}
export COCO_DIR=${COCO_DIR:-$MM_DATA/coco}                     # contains val2017/ and annotations/
MM_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export MLP_CONNECTOR=${MLP_CONNECTOR:-$MM_REPO/weights/connector_stage1_mlp.pt}   # in the repo (git pull)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}                     # models are downloaded once by download_local.sh
export STEERLING_USE_FLEX_ATTN=0                               # M0 uses SDPA; flex is only needed for training
MM_VENV=${MM_VENV:-$MM_DATA/steerling/.venv}
[ -f "$MM_VENV/bin/activate" ] && source "$MM_VENV/bin/activate"
