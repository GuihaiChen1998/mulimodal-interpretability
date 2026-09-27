#!/usr/bin/env bash
# One-time Python environment for the local server (no root needed). ~5-10 min.
# Uses uv and the Steerling repo's lockfile, then swaps torch for a build that runs on a CUDA 12.4 driver.
set -e
source "$(dirname "$0")/local_env.sh"
mkdir -p "$MM_DATA"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d "$MM_DATA/steerling" ] || git clone -q https://github.com/guidelabs/steerling.git "$MM_DATA/steerling"
cd "$MM_DATA/steerling"
uv sync --python 3.13 --extra notebook --extra test
source .venv/bin/activate
# The lockfile pins torch 2.8.0+cu128, which needs a CUDA >= 12.8 driver. torch 2.8.0+cu126 runs on 12.4-era
# drivers through CUDA minor-version compatibility; if check_env.py still fails, set TORCH_CU=cu124 to get
# torch 2.6.0+cu124 (M0 only uses SDPA attention, which 2.6 supports).
TORCH_CU=${TORCH_CU:-cu126}
if [ "$TORCH_CU" = cu124 ]; then
  uv pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
else
  uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/$TORCH_CU
fi
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
echo "torch==$TV" > /tmp/mm_constraints.txt
uv pip install -c /tmp/mm_constraints.txt pillow datasets accelerate peft hf_transfer matplotlib tiktoken
python -c "import sys,torch,transformers;print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)"
echo SETUP_DONE
