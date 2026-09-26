# Recreate the Python env after a pod restart (container disk is wiped). ~3 min.
set -e
mkdir -p /opt/src && [ -d /opt/src/steerling ] || git clone -q https://github.com/guidelabs/steerling.git /opt/src/steerling
source /workspace/env.sh
cd /opt/src/steerling
rm -rf .venv
uv sync --python 3.13 --extra notebook --extra test          # torch 2.8.0+cu128 from the lockfile
source .venv/bin/activate
# Pin torch: unpinned extras (e.g. open_clip) pull torch 2.14+cu130, which the pod's driver (570, CUDA 12.8) cannot run.
echo "torch==2.8.0" > /tmp/constraints.txt
uv pip install -c /tmp/constraints.txt pillow datasets accelerate peft hf_transfer
python -c "import sys,torch,steerling,transformers;print(sys.version);print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),transformers.__version__)"
echo SETUP_DONE
