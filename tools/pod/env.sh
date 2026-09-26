# Source on the pod. Python env lives on the container disk (/opt): the /workspace volume is a
# network filesystem (MooseFS) where uv venv installs fail with "Stale file handle".
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/root/.uv-cache
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
source /opt/src/steerling/.venv/bin/activate 2>/dev/null || true
