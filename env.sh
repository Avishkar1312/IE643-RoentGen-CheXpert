# Source this before any work: `source env.sh`
# Keeps every cache / env / temp file inside this folder, so deleting IE643/ removes everything.
export IE643_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$IE643_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$IE643_ROOT/.cache/uv-python"
export PIP_CACHE_DIR="$IE643_ROOT/.cache/pip"
export HF_HOME="$IE643_ROOT/.cache/hf"
export TORCH_HOME="$IE643_ROOT/.cache/torch"
export MPLCONFIGDIR="$IE643_ROOT/.cache/mpl"
export XDG_CACHE_HOME="$IE643_ROOT/.cache/xdg"
export TMPDIR="$IE643_ROOT/.cache/tmp"
export PYTHONPYCACHEPREFIX="$IE643_ROOT/.cache/pycache"
export PYTHONPATH="$IE643_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CXR_DATA_ROOT="$IE643_ROOT/data"
if [ -f "$IE643_ROOT/.venv/bin/activate" ]; then source "$IE643_ROOT/.venv/bin/activate"; fi
