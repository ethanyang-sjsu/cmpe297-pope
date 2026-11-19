#!/usr/bin/env bash
# Phase 1: conda env "cmpe297" for Qwen2.5-VL + lmms-eval POPE experiments. Idempotent.
# Run inside WSL (Ubuntu-22.04):  bash setup/01_conda_env.sh
# Leaves the CMPE 249 env (surroundocc) and ~/.bashrc untouched: project variables
# (CMPE297_DATA, HF_HOME) are stored on the conda env and set only while it is active.
set -euxo pipefail

DATA_ROOT=${CMPE297_DATA:-$HOME/cmpe297/data}
HF_CACHE=${HF_HOME_OVERRIDE:-$HOME/cmpe297/hf_cache}
mkdir -p "$DATA_ROOT" "$HF_CACHE"

if [ ! -d ~/miniconda3 ]; then
  wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p ~/miniconda3
  ~/miniconda3/bin/conda init bash
fi
source ~/miniconda3/etc/profile.d/conda.sh
# newer conda requires explicit ToS acceptance for the default channels (free for edu use)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda env list | grep -q '^cmpe297 ' || conda create -n cmpe297 python=3.11 -y
conda activate cmpe297

conda env config vars set CMPE297_DATA="$DATA_ROOT" HF_HOME="$HF_CACHE"

pip install --upgrade pip
# the multi-hundred-MB CUDA wheels drop mid-download on flaky links; let pip resume them
export PIP_RESUME_RETRIES=50 PIP_DEFAULT_TIMEOUT=120
# CUDA 12.4 wheels bundle their own runtime; the RTX 4080 (sm_89) is supported natively
python -c "import torch" 2>/dev/null || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install lmms-eval qwen-vl-utils "transformers>=4.49" accelerate \
    pycocotools pandas numpy scipy matplotlib pillow tqdm pyyaml \
    jupyter ipykernel ipywidgets pytest nbconvert

python -m ipykernel install --user --name cmpe297 --display-name "cmpe297 (py311)"

python - <<'EOF'
import torch, transformers, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("lmms-eval", md.version("lmms-eval"))
EOF

echo "01_conda_env DONE"
