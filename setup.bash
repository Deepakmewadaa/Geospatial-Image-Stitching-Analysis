#!/usr/bin/env bash
set -euo pipefail

# IMPORTANT: replace this with your public GitHub repository URL before submission.
REPO_URL="https://github.com/Deepakmewadaa/Geospatial-Image-Stitching-Analysis"

ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

echo "[setup] Creating conda environment: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
conda activate "${ENV_NAME}"

echo "[setup] Installing CUDA PyTorch"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "[setup] Cloning project repository"
TMP_REPO_DIR="$(mktemp -d)"
git clone "${REPO_URL}" "${TMP_REPO_DIR}/repo"
cp -a "${TMP_REPO_DIR}/repo/." .

echo "[setup] Installing Python dependencies"
python -m pip install numpy Pillow pandas transformers accelerate qwen-vl-utils huggingface_hub tokenizers

echo "[setup] Downloading Qwen2-VL-7B-Instruct weights"
python setup_model.py --model-size 7b

echo "[setup] Done"
