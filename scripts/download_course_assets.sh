#!/usr/bin/env bash
set -euo pipefail

# Download/preparation helper for the AIOS course.
#
# It intentionally keeps bulk-download routing explicit:
#   --route direct   : unset proxy variables for each download/install command
#   --route inherit  : use current shell proxy variables
#   --route proxy    : use --proxy-url only for each download/install command
#
# Example:
#   ./scripts/download_course_assets.sh --all --route direct
#   ./scripts/download_course_assets.sh --model Qwen/Qwen3-0.6B --route proxy --proxy-url http://127.0.0.1:7898
#   ./scripts/download_course_assets.sh --install-flashinfer --route inherit
#   ./scripts/download_course_assets.sh --install-cuda-dev --route direct

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/home/codex/ai/venvs/minimind}"
PYTHON="${PYTHON:-$VENV/bin/python}"
HF_HOME_DIR="${HF_HOME:-/home/codex/ai/cache/huggingface}"
MODEL_LINK_DIR="${MODEL_LINK_DIR:-/home/codex/ai/models}"

ROUTE="direct"
PROXY_URL=""
MODELS=()
INSTALL_DEPS=0
INSTALL_FLASHINFER=0
INSTALL_CUDA_DEV=0
INSTALL_AIOS=1
YES=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_course_assets.sh [options]

Options:
  --all                     Prepare default runtime assets: Qwen3-0.6B + flashinfer + editable AIOS.
  --model REPO_ID           Download/cache a Hugging Face model. Can repeat.
                            Default runtime model: Qwen/Qwen3-0.6B
  --install-flashinfer      Install flashinfer-python==0.5.3 into the venv.
  --install-cuda-dev        Install CUDA 12.8 nvcc and development headers from NVIDIA apt repo.
                            Needed by FlashInfer JIT kernels.
  --install-deps            Install AIOS Python deps from pyproject.toml into the venv.
                            This may try to install torch/flashinfer again; use carefully.
  --no-install-aios         Do not run pip install --no-deps -e .
  --route direct            Unset proxy variables for command. Good when direct is fastest.
  --route inherit           Inherit current shell proxy variables.
  --route proxy             Use --proxy-url for command only.
  --proxy-url URL           Proxy URL for --route proxy, e.g. http://127.0.0.1:7898.
  --yes                     Do not pause before large downloads/installs.
  -h, --help                Show this help.

Notes:
  - Large model files are stored in HF cache: /home/codex/ai/cache/huggingface
  - Local model symlinks are created under: /home/codex/ai/models
  - This script never changes global shell proxy settings.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODELS+=("Qwen/Qwen3-0.6B")
      INSTALL_FLASHINFER=1
      shift
      ;;
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --install-flashinfer)
      INSTALL_FLASHINFER=1
      shift
      ;;
    --install-cuda-dev)
      INSTALL_CUDA_DEV=1
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --no-install-aios)
      INSTALL_AIOS=0
      shift
      ;;
    --route)
      ROUTE="$2"
      shift 2
      ;;
    --proxy-url)
      PROXY_URL="$2"
      shift 2
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 && "$INSTALL_FLASHINFER" -eq 0 && "$INSTALL_CUDA_DEV" -eq 0 && "$INSTALL_DEPS" -eq 0 && "$INSTALL_AIOS" -eq 0 ]]; then
  MODELS+=("Qwen/Qwen3-0.6B")
fi

case "$ROUTE" in
  direct|inherit|proxy) ;;
  *)
    echo "--route must be direct, inherit, or proxy" >&2
    exit 2
    ;;
esac

if [[ "$ROUTE" == "proxy" && -z "$PROXY_URL" ]]; then
  echo "--route proxy requires --proxy-url" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found or not executable: $PYTHON" >&2
  exit 1
fi

print_proxy_state() {
  echo "[proxy env]"
  env | grep -Ei '^(http|https|all)_proxy=' | sort || true
  echo
}

guard_report() {
  local kind="$1"
  local source="$2"
  local estimate="$3"
  local target="$4"
  local route_text="$5"
  local paid="$6"
  local resume="$7"

  cat <<EOF
============================================================
Download Guard
kind:          $kind
source:        $source
estimated:     $estimate
target:        $target
route:         $route_text
paid proxy:    $paid
resume/cache:  $resume
============================================================
EOF
  if [[ "$YES" -ne 1 ]]; then
    read -r -p "Switch proxy if needed, then press Enter to continue; Ctrl-C to abort. " _
  fi
}

run_with_route() {
  case "$ROUTE" in
    direct)
      env \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        "$@"
      ;;
    inherit)
      "$@"
      ;;
    proxy)
      env \
        HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL" \
        http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL" \
        NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1" \
        "$@"
      ;;
  esac
}

route_summary() {
  case "$ROUTE" in
    direct) echo "DIRECT, proxy variables unset for command" ;;
    inherit) echo "INHERIT current shell env" ;;
    proxy) echo "PER-COMMAND proxy: $PROXY_URL" ;;
  esac
}

paid_proxy_summary() {
  case "$ROUTE" in
    direct) echo "No, assuming direct network works" ;;
    inherit)
      if env | grep -Eiq '^(http|https|all)_proxy=.*7897'; then
        echo "Likely yes: current env includes 7897"
      else
        echo "Depends on current shell env"
      fi
      ;;
    proxy) echo "Depends on the specified proxy subscription" ;;
  esac
}

safe_name_for_model() {
  "$PYTHON" - "$1" <<'PY'
import re, sys
repo = sys.argv[1]
name = repo.split("/")[-1]
name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
print(name)
PY
}

download_model() {
  local model="$1"
  local short_name
  short_name="$(safe_name_for_model "$model")"
  local link_path="$MODEL_LINK_DIR/$short_name"

  guard_report \
    "Hugging Face model" \
    "$model" \
    "Qwen/Qwen3-0.6B is about 1.5GB in local cache; larger Qwen models scale up quickly" \
    "HF_HOME=$HF_HOME_DIR; symlink=$link_path" \
    "$(route_summary)" \
    "$(paid_proxy_summary)" \
    "huggingface_hub snapshot cache resumes/deduplicates blobs"

  mkdir -p "$MODEL_LINK_DIR" "$HF_HOME_DIR"
  run_with_route \
    HF_HOME="$HF_HOME_DIR" \
    HF_HUB_CACHE="$HF_HOME_DIR/hub" \
    "$PYTHON" - "$model" "$link_path" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
link_path = Path(sys.argv[2])
snapshot = snapshot_download(repo_id=repo_id, resume_download=True)
link_path.parent.mkdir(parents=True, exist_ok=True)
if link_path.exists() or link_path.is_symlink():
    link_path.unlink()
link_path.symlink_to(snapshot, target_is_directory=True)
print("snapshot:", snapshot)
print("symlink:", link_path, "->", link_path.resolve())
PY
}

install_aios_editable() {
  echo "[install] AIOS editable package, no dependency downloads"
  "$PYTHON" -m pip install --no-deps -e "$ROOT_DIR"
}

install_deps() {
  guard_report \
    "Python dependencies" \
    "PyPI via pip install -e ." \
    "Usually hundreds of MB if torch/CUDA wheels are missing" \
    "$VENV" \
    "$(route_summary)" \
    "$(paid_proxy_summary)" \
    "pip cache resumes some wheels; failed installs can be rerun"

  run_with_route "$PYTHON" -m pip install -e "$ROOT_DIR"
}

install_flashinfer() {
  guard_report \
    "Python CUDA wheel" \
    "PyPI: flashinfer-python==0.5.3" \
    "Unknown exact wheel size; likely tens to hundreds of MB" \
    "$VENV" \
    "$(route_summary)" \
    "$(paid_proxy_summary)" \
    "pip wheel cache; rerun is safe"

  run_with_route "$PYTHON" -m pip install "flashinfer-python==0.5.3"
}

install_cuda_dev() {
  guard_report \
    "CUDA compiler and development headers" \
    "NVIDIA apt repo: cuda-nvcc-12-8, cuda-libraries-dev-12-8, libcublas-dev-12-8" \
    "About 2.6GB download total on a fresh system; about 6.5GB installed" \
    "/usr/local/cuda-12.8 and /usr/local/cuda" \
    "$(route_summary)" \
    "$(paid_proxy_summary)" \
    "apt caches downloaded .deb files; rerun resumes through apt cache"

  local keyring=/home/codex/ai/tmp/cuda-keyring_1.1-1_all.deb
  mkdir -p /home/codex/ai/tmp
  run_with_route wget -O "$keyring" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo -n dpkg -i "$keyring"
  run_with_route sudo -n apt-get update
  run_with_route sudo -n apt-get install -y \
    cuda-nvcc-12-8 \
    libcublas-dev-12-8 \
    cuda-libraries-dev-12-8
}

main() {
  echo "AIOS root: $ROOT_DIR"
  echo "Python:    $PYTHON"
  echo "HF_HOME:   $HF_HOME_DIR"
  echo "Route:     $(route_summary)"
  print_proxy_state

  for model in "${MODELS[@]}"; do
    download_model "$model"
  done

  if [[ "$INSTALL_AIOS" -eq 1 ]]; then
    install_aios_editable
  fi
  if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    install_deps
  fi
  if [[ "$INSTALL_FLASHINFER" -eq 1 ]]; then
    install_flashinfer
  fi
  if [[ "$INSTALL_CUDA_DEV" -eq 1 ]]; then
    install_cuda_dev
  fi

  echo
  echo "Done. Quick checks:"
  echo "  $PYTHON -c 'import aios; print(aios.__file__)'"
  echo "  source /home/codex/ai/projects/aios/scripts/activate_aios.sh"
  echo "  $PYTHON /home/codex/ai/projects/aios/resources/lesson-2-run-qwen3/run_qwen3.py --model /home/codex/ai/models/Qwen3-0.6B --prompt 你好 --max-tokens 4 --temperature 0 --device cuda:0"
}

main "$@"
