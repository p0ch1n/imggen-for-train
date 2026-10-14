#!/usr/bin/env bash
# =============================================================================================
#  Run on a machine WITH internet access: bundle the packages + models the isolated env needs, then
#  move the whole BUNDLE_DIR into the isolated env (portable drive / approved transfer channel).
#
#  Usage: PY_VER=3.10 CUDA_TAG=cu124 bash offline/prepare_bundle.sh /path/to/bundle
#    PY_VER   : the isolated env's Python version (python3 --version)
#    CUDA_TAG : the torch wheel CUDA tag matching the isolated env's driver (cu121 / cu124 / cu126 ...)
#    SKIP_*=1 to skip individual items (e.g. SKIP_FLUX=1)
#  Note: FLUX.1-Fill-dev requires accepting the license on HF first and `huggingface-cli login`.
# =============================================================================================
set -euo pipefail
BUNDLE="${1:?bundle dir}"
PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu124}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$BUNDLE"/{wheels,models,code}

echo "== 1. Python wheels (target: Python $PY_VER, linux x86_64, $CUDA_TAG)"
PIP_TARGET=(--platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64
            --python-version "$PY_VER" --only-binary=:all: -d "$BUNDLE/wheels")
pip download "${PIP_TARGET[@]}" torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip download "${PIP_TARGET[@]}" -r "$HERE/requirements.txt" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "== 2. models (huggingface)"
pip install -q -U "huggingface_hub[cli]"
dl() { [ "${2:-0}" = "1" ] && { echo "skip $1"; return; }; huggingface-cli download "$1" --local-dir "$BUNDLE/models/$3" "${@:4}"; }
dl Qwen/Qwen-Image             "${SKIP_QWEN_IMAGE:-0}" Qwen-Image
dl Qwen/Qwen-Image-Edit-2509   "${SKIP_QWEN_EDIT:-0}"  Qwen-Image-Edit-2509
dl lightx2v/Qwen-Image-Lightning "${SKIP_LIGHTNING:-0}" Qwen-Image-Lightning \
   --include "Qwen-Image-Lightning-8steps-V1.1.safetensors" "Qwen-Image-Edit-2509/*8steps*"
dl IDEA-Research/grounding-dino-base "${SKIP_GDINO:-0}" grounding-dino-base
dl black-forest-labs/FLUX.1-Fill-dev "${SKIP_FLUX:-0}" FLUX.1-Fill-dev

echo "== 3. RF-DETR / YOLO weights (instantiating once triggers the download)"
if [ "${SKIP_RFDETR:-0}" != "1" ]; then
  pip install -q rfdetr ultralytics
  (cd "$BUNDLE/models" && python - <<'EOF'
import glob, os, shutil
from rfdetr import RFDETRLarge, RFDETRMedium
RFDETRLarge(); RFDETRMedium()
for f in glob.glob("*.pth") + glob.glob(os.path.expanduser("~/.cache/rfdetr/*.pth")):
    print("weights:", f)
EOF
  )
  (cd "$BUNDLE/models" && python -c "from ultralytics import YOLO; YOLO('yolo12s.pt')")
fi

echo "== 4. code"
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' "$HERE/" "$BUNDLE/code/syngen_od/"

echo "== 5. checksums"
(cd "$BUNDLE" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS)
du -sh "$BUNDLE"/*
echo "DONE. In the isolated env: sha256sum -c SHA256SUMS && pip install --no-index --find-links wheels -r code/syngen_od/requirements.txt"
