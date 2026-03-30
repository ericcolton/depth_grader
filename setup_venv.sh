#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/bin/python3.11

if ! command -v "$PYTHON" &> /dev/null; then
    echo "ERROR: $PYTHON not found. Install with: brew install python@3.11"
    exit 1
fi

$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo ""
echo "Activate the venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the script (MPS fallback recommended on Apple Silicon):"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py <image> [--layers N] [--output PREFIX]"
echo ""
echo "Example:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py photo.jpg --layers 3"
