# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`depth_grader` is a Python CLI tool that takes an input image and generates N cumulative alpha-only PNG masks using MiDaS monocular depth estimation. The masks represent progressively deeper layers of a scene, intended for layered animation and compositing workflows.

## Setup

Requires Python 3.11 via Homebrew (`/opt/homebrew/bin/python3.11`):

```bash
bash setup_venv.sh
source .venv/bin/activate
```

First run downloads ~400MB of MiDaS DPT_Large model weights to `~/.cache/torch/hub/`.

## Running

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py <image> [--layers N] [--output PREFIX]
```

- `--layers N` (default: 1): number of depth layer masks to generate
- `--output PREFIX`: output filename prefix (default: input filename stem)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is required on Apple Silicon for MPS operations that fall back to CPU

Output files are named `<PREFIX>_layer_1.png`, `<PREFIX>_layer_2.png`, etc. — RGBA PNGs where R=G=B=0 and only the alpha channel carries data (255=opaque, 0=transparent). Each layer is cumulative: layer N includes everything from layer N-1 plus the next depth band.

There is no test suite or linting setup.

## Architecture

All logic lives in `depth_mask.py`. The pipeline is:

```
Input Image
  → MiDaS depth estimation   → depth map (H×W float32, higher = closer/foreground)
  → Depth quantization        → band_map (H×W int, values 0..N-1, 0 = most foreground)
  → Canny edge detection      → edges (H×W uint8, adaptive thresholds based on median brightness)
  → Boundary snapping         → snapped band_map (depth boundaries aligned to Canny edges)
  → Cumulative mask building  → N binary masks
  → PNG save                  → PREFIX_layer_i.png files
```

### Key functions

- `load_midas()` / `run_midas()`: loads Intel MiDaS DPT_Large via `torch.hub` and runs inference
- `quantize_depth()`: divides depth map into N equal-width bands
- `compute_edges()`: Canny edge detection with adaptive thresholds (0.66× and 1.33× median grayscale)
- `snap_boundaries()`: aligns band boundaries to nearby Canny edges using morphological erosion + distance transform; contested boundary pixels snap to nearest core region, except pixels on edges which revert to raw band value. `snap_radius=10` is hardcoded.
- `build_masks()`: produces N cumulative binary masks (layer i covers bands 0..i-1)
- `save_alpha_only_png()`: writes RGBA PNG with only alpha channel set
- `select_device()`: picks CUDA > MPS > CPU automatically
