depth_mask.py — Monocular Depth Layer Mask Generator
=====================================================

Takes an input image and generates N cumulative alpha-only PNG masks using
MiDaS monocular depth estimation. Each mask layer progressively includes more
of the scene from foreground to background. Masks are RGBA PNGs with R=G=B=0
and only the alpha channel set (255=opaque, 0=transparent), intended for use
as compositing masks in layered animation workflows.


SETUP (one time)
----------------
Requires Homebrew Python 3.11:

    bash setup_venv.sh
    source .venv/bin/activate

The first run of the script will download ~400MB of MiDaS DPT_Large model
weights to ~/.cache/torch/hub/. Subsequent runs use the cache.

On Apple Silicon, set PYTORCH_ENABLE_MPS_FALLBACK=1 to allow automatic
per-op CPU fallback for any MPS-unsupported operations.


USAGE
-----
    python depth_mask.py <image> [--layers N] [--output PREFIX]

Arguments:
    image           Input image file path (required)
    --layers N      Number of depth layers to generate (default: 1)
    --output PREFIX Output filename prefix (default: derived from input name)


OUTPUT FILES
------------
Output files are named <PREFIX>_layer_1.png, <PREFIX>_layer_2.png, etc.

By default the prefix is the input filename without its extension, written
to the same directory as the input. Use --output to override.

Example: photo.jpg with --layers 3 produces:
    photo_layer_1.png   — most foreground band only (smallest masked region)
    photo_layer_2.png   — layer 1 + second depth band
    photo_layer_3.png   — layers 1+2 + third depth band (largest masked region)

Each layer is cumulative: layer N always contains everything in layer N-1 plus
the next depth band further from the camera.


EXAMPLES
--------
# 3 layers from photo.jpg (outputs photo_layer_1.png through photo_layer_3.png)
PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py photo.jpg --layers 3

# Custom output prefix (outputs masks/portrait_layer_1.png etc.)
PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py photo.jpg --layers 4 --output masks/portrait

# Single foreground mask (default)
PYTORCH_ENABLE_MPS_FALLBACK=1 python depth_mask.py photo.jpg


HOW IT WORKS
------------
1. MiDaS DPT_Large (loaded via torch.hub) estimates a depth map from the
   input image. Higher values = closer to the camera (foreground).

2. The depth map is normalized and divided into N equal bands. Band 0 is the
   most foreground (highest MiDaS value), band N-1 is the most background.

3. Canny edge detection is run on the input image. Band boundaries are snapped
   to nearby Canny edges (within 10px) so masks follow visible object contours
   rather than raw depth gradient boundaries.

4. Cumulative binary alpha masks are generated and saved as RGBA PNGs.


DEPENDENCIES
------------
Listed in requirements.txt:
    torch>=2.1.0
    torchvision>=0.16.0
    opencv-python>=4.8.0
    Pillow>=10.0.0
    numpy>=1.24.0
    timm>=0.9.0         (required internally by MiDaS DPT models)
