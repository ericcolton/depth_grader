#!/usr/bin/env python3
"""
depth_mask.py — Monocular depth layer mask generator using MiDaS.

Outputs N cumulative alpha-only PNG masks (RGBA, R=G=B=0). Layer 1 covers the
most foreground depth band; each subsequent layer adds the next band behind it.

Usage:
    python depth_mask.py <image> [--layers N] [--output PREFIX]

Examples:
    python depth_mask.py photo.jpg --layers 3
    python depth_mask.py photo.jpg --layers 4 --output masks/portrait
"""

import argparse
import colorsys
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_KAPPA = 0.5522847498  # cubic bezier control point ratio for ellipse approximation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cumulative depth mask PNGs using MiDaS"
    )
    parser.add_argument("image", help="Input image file path")
    parser.add_argument(
        "--layers", type=int, default=1, metavar="N",
        help="Number of depth layers to generate (default: 1)"
    )
    parser.add_argument(
        "--output", default=None, metavar="PREFIX",
        help="Output filename prefix (default: derived from input filename)"
    )
    args = parser.parse_args()

    if args.layers < 1:
        parser.error("--layers must be >= 1")
    if not os.path.isfile(args.image):
        parser.error(f"Image file not found: {args.image}")

    return args


def derive_prefix(image_path: str, output_arg: str | None) -> str:
    if output_arg is not None:
        return output_arg
    base, _ = os.path.splitext(image_path)
    return base


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# MiDaS depth estimation
# ---------------------------------------------------------------------------

def load_midas(device: torch.device):
    print("Loading MiDaS DPT_Large model (first run downloads ~400MB)...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.dpt_transform
    return model, transform


def run_midas(
    image_path: str,
    model,
    transform,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (depth H×W float32, img_rgb H×W×3 uint8).
    depth values: higher = closer to camera (foreground).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    print("Running depth inference...")
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy().astype(np.float32)
    return depth, img_rgb


# ---------------------------------------------------------------------------
# Depth quantization
# ---------------------------------------------------------------------------

def quantize_depth(depth: np.ndarray, n_layers: int) -> np.ndarray:
    """Divide depth into n_layers equal bands.
    Band 0 = most foreground (highest MiDaS value = closest to camera).
    Returns band_map H×W int32 with values 0..n_layers-1.
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 0.01:
        print(
            "WARNING: depth range is very small (flat depth map). "
            "All masks may be nearly identical."
        )
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)  # 0=background, 1=foreground
    # Invert so that 0 maps to foreground band
    band_map = np.floor((1.0 - depth_norm) * n_layers).astype(np.int32)
    band_map = np.clip(band_map, 0, n_layers - 1)
    return band_map


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

def compute_edges(img_rgb: np.ndarray) -> np.ndarray:
    """Returns binary edge map H×W uint8 (255=edge, 0=not)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, low, high)
    return edges


# ---------------------------------------------------------------------------
# Boundary snapping
# ---------------------------------------------------------------------------

def snap_boundaries(
    band_map: np.ndarray,
    edges: np.ndarray,
    snap_radius: int = 10,
) -> np.ndarray:
    """Snap depth-band boundaries to nearby Canny edges.

    Algorithm:
      1. Erode each band's mask by snap_radius → safe core (definitely that band).
      2. Pixels not covered by any core are "contested" (near a boundary).
      3. Contested pixels: assign to the nearest core by distance transform.
      4. Override: contested pixels that lie ON a Canny edge keep their raw band value.
         This makes Canny edges act as hard boundaries inside the contested zone.

    Returns snapped band_map H×W int32.
    """
    n_layers = int(band_map.max()) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * snap_radius + 1, 2 * snap_radius + 1)
    )

    # Step 1: erode each band to its safe core
    cores = []
    for k in range(n_layers):
        mask = (band_map == k).astype(np.uint8)
        core = cv2.erode(mask, kernel)
        cores.append(core)

    # Step 2: build output seeded from cores (-1 = unassigned)
    snapped = np.full(band_map.shape, -1, dtype=np.int32)
    for k in range(n_layers):
        snapped[cores[k] > 0] = k

    unassigned_mask = snapped == -1
    if not unassigned_mask.any():
        return snapped

    # Step 3: for unassigned pixels, pick the nearest core band
    dist_to_core = []
    for k in range(n_layers):
        # distance from each pixel to the nearest pixel in core k
        core_inv = (cores[k] == 0).astype(np.uint8)
        dist = cv2.distanceTransform(core_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_to_core.append(dist)

    dist_stack = np.stack(dist_to_core, axis=0)   # (N, H, W)
    nearest_band = np.argmin(dist_stack, axis=0).astype(np.int32)  # (H, W)

    snapped[unassigned_mask] = nearest_band[unassigned_mask]

    # Step 4: within the contested zone, Canny edge pixels act as hard boundaries
    # → revert them to their raw band_map value so edges don't get blurred over
    contested_edges = unassigned_mask & (edges > 0)
    snapped[contested_edges] = band_map[contested_edges]

    return snapped


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def build_masks(band_map: np.ndarray, n_layers: int) -> list[np.ndarray]:
    """Build N cumulative binary alpha masks (0 or 255).
    Layer i (1-indexed) includes bands 0..i-1 (foreground through layer i).
    """
    masks = []
    cumulative = np.zeros(band_map.shape, dtype=bool)
    for i in range(n_layers):
        cumulative |= (band_map == i)
        alpha = np.where(cumulative, np.uint8(255), np.uint8(0))
        masks.append(alpha)
    return masks


def save_alpha_only_png(alpha_array: np.ndarray, output_path: str) -> None:
    """Save a binary alpha mask as an RGBA PNG with R=G=B=0."""
    h, w = alpha_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha_array
    img = Image.fromarray(rgba, mode="RGBA")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Orbit metadata
# ---------------------------------------------------------------------------

def _ellipse_to_bezier(
    cx: float, cy: float, rx: float, ry: float
) -> list[dict]:
    """Approximate ellipse as 4 cubic Bezier curves (normalized screen coords).

    t=0 starts at rightmost point (cx+rx, cy), proceeds counter-clockwise.
    Segment boundaries: t=0, 0.25, 0.5, 0.75, 1.0.
    """
    k = _KAPPA
    return [
        # t 0.00→0.25: right → top
        {"start": {"x": cx + rx,     "y": cy},
         "cp1":   {"x": cx + rx,     "y": cy - ry * k},
         "cp2":   {"x": cx + rx * k, "y": cy - ry},
         "end":   {"x": cx,          "y": cy - ry}},
        # t 0.25→0.50: top → left
        {"start": {"x": cx,          "y": cy - ry},
         "cp1":   {"x": cx - rx * k, "y": cy - ry},
         "cp2":   {"x": cx - rx,     "y": cy - ry * k},
         "end":   {"x": cx - rx,     "y": cy}},
        # t 0.50→0.75: left → bottom
        {"start": {"x": cx - rx,     "y": cy},
         "cp1":   {"x": cx - rx,     "y": cy + ry * k},
         "cp2":   {"x": cx - rx * k, "y": cy + ry},
         "end":   {"x": cx,          "y": cy + ry}},
        # t 0.75→1.00: bottom → right
        {"start": {"x": cx,          "y": cy + ry},
         "cp1":   {"x": cx + rx * k, "y": cy + ry},
         "cp2":   {"x": cx + rx,     "y": cy + ry * k},
         "end":   {"x": cx + rx,     "y": cy}},
    ]


def _occlusion_to_arcs(flags: list[bool]) -> list[dict]:
    """Convert per-sample occlusion flags to arc segments with t in [0, 1]."""
    n = len(flags)
    arcs: list[dict] = []
    start = 0
    current = flags[0]
    for i in range(1, n):
        if flags[i] != current:
            arcs.append({"t_start": start / n, "t_end": i / n, "occluded": current})
            current = flags[i]
            start = i
    arcs.append({"t_start": start / n, "t_end": 1.0, "occluded": current})
    return arcs


def _compute_component_orbit(
    component_mask: np.ndarray,
    img_h: int,
    img_w: int,
    padding: float = 1.35,
    n_samples: int = 360,
) -> dict:
    """Compute orbit ellipse and occlusion data for a single connected component.

    Assumes component_mask is non-empty (caller's responsibility).
    Returns a dict with normalized [0,1] coordinates.
    """
    ys, xs = np.where(component_mask > 0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    rx = max((float(xs.max()) - float(xs.min())) / 2.0 * padding, 1.0)
    ry = max((float(ys.max()) - float(ys.min())) / 2.0 * padding, 1.0)

    flags: list[bool] = []
    for i in range(n_samples):
        t = 2.0 * math.pi * i / n_samples
        px = cx + rx * math.cos(t)
        py = cy + ry * math.sin(t)
        occluded = False
        n_ray = 30
        for j in range(1, n_ray):
            lx = int(round(px + (cx - px) * j / n_ray))
            ly = int(round(py + (cy - py) * j / n_ray))
            if 0 <= ly < img_h and 0 <= lx < img_w and component_mask[ly, lx] > 0:
                occluded = True
                break
        flags.append(occluded)

    bezier_raw = _ellipse_to_bezier(cx / img_w, cy / img_h, rx / img_w, ry / img_h)
    bezier = [
        {k: {axis: round(v, 4) for axis, v in pt.items()} for k, pt in seg.items()}
        for seg in bezier_raw
    ]
    arcs = [
        {"t_start": round(a["t_start"], 4), "t_end": round(a["t_end"], 4), "occluded": a["occluded"]}
        for a in _occlusion_to_arcs(flags)
    ]

    return {
        "centroid": {"x": round(cx / img_w, 4), "y": round(cy / img_h, 4)},
        "pixel_count": int(len(xs)),
        "orbit_ellipse": {
            "center": {"x": round(cx / img_w, 4), "y": round(cy / img_h, 4)},
            "rx": round(rx / img_w, 4),
            "ry": round(ry / img_h, 4),
        },
        "bezier_curves": bezier,
        "arcs": arcs,
    }


def compute_band_orbits(
    band_mask: np.ndarray,
    img_h: int,
    img_w: int,
    min_component_pixels: int = 200,
    padding: float = 1.35,
    n_samples: int = 360,
) -> list[dict]:
    """Find connected components in band_mask and compute an orbit for each.

    Components smaller than `min_component_pixels` are ignored (noise).
    Returns a list of orbit dicts (one per component), sorted largest-first.
    """
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        band_mask, connectivity=8
    )
    orbits = []
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        component_mask = (label_map == label).astype(np.uint8) * 255
        orbit = _compute_component_orbit(component_mask, img_h, img_w, padding, n_samples)
        orbits.append(orbit)
    orbits.sort(key=lambda o: o["pixel_count"], reverse=True)
    return orbits


# ---------------------------------------------------------------------------
# Orbit visualization
# ---------------------------------------------------------------------------

def _random_saturated_color() -> tuple[int, int, int]:
    """Return a random fully-saturated BGR color (S=1, V=1, random hue)."""
    r, g, b = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV


def render_orbit_preview(
    band_mask: np.ndarray,
    orbits: list[dict],
    img_h: int,
    img_w: int,
    line_thickness: int = 2,
) -> np.ndarray:
    """Render orbit ellipses over the alpha mask as an RGBA image.

    The band mask is shown as a dim white layer; each component's orbit
    ellipse and centroid dot are drawn in a unique random full-saturation color.
    Returns H×W×4 uint8 RGBA array (fully opaque).
    """
    bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    bgr[band_mask > 0] = (55, 55, 55)

    for orbit in orbits:
        color = _random_saturated_color()
        ell = orbit["orbit_ellipse"]
        cx_px = int(round(ell["center"]["x"] * img_w))
        cy_px = int(round(ell["center"]["y"] * img_h))
        rx_px = max(1, int(round(ell["rx"] * img_w)))
        ry_px = max(1, int(round(ell["ry"] * img_h)))
        cv2.ellipse(bgr, (cx_px, cy_px), (rx_px, ry_px), 0, 0, 360, color, line_thickness)
        cv2.circle(bgr, (cx_px, cy_px), 4, color, -1)

    rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = bgr[:, :, ::-1]  # BGR → RGB
    rgba[:, :, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    prefix = derive_prefix(args.image, args.output)
    n_layers = args.layers

    device = select_device()
    model, transform = load_midas(device)
    depth, img_rgb = run_midas(args.image, model, transform, device)

    print(f"Quantizing depth into {n_layers + 1} band(s)...")
    band_map = quantize_depth(depth, n_layers + 1)

    print("Detecting edges...")
    edges = compute_edges(img_rgb)

    print("Snapping boundaries to edges...")
    snapped = snap_boundaries(band_map, edges, snap_radius=10)

    masks = build_masks(snapped, n_layers)

    for i, mask in enumerate(masks, start=1):
        out_path = f"{prefix}_layer_{i}.png"
        save_alpha_only_png(mask, out_path)
        covered = int(np.sum(mask > 0))
        total = mask.size
        print(f"  Wrote {out_path}  ({covered}/{total} px = {100*covered/total:.1f}% opaque)")

    print("Computing orbit metadata...")
    img_h, img_w = snapped.shape
    for k in range(n_layers):
        band_mask = ((snapped == k).astype(np.uint8) * 255)
        components = compute_band_orbits(band_mask, img_h, img_w)
        orbit_meta = {
            "image_size": {"width": img_w, "height": img_h},
            "layer_index": k + 1,
            "t_param": "normalized_angle",
            "note": "t=0 is rightmost point of ellipse, increases counter-clockwise. "
                    "Coordinates normalized to [0,1] (x/width, y/height).",
            "components": components,
        }
        orbit_path = f"{prefix}_layer_{k + 1}_orbits.json"
        with open(orbit_path, "w") as f:
            json.dump(orbit_meta, f, indent=2)

        preview = render_orbit_preview(band_mask, components, img_h, img_w)
        preview_path = f"{prefix}_layer_{k + 1}_orbits.png"
        preview_img = Image.fromarray(preview, mode="RGBA")
        os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)
        preview_img.save(preview_path, format="PNG")
        print(f"  Wrote {orbit_path} + {os.path.basename(preview_path)}  ({len(components)} component(s))")

    print("Done.")


if __name__ == "__main__":
    main()
