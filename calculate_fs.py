#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fractional Shortening (FS) Calculator for M-mode Echocardiography Segmentation

Computes FS from binary segmentation masks using robust peak-valley detection
on the left ventricular internal diameter (LVID) temporal curve.

Algorithm:
    1. Extract per-column LVID curve from binary mask
    2. Interpolate zero-valued gaps within valid segments
    3. Apply Savitzky-Golay smoothing with adaptive window size
    4. Detect peaks (end-diastole) and valleys (end-systole) with alternating enforcement
    5. One-to-one peak-valley pairing filtered by cycle width and amplitude
    6. FS = (LVID_d - LVID_s) / LVID_d * 100%

Input modes:
    - Curve file: --curve-file (.npy / .csv / .txt)
    - Binary mask image: --mask-image
    - YOLO polygon label: --yolo-label --image-height H --image-width W

Dependencies:
    pip install numpy opencv-python scipy matplotlib

Usage:
    python calculate_fs.py --mask-image mask.png
    python calculate_fs.py --yolo-label label.txt --image-height 512 --image-width 640
    python calculate_fs.py --curve-file curve.npy
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ═══════════════════════════════════════════════════════════════════════════════
# I/O Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def yolo_to_mask(label_path, img_shape):
    """Convert YOLO polygon annotation to binary mask.

    Args:
        label_path: Path to YOLO-format txt file.
        img_shape: (height, width) of the image.
    Returns:
        Binary mask (H, W) as uint8 array.
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    label_path = Path(label_path)
    if not label_path.exists():
        return mask

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = np.array(
                [float(x) for x in parts[1:]], dtype=np.float32
            ).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = coords.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=1)
    return mask


def mask_to_curve(mask):
    """Extract per-column LVID curve from binary mask.

    For each column, LVID = bottom_y - top_y of the foreground region.

    Args:
        mask: Binary mask (H, W).
    Returns:
        1D float array of LVID values, one per column.
    """
    _, w = mask.shape
    curve = np.zeros(w, dtype=np.float32)
    for col in range(w):
        ys = np.where(mask[:, col] > 0)[0]
        if len(ys) >= 2:
            curve[col] = float(ys[-1] - ys[0])
    return curve


def load_curve(curve_file):
    """Load curve from .npy / .csv / .txt file.

    Args:
        curve_file: Path to curve file.
    Returns:
        1D float32 array.
    """
    path = Path(curve_file)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        curve = np.load(path)
    elif suffix in {".txt", ".csv"}:
        curve = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported curve file format: {suffix}")

    curve = np.asarray(curve, dtype=np.float32).squeeze()
    if curve.ndim != 1:
        raise ValueError("Curve file must contain a 1D array")
    return curve


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Processing
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_valid_segment(curve):
    """Linearly interpolate zero-valued gaps within valid segments.

    Args:
        curve: 1D LVID curve.
    Returns:
        (interpolated_curve, (start, end)) or (curve, None) if insufficient data.
    """
    curve = np.asarray(curve, dtype=np.float32)
    valid = np.where(curve > 0)[0]
    if len(valid) < 5:
        return curve.copy(), None

    start = int(valid[0])
    end = int(valid[-1])
    segment = curve[start : end + 1].copy()
    seg_x = np.arange(len(segment))
    seg_valid = np.where(segment > 0)[0]

    if len(seg_valid) >= 2:
        segment = np.interp(seg_x, seg_valid, segment[seg_valid]).astype(np.float32)

    filled = curve.copy()
    filled[start : end + 1] = segment
    return filled, (start, end)


def smooth_curve(curve):
    """Apply adaptive Savitzky-Golay smoothing.

    Window size scales with curve length (capped at 15) to avoid
    over-smoothing short signals.

    Args:
        curve: 1D signal.
    Returns:
        Smoothed 1D signal.
    """
    n = len(curve)
    if n < 7:
        return curve.copy()

    window = max(5, int(round(n * 0.15)))
    if window % 2 == 0:
        window += 1
    window = min(window, 15)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return curve.copy()
    if window % 2 == 0:
        window -= 1

    polyorder = 3 if window >= 7 else 2
    return savgol_filter(curve, window_length=window, polyorder=polyorder)


# ═══════════════════════════════════════════════════════════════════════════════
# Peak-Valley Detection
# ═══════════════════════════════════════════════════════════════════════════════

def build_alternating_extrema(peaks, valleys, smoothed):
    """Merge peaks and valleys and enforce strict alternation.

    When consecutive extrema are of the same type, only the more extreme
    one is kept (higher for peaks, lower for valleys).

    Args:
        peaks: Array of peak indices.
        valleys: Array of valley indices.
        smoothed: Smoothed curve for value comparison.
    Returns:
        List of (index, 'peak'|'valley') tuples in temporal order.
    """
    extrema = [(int(idx), "peak") for idx in peaks] + [
        (int(idx), "valley") for idx in valleys
    ]
    extrema.sort(key=lambda x: x[0])

    alternating = []
    for idx, kind in extrema:
        if not alternating:
            alternating.append((idx, kind))
            continue

        prev_idx, prev_kind = alternating[-1]
        if kind != prev_kind:
            alternating.append((idx, kind))
            continue

        prev_val = smoothed[prev_idx]
        curr_val = smoothed[idx]

        if kind == "peak" and curr_val > prev_val:
            alternating[-1] = (idx, kind)
        elif kind == "valley" and curr_val < prev_val:
            alternating[-1] = (idx, kind)

    return alternating


def pair_peak_valley(curve, smoothed, extrema, min_distance, prominence_threshold):
    """Extract one-to-one peak->valley pairs and compute FS per pair.

    Filtering criteria:
        - Cycle width within [0.5x, 1.8x] median width
        - Amplitude drop >= 0.5x median drop
        - Minimum drop threshold based on curve statistics

    Args:
        curve: Original (interpolated) LVID curve.
        smoothed: Smoothed LVID curve.
        extrema: Alternating peak/valley list from build_alternating_extrema.
        min_distance: Minimum allowed peak-to-valley distance in samples.
        prominence_threshold: Minimum prominence for peak detection.
    Returns:
        List of dicts, each containing peak_idx, valley_idx, lvid_d, lvid_s, fs.
    """
    raw_pairs = []
    min_width = max(2, min_distance // 2)
    max_width = max(min_width + 1, min_distance * 3)
    curve_range = float(np.max(smoothed) - np.min(smoothed))
    min_drop = max(prominence_threshold * 0.5, curve_range * 0.05, 1.0)

    for i in range(len(extrema) - 1):
        peak_idx, peak_kind = extrema[i]
        valley_idx, valley_kind = extrema[i + 1]
        if peak_kind != "peak" or valley_kind != "valley":
            continue

        width = valley_idx - peak_idx
        if width < min_width or width > max_width:
            continue

        lvid_d = float(curve[peak_idx])
        lvid_s = float(curve[valley_idx])
        drop = lvid_d - lvid_s
        if lvid_d <= 0 or lvid_s < 0 or drop <= min_drop:
            continue
        if lvid_s >= lvid_d:
            continue

        fs = drop / lvid_d * 100.0
        raw_pairs.append({
            "peak_idx": int(peak_idx),
            "valley_idx": int(valley_idx),
            "width": int(width),
            "lvid_d": lvid_d,
            "lvid_s": lvid_s,
            "drop": drop,
            "fs": fs,
        })

    if not raw_pairs:
        return []

    # Secondary filtering by cycle consistency
    widths = np.array([p["width"] for p in raw_pairs], dtype=np.float32)
    drops = np.array([p["drop"] for p in raw_pairs], dtype=np.float32)
    median_width = float(np.median(widths))
    median_drop = float(np.median(drops))

    filtered = []
    for pair in raw_pairs:
        width_ok = 0.5 * median_width <= pair["width"] <= 1.8 * median_width
        drop_ok = pair["drop"] >= 0.5 * median_drop
        if width_ok and drop_ok:
            filtered.append(pair)

    return filtered if filtered else raw_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_fs_stable(curve, min_distance=None, prominence_scale=0.2):
    """Robust FS analysis pipeline.

    Args:
        curve: 1D LVID curve (float32).
        min_distance: Minimum peak-valley distance in samples. Auto-estimated if None.
        prominence_scale: Scale factor for prominence threshold.

    Returns:
        dict with keys:
            fs_median: Median FS value (%) or None if no valid pairs found.
            fs_values: List of per-cycle FS values.
            pairs: List of detailed pair dicts.
            smoothed_curve: Full smoothed curve as list.
            valid_range: [start, end] indices of valid segment or None.
            peaks: List of peak indices after alternating enforcement.
            valleys: List of valley indices after alternating enforcement.
    """
    curve = np.asarray(curve, dtype=np.float32).squeeze()
    if curve.ndim != 1:
        raise ValueError("Curve must be a 1D array")

    filled_curve, valid_range = interpolate_valid_segment(curve)
    if valid_range is None:
        return {
            "fs_median": None,
            "fs_values": [],
            "pairs": [],
            "smoothed_curve": curve.tolist(),
            "valid_range": None,
            "peaks": [],
            "valleys": [],
        }

    start, end = valid_range
    segment = filled_curve[start : end + 1]
    smoothed_segment = smooth_curve(segment)

    if min_distance is None:
        min_distance = max(6, int(len(segment) * 0.06))

    seg_std = float(np.std(smoothed_segment))
    seg_range = float(np.max(smoothed_segment) - np.min(smoothed_segment))
    prominence_threshold = max(seg_std * prominence_scale, seg_range * 0.08, 1.0)

    peaks, _ = find_peaks(
        smoothed_segment, distance=min_distance, prominence=prominence_threshold
    )
    valleys, _ = find_peaks(
        -smoothed_segment, distance=min_distance, prominence=prominence_threshold
    )

    peaks = peaks + start
    valleys = valleys + start

    smoothed_full = curve.copy()
    smoothed_full[start : end + 1] = smoothed_segment

    extrema = build_alternating_extrema(peaks, valleys, smoothed_full)
    pairs = pair_peak_valley(
        filled_curve, smoothed_full, extrema, min_distance, prominence_threshold
    )

    fs_values = [p["fs"] for p in pairs]
    fs_median = float(np.median(fs_values)) if fs_values else None

    final_peaks = [idx for idx, kind in extrema if kind == "peak"]
    final_valleys = [idx for idx, kind in extrema if kind == "valley"]

    return {
        "fs_median": fs_median,
        "fs_values": fs_values,
        "pairs": pairs,
        "smoothed_curve": smoothed_full.tolist(),
        "valid_range": [int(start), int(end)],
        "peaks": final_peaks,
        "valleys": final_valleys,
        "min_distance": int(min_distance),
        "prominence_threshold": float(prominence_threshold),
    }


def load_curve_from_args(args):
    """Load LVID curve from command-line arguments.

    Supports three input modes: curve file, binary mask, or YOLO label.
    """
    if args.curve_file:
        return load_curve(args.curve_file)

    if args.mask_image:
        mask = cv2.imread(str(args.mask_image), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask image: {args.mask_image}")
        mask = (mask > 0).astype(np.uint8)
        return mask_to_curve(mask)

    if args.yolo_label:
        if args.image_height is None or args.image_width is None:
            raise ValueError(
                "--image-height and --image-width are required with --yolo-label"
            )
        mask = yolo_to_mask(args.yolo_label, (args.image_height, args.image_width))
        return mask_to_curve(mask)

    raise ValueError(
        "Provide one of: --curve-file, --mask-image, or --yolo-label"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Robust peak-valley pairing and FS computation for M-mode echocardiography"
    )
    parser.add_argument("--curve-file", type=str, help="Input curve file (.npy / .csv / .txt)")
    parser.add_argument("--mask-image", type=str, help="Input binary mask image")
    parser.add_argument("--yolo-label", type=str, help="Input YOLO polygon annotation file")
    parser.add_argument("--image-height", type=int, help="Image height for YOLO label")
    parser.add_argument("--image-width", type=int, help="Image width for YOLO label")
    parser.add_argument(
        "--min-distance", type=int, default=None,
        help="Minimum peak-valley distance in samples (auto-estimated if not set)"
    )
    parser.add_argument(
        "--prominence-scale", type=float, default=0.2,
        help="Scale factor for prominence threshold"
    )
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    curve = load_curve_from_args(args)
    result = analyze_fs_stable(
        curve=curve,
        min_distance=args.min_distance,
        prominence_scale=args.prominence_scale,
    )

    summary = {
        "fs_median": result["fs_median"],
        "n_pairs": len(result["pairs"]),
        "fs_values": [round(v, 4) for v in result["fs_values"]],
        "pairs": [
            {
                "peak_idx": p["peak_idx"],
                "valley_idx": p["valley_idx"],
                "lvid_d": round(p["lvid_d"], 4),
                "lvid_s": round(p["lvid_s"], 4),
                "fs": round(p["fs"], 4),
                "width": p["width"],
            }
            for p in result["pairs"]
        ],
        "min_distance": result.get("min_distance"),
        "prominence_threshold": result.get("prominence_threshold"),
        "valid_range": result.get("valid_range"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
