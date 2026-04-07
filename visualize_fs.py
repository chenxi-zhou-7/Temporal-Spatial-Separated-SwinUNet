#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS Measurement and Visualization Script for TSSwinUNet

Loads a trained TSSwinUNet model, performs prediction on M-mode echocardiography
images, computes Fractional Shortening (FS) via robust peak-valley detection,
and produces publication-quality visualizations.

Outputs:
    1. CSV table: sample, LVID_d (mm), LVID_s (mm), FS (%)
    2. Four-panel figure per sample:
       - (a) Original image
       - (b) Predicted segmentation mask
       - (c) LVID curve with peak (D) / valley (S) annotations and pair lines
       - (d) Mask overlay on original image with border annotations

Dependencies:
    pip install torch timm numpy opencv-python scipy matplotlib pandas pillow

Usage:
    python visualize_fs.py \
        --weight /path/to/best_model.pth \
        --input /path/to/image.tif \
        --output-dir /path/to/output \
        --height-mm 10.8
"""

import os
import sys
import csv
import argparse
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TIMM_SKIP_MODEL_CHECK"] = "1"

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import find_peaks, savgol_filter

# Import model from TSSwinUNet module
from TSSwinUNet import TSSwinUNet

# Import FS calculation from calculate_fs module
from calculate_fs import mask_to_curve, interpolate_valid_segment, smooth_curve, \
    build_alternating_extrema, pair_peak_valley, analyze_fs_stable, yolo_to_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def predict_mask(model, image_path, device):
    """Run TSSwinUNet inference, return predicted mask at original resolution."""
    orig = Image.open(image_path).convert("RGB")
    orig_h, orig_w = orig.size[::-1]
    img_224 = np.array(orig.resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_224).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()
    mask_224 = (prob > 0.5).astype(np.uint8)
    return cv2.resize(mask_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ═══════════════════════════════════════════════════════════════════════════════
# FS Analysis (wrapper around calculate_fs module)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_fs(mask, height_mm=None):
    """Analyze FS from a binary mask.

    Args:
        mask: Binary mask (H, W) uint8.
        height_mm: Physical image height in mm (for unit conversion).

    Returns:
        dict with fs_median, pairs, lvid_d_mm, lvid_s_mm, curve, peaks, valleys.
    """
    curve = mask_to_curve(mask)
    filled, vr = interpolate_valid_segment(curve)
    if vr is None:
        return {"fs_median": None, "pairs": [], "curve": curve,
                "lvid_d_px": None, "lvid_s_px": None, "lvid_d_mm": None, "lvid_s_mm": None}

    start, end = vr
    seg = filled[start:end + 1]
    smoothed_seg = smooth_curve(seg)
    md = max(6, int(len(seg) * 0.06))
    ss = float(np.std(smoothed_seg))
    sr = float(np.max(smoothed_seg) - np.min(smoothed_seg))
    prom = max(ss * 0.2, sr * 0.08, 1.0)

    pk, _ = find_peaks(smoothed_seg, distance=md, prominence=prom)
    vl, _ = find_peaks(-smoothed_seg, distance=md, prominence=prom)
    pk, vl = pk + start, vl + start

    sf = curve.copy()
    sf[start:end + 1] = smoothed_seg
    ext = build_alternating_extrema(pk, vl, sf)
    pairs = pair_peak_valley(filled, sf, ext, md, prom)

    fs_vals = [p["fs"] for p in pairs]
    fs_med = float(np.median(fs_vals)) if fs_vals else None

    peak_list = [(i, float(sf[i])) for i, k in ext if k == "peak"]
    valley_list = [(i, float(sf[i])) for i, k in ext if k == "valley"]

    lvid_d_med = float(np.median([p["lvid_d"] for p in pairs])) if pairs else None
    lvid_s_med = float(np.median([p["lvid_s"] for p in pairs])) if pairs else None

    lvid_d_mm = None
    lvid_s_mm = None
    if height_mm is not None and lvid_d_med is not None and lvid_s_med is not None:
        px_per_mm = mask.shape[0] / height_mm
        lvid_d_mm = lvid_d_med / px_per_mm
        lvid_s_mm = lvid_s_med / px_per_mm

    return {
        "fs_median": fs_med,
        "lvid_d_px": lvid_d_med,
        "lvid_s_px": lvid_s_med,
        "lvid_d_mm": lvid_d_mm,
        "lvid_s_mm": lvid_s_mm,
        "pairs": pairs,
        "curve": sf,
        "peaks": peak_list,
        "valleys": valley_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _px_to_mm(px_arr, height_mm, img_height_px):
    return np.asarray(px_arr, dtype=np.float32) * (height_mm / img_height_px)


def _interpolate_curve(curve):
    valid = ~np.isnan(curve)
    if np.sum(valid) == 0:
        return curve.copy()
    x = np.arange(len(curve))
    out = curve.copy()
    out[~valid] = np.interp(x[~valid], x[valid], curve[valid])
    return out


def extract_borders(mask):
    """Extract top (IVS) and bottom (LVPW) boundary curves."""
    _, w = mask.shape
    top = np.full(w, np.nan)
    bot = np.full(w, np.nan)
    for col in range(w):
        ys = np.where(mask[:, col] > 0)[0]
        if len(ys) > 0:
            top[col] = ys.min()
            bot[col] = ys.max()
    return top, bot


def create_four_panel_figure(image_path, pred_mask, fs_result, sample_name,
                             height_mm, output_dir):
    """Create a 4-panel publication figure.

    Panels:
        (a) Original image
        (b) Predicted segmentation mask
        (c) LVID curve with peak/valley annotations and FS pair lines
        (d) Mask overlay with boundary borders and per-cycle LVID_d/LVID_s labels
    """
    orig_img = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig_img.shape[:2]
    mm_per_px = height_mm / h

    fs_label = f"FS={fs_result['fs_median']:.2f}%" if fs_result["fs_median"] is not None else "FS=N/A"

    overlay_color = np.array([244, 177, 131, 128], dtype=np.uint8)
    top_border, bot_border = extract_borders(pred_mask)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(sample_name, fontsize=16, fontweight="bold", y=0.98)

    # ── (a) Original image ──
    ax = axes[0, 0]
    ax.imshow(orig_img)
    ax.set_title("(a) Original Image", fontsize=13, fontweight="bold")
    ax.axis("off")

    # ── (b) Predicted mask ──
    ax = axes[0, 1]
    mask_display = np.zeros((h, w, 3), dtype=np.uint8)
    mask_display[pred_mask > 0] = [244, 177, 131]
    ax.imshow(mask_display)
    ax.set_title(f"(b) Predicted Mask ({fs_label})", fontsize=13, fontweight="bold")
    ax.axis("off")

    # ── (c) LVID curve ──
    ax = axes[1, 0]
    curve_mm = fs_result["curve"] * mm_per_px
    x = np.arange(len(curve_mm))
    ax.plot(x, curve_mm, color="#C8C8C8", lw=1.0, alpha=0.6, label="Raw")

    filled, vr = interpolate_valid_segment(fs_result["curve"])
    if vr:
        s, e = vr
        sm = fs_result["curve"].copy()
        sm[s:e + 1] = smooth_curve(filled[s:e + 1])
        ax.plot(x, sm * mm_per_px, color="#D89C27", lw=2.2, label="Smoothed")

    for idx, val in fs_result["peaks"]:
        val_mm = val * mm_per_px
        ax.scatter([idx], [val_mm], color="#1E8A5B", s=70, marker="^", zorder=5)
        ax.annotate(f"D {val_mm:.2f}", (idx, val_mm), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="#1E8A5B", fontweight="bold")

    for idx, val in fs_result["valleys"]:
        val_mm = val * mm_per_px
        ax.scatter([idx], [val_mm], color="#C85250", s=70, marker="v", zorder=5)
        ax.annotate(f"S {val_mm:.2f}", (idx, val_mm), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="#C85250", fontweight="bold")

    for pair in fs_result["pairs"]:
        pi, vi = pair["peak_idx"], pair["valley_idx"]
        ax.plot([pi, vi], [pair["lvid_d"] * mm_per_px, pair["lvid_s"] * mm_per_px],
                color="#D89C27", lw=1.8, alpha=0.85)

    ax.set_title(f"(c) LVID Curve ({fs_label})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (Column)")
    ax.set_ylabel("LV Diameter (mm)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    # ── (d) Overlay with borders and per-cycle annotations ──
    ax = axes[1, 1]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[pred_mask > 0] = overlay_color
    ax.imshow(orig_img)
    ax.imshow(overlay)

    valid_top = ~np.isnan(top_border)
    valid_bot = ~np.isnan(bot_border)
    ax.plot(np.where(valid_top)[0], top_border[valid_top],
            color="#1E8A5B", linewidth=1.5, label="IVS border")
    ax.plot(np.where(valid_bot)[0], bot_border[valid_bot],
            color="#C85250", linewidth=1.5, label="LVPW border")

    for i, pair in enumerate(fs_result["pairs"]):
        pi_col = pair["peak_idx"]
        vi_col = pair["valley_idx"]
        if pi_col < w:
            ax.axvline(x=pi_col, color="#1E8A5B", ls=":", lw=1.2, alpha=0.7)
            ax.annotate(f"D{i+1}: {pair['lvid_d']*mm_per_px:.2f}mm", xy=(pi_col, 12),
                        fontsize=7, color="#1E8A5B", fontweight="bold", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#1E8A5B", alpha=0.85))
        if vi_col < w:
            ax.axvline(x=vi_col, color="#C85250", ls=":", lw=1.2, alpha=0.7)
            ax.annotate(f"S{i+1}: {pair['lvid_s']*mm_per_px:.2f}mm", xy=(vi_col, 30),
                        fontsize=7, color="#C85250", fontweight="bold", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#C85250", alpha=0.85))

    ax.set_title(f"(d) Overlay with Borders ({fs_label})", fontsize=13, fontweight="bold")
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = output_dir / f"{sample_name}_FS_Visualization.png"
    out_pdf = output_dir / f"{sample_name}_FS_Visualization.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Output
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(rows, output_path):
    fieldnames = ["sample", "LVID_d_px", "LVID_s_px", "LVID_d_mm", "LVID_s_mm", "FS (%)"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Single Sample Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_single(image_path, model, device, height_mm, output_dir):
    """Run full pipeline for one image: predict -> analyze -> visualize -> csv."""
    sample_name = Path(image_path).stem
    print(f"[*] Processing: {sample_name}")

    pred_mask = predict_mask(model, image_path, device)
    fs_result = analyze_fs(pred_mask, height_mm)

    create_four_panel_figure(image_path, pred_mask, fs_result, sample_name,
                             height_mm, output_dir)

    row = {
        "sample": sample_name,
        "LVID_d_px": f"{fs_result['lvid_d_px']:.2f}" if fs_result['lvid_d_px'] is not None else "",
        "LVID_s_px": f"{fs_result['lvid_s_px']:.2f}" if fs_result['lvid_s_px'] is not None else "",
        "LVID_d_mm": f"{fs_result['lvid_d_mm']:.2f}" if fs_result['lvid_d_mm'] is not None else "",
        "LVID_s_mm": f"{fs_result['lvid_s_mm']:.2f}" if fs_result['lvid_s_mm'] is not None else "",
        "FS (%)": f"{fs_result['fs_median']:.2f}" if fs_result['fs_median'] is not None else "",
    }
    print(f"    LVID_d = {row['LVID_d_mm']} mm, LVID_s = {row['LVID_s_mm']} mm, FS = {row['FS (%)']}%")
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TSSwinUNet: M-Mode Echocardiography Segmentation & FS Measurement",
    )
    parser.add_argument("--weight", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--input", required=True, help="Input image or directory of images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--height-mm", type=float, required=True,
                        help="Physical image height in mm")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Load model
    print(f"[*] Loading model: {args.weight}")
    model = TSSwinUNet(in_channels=3, num_classes=2)
    checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    print("[*] Model loaded.")

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_dir():
        extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in extensions])
    elif input_path.is_file():
        images = [input_path]
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    print(f"[*] Found {len(images)} image(s)")

    # Process
    rows = []
    for img_path in images:
        row = process_single(str(img_path), model, device, args.height_mm, output_dir)
        rows.append(row)

    # Save CSV
    csv_path = output_dir / "fs_measurements.csv"
    save_csv(rows, csv_path)
    print(f"[*] Results saved to {output_dir}")
    print(f"[*] CSV: {csv_path}")


if __name__ == "__main__":
    main()
