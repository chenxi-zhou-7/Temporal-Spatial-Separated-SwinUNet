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

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import find_peaks, savgol_filter

import timm


# ═══════════════════════════════════════════════════════════════════════════════
# PTS Block: Physiological Time-Space Decoupling
# ═══════════════════════════════════════════════════════════════════════════════

class PTSBlock(nn.Module):
    """Physiological Time-Space Block.

    Decomposes features along orthogonal axes:
      - Spatial branch (Y / depth): (K_s, 1) conv + BatchNorm
      - Temporal branch (X / rhythm): (1, K_t) conv + InstanceNorm
      - Gated fusion: temporal attention modulates spatial features.
    """

    def __init__(self, in_channels, out_channels=None,
                 spatial_kernel=7, temporal_kernel=7, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        mid_ch = in_channels // reduction if in_channels >= reduction else in_channels

        self.spatial_conv = nn.Conv2d(
            in_channels, self.out_channels,
            kernel_size=(spatial_kernel, 1),
            padding=(spatial_kernel // 2, 0), bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(self.out_channels)
        self.spatial_act = nn.GELU()

        self.temporal_conv = nn.Conv2d(
            in_channels, self.out_channels,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2), bias=False,
        )
        self.temporal_in = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.temporal_act = nn.GELU()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, self.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Conv2d(
            self.out_channels, self.out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.fusion_bn = nn.BatchNorm2d(self.out_channels)

        self.residual_proj = (
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            if in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = x
        spatial = self.spatial_act(self.spatial_bn(self.spatial_conv(x)))
        temporal = self.temporal_act(self.temporal_in(self.temporal_conv(x)))
        gate = self.gate_conv(temporal)
        gated = spatial * gate
        out = self.fusion_bn(self.fusion_conv(gated))
        return out + self.residual_proj(identity)


# ═══════════════════════════════════════════════════════════════════════════════
# TSSwinUNet Model
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockWithPTS(nn.Module):
    """Upsample -> Skip concat -> Conv -> PTS block."""

    def __init__(self, in_ch, skip_ch, out_ch, spatial_kernel=7, temporal_kernel=7):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(ConvBlock(out_ch + skip_ch, out_ch))
        self.pts = PTSBlock(out_ch, out_ch,
                            spatial_kernel=spatial_kernel,
                            temporal_kernel=temporal_kernel)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.pts(x)
        return x


class TSSwinUNet(nn.Module):
    """Temporal-Spatial Swin UNet for M-mode echocardiography segmentation.

    Encoder: Swin Transformer (pretrained=False).
    Decoder: 3 stages, each with PTS block.
    Head: 4x bilinear upsample + 1x1 conv.
    """

    def __init__(self, in_channels=3, num_classes=2,
                 encoder_name='swin_tiny_patch4_window7_224',
                 spatial_kernel=7, temporal_kernel=7):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=False, features_only=True, in_chans=in_channels,
        )
        enc_ch = self.encoder.feature_info.channels()

        self.decoder1 = DecoderBlockWithPTS(
            enc_ch[3], enc_ch[2], enc_ch[2],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )
        self.decoder2 = DecoderBlockWithPTS(
            enc_ch[2], enc_ch[1], enc_ch[1],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )
        self.decoder3 = DecoderBlockWithPTS(
            enc_ch[1], enc_ch[0], enc_ch[0],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(enc_ch[0], num_classes, kernel_size=1),
        )

    def _ensure_nchw(self, features):
        out = []
        for f in features:
            if f.shape[1] == f.shape[2] and f.shape[3] != f.shape[1]:
                f = f.permute(0, 3, 1, 2)
            out.append(f)
        return out

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self._ensure_nchw(self.encoder(x))
        c0, c1, c2, c3 = features
        x = self.decoder1(c3, c2)
        x = self.decoder2(x, c1)
        x = self.decoder3(x, c0)
        x = self.head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO Label Parser
# ═══════════════════════════════════════════════════════════════════════════════

def yolo_to_mask(label_path, img_shape):
    """Convert YOLO polygon annotation to binary mask."""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    p = Path(label_path)
    if not p.exists():
        return mask
    with open(p, "r", encoding="utf-8") as f:
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


# ═══════════════════════════════════════════════════════════════════════════════
# FS Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_curve(mask):
    """Extract per-column LVID curve from binary mask."""
    _, w = mask.shape
    curve = np.zeros(w, dtype=np.float32)
    for col in range(w):
        ys = np.where(mask[:, col] > 0)[0]
        if len(ys) >= 2:
            curve[col] = float(ys[-1] - ys[0])
    return curve


def _interpolate_valid_segment(curve):
    curve = np.asarray(curve, dtype=np.float32)
    valid = np.where(curve > 0)[0]
    if len(valid) < 5:
        return curve.copy(), None
    start, end = int(valid[0]), int(valid[-1])
    seg = curve[start:end + 1].copy()
    sx = np.arange(len(seg))
    sv = np.where(seg > 0)[0]
    if len(sv) >= 2:
        seg = np.interp(sx, sv, seg[sv]).astype(np.float32)
    filled = curve.copy()
    filled[start:end + 1] = seg
    return filled, (start, end)


def _smooth_curve(curve):
    n = len(curve)
    if n < 7:
        return curve.copy()
    w = max(5, int(round(n * 0.15)))
    if w % 2 == 0:
        w += 1
    w = min(w, 15)
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    if w < 5:
        return curve.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(curve, window_length=w, polyorder=3 if w >= 7 else 2)


def _build_alternating_extrema(peaks, valleys, smoothed):
    extrema = [(int(i), "peak") for i in peaks] + [(int(i), "valley") for i in valleys]
    extrema.sort(key=lambda x: x[0])
    alt = []
    for idx, kind in extrema:
        if not alt:
            alt.append((idx, kind))
            continue
        pi, pk = alt[-1]
        if kind != pk:
            alt.append((idx, kind))
            continue
        pv, cv_ = smoothed[pi], smoothed[idx]
        if kind == "peak" and cv_ > pv:
            alt[-1] = (idx, kind)
        elif kind == "valley" and cv_ < pv:
            alt[-1] = (idx, kind)
    return alt


def _pair_peak_valley(curve, smoothed, extrema, min_dist, prom):
    raw = []
    mw_min, mw_max = max(2, min_dist // 2), max(3, min_dist * 3)
    cr = float(np.max(smoothed) - np.min(smoothed))
    md = max(prom * 0.5, cr * 0.05, 1.0)
    for i in range(len(extrema) - 1):
        pi, pk = extrema[i]
        vi, vk = extrema[i + 1]
        if pk != "peak" or vk != "valley":
            continue
        w = vi - pi
        if w < mw_min or w > mw_max:
            continue
        ld, ls = float(curve[pi]), float(curve[vi])
        drop = ld - ls
        if ld <= 0 or ls < 0 or drop <= md or ls >= ld:
            continue
        fs = drop / ld * 100.0
        raw.append({"peak_idx": pi, "valley_idx": vi, "width": w,
                     "lvid_d": ld, "lvid_s": ls, "drop": drop, "fs": fs})
    if not raw:
        return []
    ws = np.array([p["width"] for p in raw], dtype=np.float32)
    ds = np.array([p["drop"] for p in raw], dtype=np.float32)
    med_w, med_d = float(np.median(ws)), float(np.median(ds))
    filt = [p for p in raw if 0.5 * med_w <= p["width"] <= 1.8 * med_w and p["drop"] >= 0.5 * med_d]
    return filt if filt else raw


def analyze_fs(mask, height_mm=None):
    """Analyze FS from a binary mask.

    Args:
        mask: Binary mask (H, W) uint8.
        height_mm: Physical image height in mm (for unit conversion).

    Returns:
        dict with fs_median, pairs, lvid_d_mm, lvid_s_mm, curve, peaks, valleys.
    """
    curve = mask_to_curve(mask)
    filled, vr = _interpolate_valid_segment(curve)
    if vr is None:
        return {"fs_median": None, "pairs": [], "curve": curve,
                "lvid_d_px": None, "lvid_s_px": None, "lvid_d_mm": None, "lvid_s_mm": None}

    start, end = vr
    seg = filled[start:end + 1]
    smoothed_seg = _smooth_curve(seg)
    md = max(6, int(len(seg) * 0.06))
    ss = float(np.std(smoothed_seg))
    sr = float(np.max(smoothed_seg) - np.min(smoothed_seg))
    prom = max(ss * 0.2, sr * 0.08, 1.0)

    pk, _ = find_peaks(smoothed_seg, distance=md, prominence=prom)
    vl, _ = find_peaks(-smoothed_seg, distance=md, prominence=prom)
    pk, vl = pk + start, vl + start

    sf = curve.copy()
    sf[start:end + 1] = smoothed_seg
    ext = _build_alternating_extrema(pk, vl, sf)
    pairs = _pair_peak_valley(filled, sf, ext, md, prom)

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
        (d) Mask overlay with boundary borders on original image
    """
    orig_img = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig_img.shape[:2]
    mm_per_px = height_mm / h
    overlay_color = np.array([244, 177, 131, 128], dtype=np.uint8)

    top_border, bot_border = extract_borders(pred_mask)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(sample_name, fontsize=16, fontweight="bold", y=0.98)

    # (a) Original Image
    ax = axes[0, 0]
    ax.imshow(orig_img)
    ax.set_title("(a) Original Image", fontsize=13, fontweight="bold")
    ax.axis("off")

    # (b) Predicted Mask
    ax = axes[0, 1]
    mask_display = np.zeros((h, w, 3), dtype=np.uint8)
    mask_display[pred_mask > 0] = [244, 177, 131]
    ax.imshow(mask_display)
    fs_text = f"FS = {fs_result['fs_median']:.2f}%" if fs_result['fs_median'] is not None else "FS = N/A"
    ax.set_title(f"(b) Predicted Mask ({fs_text})", fontsize=13, fontweight="bold")
    ax.axis("off")

    # (c) LVID Curve
    ax = axes[1, 0]
    curve = fs_result["curve"]
    x_arr = np.arange(len(curve))
    curve_mm = _px_to_mm(curve, height_mm, h)

    filled, vr = _interpolate_valid_segment(curve)
    if vr is not None:
        start, end = vr
        sm = _smooth_curve(filled[start:end + 1])
        sm_full = curve.copy()
        sm_full[start:end + 1] = sm
    else:
        sm_full = curve.copy()
    sm_mm = _px_to_mm(sm_full, height_mm, h)

    ax.plot(x_arr, curve_mm, color="#C8C8C8", linewidth=1.0, alpha=0.6, label="Raw")
    ax.plot(x_arr, sm_mm, color="#D89C27", linewidth=2.2, label="Smoothed")

    for idx, val in fs_result["peaks"]:
        val_mm = val * mm_per_px
        ax.scatter([idx], [val_mm], color="#1E8A5B", s=70, marker="^", zorder=5)
        ax.annotate(f"D {val_mm:.2f}", (idx, val_mm),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="#1E8A5B", fontweight="bold")

    for idx, val in fs_result["valleys"]:
        val_mm = val * mm_per_px
        ax.scatter([idx], [val_mm], color="#C85250", s=70, marker="v", zorder=5)
        ax.annotate(f"S {val_mm:.2f}", (idx, val_mm),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#C85250", fontweight="bold")

    for pair in fs_result["pairs"]:
        pi, vi = pair["peak_idx"], pair["valley_idx"]
        ax.plot([pi, vi], [pair["lvid_d"] * mm_per_px, pair["lvid_s"] * mm_per_px],
                color="#D89C27", linewidth=1.8, alpha=0.85)

    fs_label = f"FS = {fs_result['fs_median']:.2f}%" if fs_result['fs_median'] is not None else "FS = N/A"
    ax.set_title(f"(c) LVID Curve ({fs_label})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (Column)")
    ax.set_ylabel("LV Diameter (mm)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    # (d) Overlay with borders
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
    """Save measurement results to CSV."""
    fieldnames = ["sample", "LVID_d_px", "LVID_s_px", "LVID_d_mm", "LVID_s_mm", "FS (%)"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def process_single(image_path, model, device, height_mm, output_dir):
    """Process a single M-mode image."""
    sample_name = Path(image_path).stem
    print(f"[*] Processing: {sample_name}")

    pred_mask = predict_mask(model, image_path, device)
    fs_result = analyze_fs(pred_mask, height_mm=height_mm)

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


def main():
    parser = argparse.ArgumentParser(
        description="FS measurement and visualization for TSSwinUNet"
    )
    parser.add_argument("--weight", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output-dir", type=str, default="./fs_output",
                        help="Output directory")
    parser.add_argument("--height-mm", type=float, default=10.8,
                        help="Physical image height in mm (default: 10.8)")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (auto-detected if not set)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Device: {device}")

    print(f"[*] Loading model: {args.weight}")
    model = TSSwinUNet(in_channels=3, num_classes=2)
    state = torch.load(args.weight, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print("[*] Model loaded.")

    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        image_paths = [input_path]

    csv_rows = []
    for img_path in image_paths:
        row = process_single(str(img_path), model, device, args.height_mm, output_dir)
        csv_rows.append(row)

    csv_path = output_dir / "fs_measurements.csv"
    save_csv(csv_rows, csv_path)
    print(f"\n[*] Results saved to {output_dir}")
    print(f"[*] CSV: {csv_path}")


if __name__ == "__main__":
    main()
