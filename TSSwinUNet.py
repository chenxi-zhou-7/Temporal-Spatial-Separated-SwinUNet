#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSSwinUNet: Temporal-Spatial Swin UNet for M-mode Echocardiography Segmentation

A novel architecture that decouples spatial (anatomical depth) and temporal
(cardiac rhythm) features via the Physiological Time-Space (PTS) block,
designed specifically for M-mode echocardiography segmentation.

Architecture:
    - Encoder: Swin Transformer (pretrained on ImageNet-1K)
    - Decoder: 3-stage progressive upsampling with PTS blocks at each stage
    - Head: 4x bilinear upsampling + 1x1 convolution

Key Innovation:
    PTS Block decomposes features along orthogonal axes:
      - Spatial branch: (K_s, 1) convolution along Y-axis (depth) + BatchNorm
      - Temporal branch: (1, K_t) convolution along X-axis (time) + InstanceNorm
      - Gated fusion: temporal attention modulates spatial features

Loss Function (CombinedPhysiologyLoss):
    L = alpha * L_Dice + beta * L_CE + gamma * L_Freq + delta * L_Topo

    - L_Dice: Soft Dice loss for region overlap
    - L_CE: Cross-entropy loss for pixel-wise classification
    - L_Freq: Frequency-domain consistency (FFT amplitude spectrum matching)
    - L_Topo: Topological continuity (velocity + smoothness constraints)

Dependencies:
    pip install torch timm

Usage:
    from TSSwinUNet import TSSwinUNet, CombinedPhysiologyLoss

    model = TSSwinUNet(in_channels=3, num_classes=2)
    criterion = CombinedPhysiologyLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

    x = torch.randn(2, 3, 224, 224)
    out = model(x)                          # (2, 2, 224, 224)
    loss, components = criterion(out, target)  # target: (2, 224, 224) LongTensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

try:
    import timm
except ImportError:
    raise ImportError("timm is required. Install with: pip install timm")


# ═══════════════════════════════════════════════════════════════════════════════
# PTS Block: Physiological Time-Space Decoupling
# ═══════════════════════════════════════════════════════════════════════════════

class PTSBlock(nn.Module):
    """
    Physiological Time-Space Block.

    Decomposes 2D feature maps along orthogonal physiological axes:
      - Spatial branch (Y-axis / anatomical depth): (K_s, 1) conv + BatchNorm
        preserves absolute scale information across the depth dimension.
      - Temporal branch (X-axis / cardiac rhythm): (1, K_t) conv + InstanceNorm
        normalizes per-instance to extract pure rhythmic phase patterns.

    A gated fusion mechanism generates attention weights from the temporal
    stream that modulate the spatial features, enabling anatomy-aware
    temporal attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        spatial_kernel: int = 7,
        temporal_kernel: int = 7,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        mid_ch = in_channels // reduction if in_channels >= reduction else in_channels

        # --- Spatial branch: (K_s, 1) convolution along depth (Y) ---
        self.spatial_conv = nn.Conv2d(
            in_channels, self.out_channels,
            kernel_size=(spatial_kernel, 1),
            padding=(spatial_kernel // 2, 0),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(self.out_channels)
        self.spatial_act = nn.GELU()

        # --- Temporal branch: (1, K_t) convolution along time (X) ---
        self.temporal_conv = nn.Conv2d(
            in_channels, self.out_channels,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.temporal_in = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.temporal_act = nn.GELU()

        # --- Spatiotemporal gating: temporal → attention over spatial ---
        self.gate_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, self.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # --- Fusion + residual ---
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        spatial = self.spatial_act(self.spatial_bn(self.spatial_conv(x)))
        temporal = self.temporal_act(self.temporal_in(self.temporal_conv(x)))

        gate = self.gate_conv(temporal)
        gated = spatial * gate

        out = self.fusion_bn(self.fusion_conv(gated))
        return out + self.residual_proj(identity)


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsample -> Skip concat -> ConvBlock -> PTS block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        spatial_kernel: int = 7,
        temporal_kernel: int = 7,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(ConvBlock(out_channels + skip_channels, out_channels))
        self.pts = PTSBlock(
            out_channels, out_channels,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.pts(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# TSSwinUNet: Main Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class TSSwinUNet(nn.Module):
    """
    Temporal-Spatial Swin UNet.

    Encoder-decoder architecture with PTS blocks at every decoder stage
    for M-mode echocardiography segmentation.

    Args:
        in_channels: Number of input channels (default 3 for RGB).
        num_classes: Number of output classes (default 2 for binary segmentation).
        encoder_name: Timm encoder name (default swin_tiny_patch4_window7_224).
        spatial_kernel: PTS spatial convolution kernel size along depth axis.
        temporal_kernel: PTS temporal convolution kernel size along time axis.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_name: str = 'swin_tiny_patch4_window7_224',
        spatial_kernel: int = 7,
        temporal_kernel: int = 7,
    ):
        super().__init__()

        # --- Encoder: Swin Transformer ---
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            in_chans=in_channels,
        )
        enc_ch = self.encoder.feature_info.channels()
        # swin_tiny_patch4_window7_224: [96, 192, 384, 768]

        # --- Decoder: 3 stages with PTS blocks ---
        self.decoder1 = DecoderBlock(
            enc_ch[3], enc_ch[2], enc_ch[2],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )
        self.decoder2 = DecoderBlock(
            enc_ch[2], enc_ch[1], enc_ch[1],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )
        self.decoder3 = DecoderBlock(
            enc_ch[1], enc_ch[0], enc_ch[0],
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel,
        )

        # --- Head: 4x upsample + 1x1 conv ---
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(enc_ch[0], num_classes, kernel_size=1),
        )

    def _ensure_nchw(self, features):
        """Guarantee NCHW format (timm may return NHWC for some backbones)."""
        out = []
        for f in features:
            if f.shape[1] == f.shape[2] and f.shape[3] != f.shape[1]:
                f = f.permute(0, 3, 1, 2)
            out.append(f)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image.
        Returns:
            (B, num_classes, H, W) raw logits.
        """
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
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

class FrequencyConsistencyLoss(nn.Module):
    """
    Frequency-domain consistency loss.

    Projects 2D segmentation maps onto 1D temporal curves (by summing along
    the depth axis), then matches FFT amplitude spectra between prediction
    and ground truth. Enforces correct cardiac cycle frequency learning.

    This ensures that the predicted segmentation preserves the periodicity
    and rhythmic characteristics of cardiac motion.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Project to 1D temporal curves along depth axis
        pred_curve = pred.sum(dim=2)
        target_curve = target.sum(dim=2)

        # Normalize to eliminate amplitude bias
        pred_curve = pred_curve / (pred_curve.sum(dim=-1, keepdim=True) + 1e-8)
        target_curve = target_curve / (target_curve.sum(dim=-1, keepdim=True) + 1e-8)

        # FFT amplitude spectra
        pred_amp = torch.abs(torch.fft.rfft(pred_curve, dim=-1))
        target_amp = torch.abs(torch.fft.rfft(target_curve, dim=-1))

        # Normalize spectra
        pred_amp = pred_amp / (pred_amp.sum(dim=-1, keepdim=True) + 1e-8)
        target_amp = target_amp / (target_amp.sum(dim=-1, keepdim=True) + 1e-8)

        return F.l1_loss(pred_amp, target_amp, reduction=self.reduction)


class ContinuityTopologyLoss(nn.Module):
    """
    Topological continuity loss.

    Enforces smooth myocardial motion by matching temporal velocity profiles
    (first derivative of depth curves) between prediction and ground truth,
    plus a smoothness penalty on the second derivative to suppress
    non-physiological high-frequency jitter.

    This preserves the topological consistency of the segmented boundary
    across consecutive time frames.
    """

    def __init__(self, reduction: str = 'mean', smoothness_weight: float = 0.1):
        super().__init__()
        self.reduction = reduction
        self.smoothness_weight = smoothness_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Project to 1D temporal curves
        pred_curve = pred.sum(dim=2)
        target_curve = target.sum(dim=2)

        pred_curve = pred_curve / (pred_curve.max() + 1e-8)
        target_curve = target_curve / (target_curve.max() + 1e-8)

        # Velocity matching (first derivative)
        pred_vel = torch.diff(pred_curve, dim=-1)
        target_vel = torch.diff(target_curve, dim=-1)
        vel_loss = F.mse_loss(pred_vel, target_vel, reduction=self.reduction)

        # Smoothness penalty (second derivative)
        pred_acc = torch.diff(pred_vel, dim=-1)
        smooth_loss = torch.mean(pred_acc ** 2)

        return vel_loss + self.smoothness_weight * smooth_loss


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return (1 - (2 * intersection + self.smooth) / (union + self.smooth)).mean()


class CombinedPhysiologyLoss(nn.Module):
    """
    Physics-informed combined loss for M-mode echocardiography segmentation.

    L = alpha * L_Dice + beta * L_CE + gamma * L_Freq + delta * L_Topo

    Components:
        - L_Dice: Soft Dice loss for region overlap maximization.
        - L_CE: Cross-entropy loss for pixel-wise classification.
        - L_Freq: Frequency-domain consistency loss — projects segmentation
          maps onto 1D temporal curves and matches FFT amplitude spectra,
          enforcing correct cardiac cycle frequency learning.
        - L_Topo: Topological continuity loss — matches temporal velocity
          profiles (first derivative) between prediction and ground truth,
          with a smoothness penalty on the second derivative to suppress
          non-physiological high-frequency jitter.

    Args:
        alpha: Dice loss weight (default 1.0).
        beta: Cross-entropy loss weight (default 1.0).
        gamma: Frequency consistency loss weight (default 1.0).
        delta: Topological continuity loss weight (default 1.0).
        num_classes: Number of segmentation classes (default 2).

    Returns:
        Tuple of (total_loss, components_dict).
            components_dict keys: dice, ce, freq, topo, total
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.num_classes = num_classes

        self.dice_loss = DiceLoss()
        self.freq_loss = FrequencyConsistencyLoss()
        self.topo_loss = ContinuityTopologyLoss()

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, num_classes, H, W) raw model output.
            target: (B, H, W) ground truth class indices.
        Returns:
            (total_loss, components_dict)
        """
        prob = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()

        dice = self.dice_loss(prob, target_oh)
        ce = F.cross_entropy(logits, target.long())

        pred_fg = prob[:, 1:2]
        target_fg = target_oh[:, 1:2]
        freq = self.freq_loss(pred_fg, target_fg)
        topo = self.topo_loss(pred_fg, target_fg)

        total = (self.alpha * dice + self.beta * ce +
                 self.gamma * freq + self.delta * topo)

        components = {
            'dice': dice.item(),
            'ce': ce.item(),
            'freq': freq.item(),
            'topo': topo.item(),
            'total': total.item(),
        }
        return total, components


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 60)
    print("TSSwinUNet - Unit Tests")
    print("=" * 60)

    # PTS Block
    blk = PTSBlock(64, 64)
    x = torch.randn(2, 64, 32, 64)
    y = blk(x)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)
    print("[PASS] PTSBlock")

    # TSSwinUNet
    model = TSSwinUNet(in_channels=3, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2, 224, 224)
    params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] TSSwinUNet (params: {params/1e6:.2f}M)")

    # CombinedPhysiologyLoss
    criterion = CombinedPhysiologyLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    logits = torch.randn(2, 2, 64, 128, requires_grad=True)
    target = torch.randint(0, 2, (2, 64, 128))
    loss, components = criterion(logits, target)
    assert loss.requires_grad
    assert set(components.keys()) == {'dice', 'ce', 'freq', 'topo', 'total'}
    loss.backward()
    print(f"[PASS] CombinedPhysiologyLoss (total={components['total']:.4f})")

    # End-to-end gradient flow
    model = TSSwinUNet(in_channels=3, num_classes=2)
    criterion = CombinedPhysiologyLoss()
    x = torch.randn(2, 3, 224, 224)
    target = torch.randint(0, 2, (2, 224, 224))
    out = model(x)
    loss, _ = criterion(out, target)
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    assert has_grad == total
    print(f"[PASS] Gradient flow ({has_grad}/{total} params)")

    # Weight round-trip
    torch.manual_seed(0)
    model = TSSwinUNet(in_channels=3, num_classes=2)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y1 = model(x)
    model2 = TSSwinUNet(in_channels=3, num_classes=2)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y1, y2, atol=1e-6)
    print("[PASS] Weight save/load round-trip")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
