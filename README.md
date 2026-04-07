# TSSwinUNet - Ultrasound M-Mode Cardiac Segmentation and FS Measurement

Automated left ventricle segmentation and Fractional Shortening (FS) calculation from ultrasound M-Mode images using TSSwinUNet.

## Files

| File | Description |
|------|-------------|
| `TSSwinUNet.py` | Model architecture definition |
| `calculate_fs.py` | FS computation module |
| `visualize_fs.py` | Inference, FS calculation, and visualization |
| `requirements.txt` | Python dependencies |

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- See `requirements.txt` for full dependency list

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python visualize_fs.py --weight weight.pth --input image.tif --output-dir ./results --height-mm 10.8
```

`--input` accepts a single image or a directory (auto-detects `.tif`, `.png`, `.jpg`).

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--weight` | Yes | Path to model weights (`.pth`) |
| `--input` | Yes | Input image or a directory of images |
| `--output-dir` | Yes | Output directory |
| `--height-mm` | Yes | Physical height of the image in mm |

## Output

### CSV Table (`fs_measurements.csv`)

| Column | Description |
|--------|-------------|
| `sample` | Sample identifier |
| `LVID_d_px` | LV internal diameter at diastole (pixels) |
| `LVID_s_px` | LV internal diameter at systole (pixels) |
| `LVID_d_mm` | LV internal diameter at diastole (mm) |
| `LVID_s_mm` | LV internal diameter at systole (mm) |
| `FS (%)` | Fractional shortening |

### Visualization (`{sample}_FS_Visualization.png/pdf`)

Four-panel figure per sample:

- **(a) Original Image** - Raw ultrasound M-Mode input
- **(b) Predicted Mask** - Segmentation overlay with FS value
- **(c) LVID Curve** - Diameter over time with D (diastole) and S (systole) markers and cycle pairings
- **(d) Annotated Overlay** - Original image with segmentation overlay, IVS/LVPW borders, and per-cycle LVID_d / LVID_s measurements

## License

Proprietary. All rights reserved.
