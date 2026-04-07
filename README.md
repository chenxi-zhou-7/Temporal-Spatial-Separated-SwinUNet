<img width="1600" height="802" alt="image" src="https://github.com/user-attachments/assets/95b951e7-875b-4876-b8e1-a57f22d88a4f" /># TSSwinUNet - Ultrasound M-Mode Cardiac Segmentation and FS Measurement

Automated left ventricle segmentation and Fractional Shortening (FS) calculation from ultrasound M-Mode images using TSSwinUNet.

## Architecture
###TSSwinUnet
<img width="1600" height="802" alt="image" src="https://github.com/user-attachments/assets/810e3af8-0587-4099-90a8-40f2fc2d4079" />

###PTSBlock
<img width="1073" height="533" alt="image" src="https://github.com/user-attachments/assets/51ec70be-5bf2-41da-88df-60960018c176" />


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

### Visualization 
<img width="2780" height="1970" alt="0306_A16-2_FS_Visualization" src="https://github.com/user-attachments/assets/708aec0d-6a78-4d1f-9113-3a56e624f935" />

Four-panel figure per sample:

- **(a) Original Image** - Raw ultrasound M-Mode input
- **(b) Predicted Mask** - Segmentation overlay with FS value
- **(c) LVID Curve** - Diameter over time with D (diastole) and S (systole) markers and cycle pairings
- **(d) Annotated Overlay** - Original image with segmentation overlay, IVS/LVPW borders, and per-cycle LVID_d / LVID_s measurements

## License

Proprietary. All rights reserved.
