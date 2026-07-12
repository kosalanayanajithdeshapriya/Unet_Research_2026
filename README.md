# Tomato Leaf Segmentation — U-Net (Branch 01)

Binary leaf/background segmentation model for the **Tomato Plant Growth Stage Monitoring System**. This is Branch 01 of a dual-branch fusion architecture — it segments tomato leaves from RGB images and computes a **Leaf Pixel Fraction (LPF)** scalar, which is later fused with ResNet50 classification features (Branch 02) to predict the plant's growth stage.

This repository is trained and evaluated **independently** of the ResNet50 and fusion projects. It exports a standalone, self-contained artifact that downstream projects consume without any dependency on this codebase.

---

## Overview

| | |
|---|---|
| **Task** | Binary semantic segmentation (leaf vs. background) |
| **Architecture** | U-Net, encoder-decoder with skip connections |
| **Framework** | PyTorch |
| **Input** | 224 × 224 × 3 RGB image |
| **Output** | 224 × 224 × 1 binary mask (sigmoid, threshold 0.5) |
| **Downstream use** | Leaf Pixel Fraction (LPF) scalar feature for fusion classifier |

---

## Dataset

- ~982 self-collected tomato plant images across four growth stages (seeding, developing, flowering, fruiting)
- Annotated on Roboflow with polygon segmentation masks, exported in COCO format
- Split into `train/`, `valid/`, `test/` folders
- COCO polygons converted to binary PNG masks prior to training
- The authoritative train/valid/test split is recorded in [`export/lpf_full_dataset.csv`](export/lpf_full_dataset.csv) and should be treated as the single source of truth by any downstream project (e.g. the fusion pipeline) to avoid data leakage from re-splitting

## Architecture

```
INPUT        224 × 224 × 3 RGB image
ENC 1        Conv → Conv → MaxPool   [64 filters]
ENC 2        Conv → Conv → MaxPool   [128 filters]
ENC 3        Conv → Conv → MaxPool   [256 filters]
ENC 4        Conv → Conv → MaxPool   [512 filters]
BOTTLENECK   Conv → Conv             [1024 filters]
DEC 4        UpSample + Skip 4 → Conv → Conv   [512 filters]
DEC 3        UpSample + Skip 3 → Conv → Conv   [256 filters]
DEC 2        UpSample + Skip 2 → Conv → Conv   [128 filters]
DEC 1        UpSample + Skip 1 → Conv → Conv   [64 filters]
OUTPUT       Conv(1) + Sigmoid → binary mask
```

**Loss:** combined Binary Cross-Entropy + Dice Loss, to balance pixel-level accuracy with mask overlap under leaf/background class imbalance.

## Results

Trained for 60 epochs, best checkpoint selected by validation Dice score.

| Metric | Train | Validation |
|---|---|---|
| Loss (BCE + Dice) | 0.1580 | 0.1471 |
| Dice score | 0.8027 | **0.8134** (best) |
| IoU | 0.7253 | 0.7243 |

Train and validation metrics are closely matched, indicating the model generalizes well rather than overfitting on this dataset size. Per-growth-stage Dice/IoU breakdown is available in `export/metadata.json`.

> Segmentation quality is evaluated relative to its purpose — producing a reliable LPF scalar for the fusion classifier — not as a standalone segmentation benchmark.

## Repository structure

```
.
├── notebooks/                      training and evaluation notebooks
├── data/                           train/valid/test images + converted masks
├── export/                         standalone, versioned model artifacts (see below)
│   ├── unet_tomato_final.pth       state_dict only, no pickled model object
│   ├── metadata.json               architecture, I/O shapes, normalization, threshold, metrics
│   ├── unet_lpf_extractor.py       standalone loader — no dependency on training code
│   └── lpf_full_dataset.csv        filename, growth_stage, split, lpf — for all 982 images
└── README.md
```

## Using the exported model

The `export/` folder is self-contained and can be copied into another project (e.g. the fusion pipeline) without any dependency on this repo's training code.

```python
from unet_lpf_extractor import load_unet, compute_lpf, compute_lpf_batch

# Load the trained model
model = load_unet("export/unet_tomato_final.pth")

# Compute LPF for a single image — works on ANY image, annotated or not
lpf = compute_lpf(model, "path/to/image.jpg")
print(lpf)  # float in [0, 1]

# Compute LPF for a whole directory and save results to CSV
compute_lpf_batch(model, "path/to/images/", output_csv_path="lpf_output.csv")
```

**Note:** annotation is only required during training. At inference time, U-Net predicts its own segmentation mask, so this works on completely new, unannotated images.

## Reproducing training

```bash
pip install -r requirements.txt
jupyter notebook notebooks/train_unet.ipynb
```

Key dependencies: `torch`, `torchvision`, `albumentations`, `pycocotools`, `imagehash`.

## Known limitations / open work

- Aggregate Dice (0.81) has room to improve; a pretrained encoder backbone (e.g. ResNet34 via `segmentation_models_pytorch`) is planned as the next iteration
- Segmentation difficulty likely varies by growth stage (e.g. occlusion at the fruiting stage) — see per-stage breakdown in `export/metadata.json`
- A data-leakage audit (perceptual hashing across train/valid/test) is in progress for the broader project; this repo's split is already enforced via `lpf_full_dataset.csv`

## Part of

This repository is Branch 01 of the [Tomato Plant Growth Stage Monitoring System](https://tomatoresearch.vercel.app/) research project. See also: ResNet50 classification branch, and the fusion project combining both branches.

## License

[Add your license here]
