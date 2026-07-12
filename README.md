# Tomato Leaf Segmentation — U-Net (Branch 01)

Binary leaf/background segmentation model for the **Tomato Plant Growth Stage Monitoring System**. This is Branch 01 of a dual-branch fusion architecture — it segments tomato leaves from RGB images and computes a **Leaf Pixel Fraction (LPF)** scalar, which is later fused with ResNet50 classification features (Branch 02) to predict the plant's growth stage.

This repository is trained and evaluated **independently** of the ResNet50 and fusion projects. It exports a standalone, self-contained artifact that downstream projects consume without any dependency on this codebase.

---

## Overview

| | |
|---|---|
| **Task** | Binary semantic segmentation (leaf vs. background) |
| **Architecture** | U-Net, encoder-decoder with skip connections, trained from scratch (no pretrained backbone) |
| **Framework** | PyTorch |
| **Input** | 224 × 224 × 3 RGB image |
| **Output** | 224 × 224 × 1 binary mask (sigmoid, threshold 0.5) |
| **Downstream use** | Leaf Pixel Fraction (LPF) scalar feature for fusion classifier |

---

## Dataset

- 982 self-collected tomato plant images across four growth stages (seeding, developing, flowering, fruiting)
- Annotated on Roboflow with polygon segmentation masks, exported in COCO format
- Split into `train/` (687), `valid/` (135), `test/` (160) folders
- COCO polygons converted to binary PNG masks prior to training (see `unet_pipeline.ipynb`, Task 2)
- The authoritative train/valid/test split is recorded in [`export/lpf_full_dataset.csv`](export/lpf_full_dataset.csv) and should be treated as the single source of truth by any downstream project (e.g. the fusion pipeline) to avoid data leakage from re-splitting
- Raw and processed image files are not committed to this repo (regeneratable, see [Reproducing training](#reproducing-training)) — only code and exported artifacts are versioned here

## Architecture

Custom U-Net (Ronneberger et al.), `base_channels=32`:

```
INPUT        224 × 224 × 3 RGB image
ENC 1        Conv → Conv → MaxPool   [32 filters]
ENC 2        Conv → Conv → MaxPool   [64 filters]
ENC 3        Conv → Conv → MaxPool   [128 filters]
ENC 4        Conv → Conv → MaxPool   [256 filters]
BOTTLENECK   Conv → Conv             [512 filters]
DEC 4        UpSample + Skip 4 → Conv → Conv   [256 filters]
DEC 3        UpSample + Skip 3 → Conv → Conv   [128 filters]
DEC 2        UpSample + Skip 2 → Conv → Conv   [64 filters]
DEC 1        UpSample + Skip 1 → Conv → Conv   [32 filters]
OUTPUT       Conv(1) + Sigmoid → binary mask
```

7.77M parameters total.

**Loss:** combined Binary Cross-Entropy + Dice Loss (0.5 / 0.5), to balance pixel-level accuracy with mask overlap under leaf/background class imbalance.

**Augmentation:** random horizontal flip, vertical flip, and 90°-multiple rotation, applied identically to image and mask.

## Results

Trained for 60 epochs (Adam, lr=1e-4, `ReduceLROnPlateau` on validation Dice), best checkpoint selected by validation Dice score (epoch 58/60).

| Metric | Validation | Test (held out) |
|---|---|---|
| Dice score | 0.8127 | 0.7572 |
| IoU | 0.7241 | 0.6650 |

Per-growth-stage test Dice/IoU (see `export/metadata.json` for full detail):

| Stage | Dice | IoU | n |
|---|---|---|---|
| seeding | 0.858 | 0.776 | 38 |
| developing | 0.774 | 0.687 | 41 |
| flowering | 0.729 | 0.631 | 60 |
| fruiting | 0.625 | 0.519 | 21 |

`fruiting` is consistently the weakest stage — smallest class (21 test images) and leaves are frequently occluded by fruit.

> Segmentation quality is evaluated relative to its purpose — producing a reliable LPF scalar for the fusion classifier — not as a standalone segmentation benchmark.

## Repository structure

```
.
├── unet_pipeline.ipynb             training, evaluation, and LPF extraction (single notebook, PyTorch)
├── checkpoints/                    unet_best.pth / unet_final.pth (full training state dicts)
├── outputs/                        test-set scores, training curves, qualitative examples
├── export/                         standalone, versioned model artifacts (see below)
│   ├── unet_tomato_final.pth       state_dict only, no pickled model object
│   ├── metadata.json               architecture, I/O shapes, normalization, threshold, metrics
│   ├── unet_lpf_extractor.py       standalone loader — no dependency on training code
│   └── lpf_full_dataset.csv        filename, growth_stage, split, lpf — for all 982 images
├── requirements.txt
└── README.md
```

`tomato final dataset/` (raw Roboflow export) and `processed/` (merged images + converted masks) are used locally during training but are not committed — see [Reproducing training](#reproducing-training).

## Using the exported model

The `export/` folder is self-contained and can be copied into another project (e.g. the fusion pipeline) without any dependency on this repo's training code.

```python
from unet_lpf_extractor import load_unet, compute_lpf, compute_lpf_batch

# Load the trained model
model = load_unet("export/unet_tomato_final.pth")

# Compute LPF for a single image — works on ANY image, annotated or not
lpf, mask = compute_lpf(model, "path/to/image.jpg")
print(lpf)  # float in [0, 1]

# Compute LPF for a whole directory and save results to CSV
compute_lpf_batch(model, "path/to/images/", output_csv_path="lpf_output.csv")
```

**Note:** annotation is only required during training. At inference time, U-Net predicts its own segmentation mask, so this works on completely new, unannotated images.

## Reproducing training

```bash
pip install -r requirements.txt
```

Place the Roboflow COCO export under `tomato final dataset/{train,valid,test}/{fruiting,flowering,developing,seeding}/`, each split folder containing a shared `_annotations.coco.json`. Then open and run `unet_pipeline.ipynb` top to bottom — Task 2 converts the COCO polygons into merged `processed/{split}/images` + `masks` folders before training starts.

Key dependencies: `torch`, `torchvision`, `opencv-python`, `pycocotools`.

## Known limitations / open work

- Aggregate Dice (0.81 validation, 0.76 test) has room to improve; a pretrained encoder backbone (e.g. ResNet34 via `segmentation_models_pytorch`) is a candidate next iteration
- Segmentation is weakest at the fruiting stage (occlusion by fruit, smallest class) — see per-stage breakdown in `export/metadata.json`
- This repo's train/valid/test split is enforced via `lpf_full_dataset.csv`; a data-leakage audit across the broader fusion project is separate, ongoing work

## Part of

This repository is Branch 01 of the [Tomato Plant Growth Stage Monitoring System](https://tomatoresearch.vercel.app/) research project. See also: ResNet50 classification branch, and the fusion project combining both branches.

## License

[Add your license here]
