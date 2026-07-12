"""Standalone U-Net inference + LPF (Leaf Pixel Fraction) extraction.

No dependency on the training notebook or its pipeline. Requires only:
export/unet_tomato_final.pth (state_dict) alongside this file.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 224
MASK_THRESHOLD = 0.5
BASE_CHANNELS = 32
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STAGE_FOLDER_NAMES = {"fruiting", "flowering", "developing", "seeding"}
STAGE_FILENAME_PREFIXES = {
    "photo_dev": "developing",
    "photo_flow": "flowering",
    "photo_fruit": "fruiting",
    "photo_seed": "seeding",
}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder-decoder U-Net with skip connections (Ronneberger et al.), trained from scratch."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=BASE_CHANNELS):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # logits


def load_unet(checkpoint_path, device=None):
    """Instantiate the U-Net architecture, load a state_dict checkpoint, return it in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=BASE_CHANNELS)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _preprocess(image_path, img_size=IMG_SIZE):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float()
    return img_t


@torch.no_grad()
def compute_lpf(model, image_path, threshold=MASK_THRESHOLD):
    """Run U-Net inference on a single image and return (LPF, predicted binary mask).

    LPF = (predicted leaf pixels) / (total pixels)
    """
    device = next(model.parameters()).device
    img_t = _preprocess(image_path).to(device)

    logits = model(img_t)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > threshold).float().cpu().squeeze().numpy()

    lpf = float(pred_mask.sum() / pred_mask.size)
    return lpf, pred_mask


def _infer_growth_stage(image_path):
    """Best-effort stage label: parent folder name if it's a known stage, else filename prefix."""
    parent = image_path.parent.name.lower()
    if parent in STAGE_FOLDER_NAMES:
        return parent
    fname = image_path.name.lower()
    for prefix, stage in STAGE_FILENAME_PREFIXES.items():
        if prefix in fname:
            return stage
    return "unknown"


def compute_lpf_batch(model, image_dir, output_csv_path, threshold=MASK_THRESHOLD,
                       extensions=(".jpg", ".jpeg", ".png")):
    """Run compute_lpf over every image in image_dir (recursive), save filename/growth_stage/lpf CSV."""
    import csv

    image_dir = Path(image_dir)
    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in extensions)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "growth_stage", "lpf"])
        for path in image_paths:
            lpf, _ = compute_lpf(model, path, threshold=threshold)
            writer.writerow([path.name, _infer_growth_stage(path), lpf])

    print(f"wrote {len(image_paths)} rows to {output_csv_path}")
    return output_csv_path


if __name__ == "__main__":
    ckpt_path = Path(__file__).parent / "unet_tomato_final.pth"
    model = load_unet(ckpt_path)
    print("model loaded, params:", sum(p.numel() for p in model.parameters()))
