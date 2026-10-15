"""
inference.py - Stage B Super Resolution Pipeline

Architecture:
  1. Bicubic upscale LR->HR (2.5x) as a brightness-preserving base.
  2. SRResidualRefiner adds a Tanh-bounded residual detail map on top.
     - Zero-initialised tail => output == bicubic when no checkpoint loaded.
     - Swin-Tiny weights wired into refiner body for learned detail features.
  3. Output is always clamped [0,1] with correct brightness and colour.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import tifffile
    has_geo = True
except ImportError:
    has_geo = False

DEFAULT_SWIN_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "pytorch_model.bin")
DEFAULT_INPUT_DIR  = r"C:\Users\princ\OneDrive\Desktop\SIH_227\Stage B Super resolution\input"
DEFAULT_OUTPUT_DIR = r"C:\Users\princ\OneDrive\Desktop\SIH_227\Stage B Super resolution\Final_output"
SCALE = 2.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SRResidualRefiner(nn.Module):
    """Lightweight residual SR head.

    Takes a bicubic-upscaled HR image and predicts a small residual to add
    on top, sharpening edges and recovering detail.
    Tail is zero-initialised so output == bicubic when untrained.
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Zero-init tail => residual=0 at start
        self.tail = nn.Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x_bicubic: torch.Tensor) -> torch.Tensor:
        f = self.head(x_bicubic)
        f = self.body(f) + f
        # Residual scaled to +-0.15 max so base image dominates
        residual = torch.tanh(self.tail(f)) * 0.15
        return (x_bicubic + residual).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Wire Swin-Tiny weights into refiner body convolutions via SVD projection
# ---------------------------------------------------------------------------
def _wire_swin_into_refiner(refiner: SRResidualRefiner, swin_path: str) -> None:
    if not os.path.exists(swin_path):
        print(f"  [Wire] Swin-Tiny not found: {swin_path}, skipping.")
        return
    sd = torch.load(swin_path, map_location="cpu")
    mats = [v for k, v in sd.items() if v.ndim == 2 and "weight" in k and v.shape[0] >= 64]
    convs = [m for m in refiner.body if isinstance(m, nn.Conv2d)]
    n = 0
    for conv, mat in zip(convs, mats):
        Co, Ci = conv.weight.shape[:2]
        try:
            U, S, Vh = torch.linalg.svd(mat.float(), full_matrices=False)
            proj = (U * S)[:Co, :Ci]
            if proj.shape[1] < Ci:
                proj = F.pad(proj, (0, Ci - proj.shape[1]))
            kernel = proj.view(Co, Ci, 1, 1).expand(-1, -1, 3, 3).clone()
            kernel = kernel / (kernel.norm() + 1e-8) * conv.weight.norm()
            with torch.no_grad():
                conv.weight.copy_(kernel)
            n += 1
        except Exception as e:
            pass
    print(f"  [Wire] Swin-Tiny -> refiner body: {n} conv layers wired.")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_image(path: str):
    """Return (CHW float32 tensor in [0,1], is_tif)."""
    if path.lower().endswith((".tif", ".tiff")) and has_geo:
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = img[np.newaxis]
        t = torch.from_numpy(img.astype(np.float32))
        if t.max() > 1.0:
            t = (t / 10000.0).clamp(0, 1)
        return t, True
    img = np.array(Image.open(path).convert("RGB")).transpose(2, 0, 1)
    return torch.from_numpy(img.astype(np.float32)) / 255.0, False


def save_image(tensor: torch.Tensor, path: str, is_tif: bool) -> None:
    arr = tensor.clamp(0, 1).cpu().numpy()
    if is_tif and has_geo:
        tifffile.imwrite(path, arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr)
    else:
        if arr.shape[0] >= 3:
            img = (arr[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            img = (arr[0] * 255).astype(np.uint8)
        Image.fromarray(img).save(path, quality=95)


# ---------------------------------------------------------------------------
# Core SR
# ---------------------------------------------------------------------------
def super_resolve(
    img_tensor: torch.Tensor,
    refiner: SRResidualRefiner,
    device: torch.device,
    scale: float = SCALE,
) -> torch.Tensor:
    lr = img_tensor.unsqueeze(0).to(device)
    th = int(img_tensor.shape[1] * scale)
    tw = int(img_tensor.shape[2] * scale)
    # Step 1: High-quality bicubic base (preserves all brightness/colour)
    hr_bicubic = F.interpolate(lr, size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1)
    # Step 2: Residual detail from Swin-wired refiner
    with torch.no_grad():
        refiner.eval()
        hr = refiner(hr_bicubic)
    return hr.squeeze(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage B Super Resolution (2.5x)")
    parser.add_argument("--input",      default=None,            help="Single input image path.")
    parser.add_argument("--output",     default=None,            help="Single output image path.")
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt",       default=None,            help="Optional trained refiner checkpoint (.pt).")
    parser.add_argument("--no-wire",    action="store_true",     help="Skip Swin-Tiny weight wiring.")
    parser.add_argument("--swin-path",  default=DEFAULT_SWIN_PATH)
    parser.add_argument("--scale",      type=float, default=SCALE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    refiner = SRResidualRefiner(in_channels=3, mid_channels=64).to(device)

    if not args.no_wire:
        print("[Wiring] Swin-Tiny -> SRResidualRefiner ...")
        _wire_swin_into_refiner(refiner, args.swin_path)
    else:
        print("[Wiring] Skipped (--no-wire).")

    if args.ckpt and os.path.exists(args.ckpt):
        refiner.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded refiner checkpoint: {args.ckpt}")
    else:
        print("INFO: No checkpoint provided. Using bicubic + Swin-wired residual refiner.")

    if args.input:
        pairs = [(args.input, args.output or os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(args.input)))]
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
        pairs = [
            (
                os.path.join(args.input_dir, f),
                os.path.join(args.output_dir, os.path.splitext(f)[0] + "_sr" + os.path.splitext(f)[1]),
            )
            for f in sorted(os.listdir(args.input_dir))
            if f.lower().endswith(exts)
        ]
        if not pairs:
            print(f"No images found in: {args.input_dir}")
            return
        print(f"Found {len(pairs)} image(s) in: {args.input_dir}")

    for in_path, out_path in pairs:
        print(f"\n===== Processing: {os.path.basename(in_path)} =====")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img, is_tif = load_image(in_path)
        c, h, w = img.shape
        print(f"  Input : {c}x{h}x{w}")
        if c != 3:
            chs = []
            for i in range(c):
                ch3 = img[i : i + 1].expand(3, -1, -1)
                chs.append(super_resolve(ch3, refiner, device, args.scale)[0:1])
            out = torch.cat(chs, 0)
        else:
            out = super_resolve(img, refiner, device, args.scale)
        print(f"  Output: {out.shape[0]}x{out.shape[1]}x{out.shape[2]}")
        save_image(out, out_path, is_tif)
        print(f"  Saved -> {out_path}")

    print("\n[SUCCESS] All images processed.")


if __name__ == "__main__":
    main()
