"""Prithvi-EO-1.0-100M Masked Autoencoder (MAE) Foundation Model Inference.

This script:
1. Dynamically imports the model definition (PrithviMAE) directly from Hugging Face
   ('ibm-nasa-geospatial/Prithvi-100M') using the Hugging Face API and token.
2. Loads pretrained weights from the local checkpoint file (.pt) such as
   'model/Prithvi_EO_V1_100M.pt'.
3. Performs reconstruction inference across temporal and spatial dimensions on
   multi-band geospatial GeoTIFFs (HLS reflectance bands B02, B03, B04, B05, B06, B07).
4. Handles sliding-window tiling for arbitrary spatial resolutions, recombines patches,
   and exports both full-band GeoTIFFs and contrast-enhanced RGB visualizations.

Example usage:
--------------
# Run inference with default weights and example files from Hugging Face:
python model/inference_Prithvi.py --download-examples

# Run inference on custom GeoTIFF files:
python model/inference_Prithvi.py \
    --checkpoint model/Prithvi_EO_V1_100M.pt \
    --data-files path/to/t1.tif path/to/t2.tif path/to/t3.tif \
    --output-dir output_dir \
    --rgb-outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import rasterio
import torch
import yaml
from einops import rearrange
from huggingface_hub import hf_hub_download

# Default constants
DEFAULT_HF_TOKEN = "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"
DEFAULT_REPO_ID = "ibm-nasa-geospatial/Prithvi-100M"
DEFAULT_CHECKPOINT_NAME = "Prithvi_EO_V1_100M.pt"

# Nodata values specified by the Prithvi reference specification
NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99.9

# Fallback configuration if network is offline and no local config exists
DEFAULT_CONFIG = {
    "architecture": "prithvi_eo_v1_100",
    "num_features": 768,
    "pretrained_cfg": {
        "img_size": 224,
        "patch_size": [1, 16, 16],
        "num_frames": 3,
        "in_chans": 6,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        "mask_ratio": 0.75,
        "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
        "mean": [
            775.2290211032589,
            1080.992780391705,
            1228.5855250417867,
            2497.2022620507532,
            2204.2139147975554,
            1610.8324823273745,
        ],
        "std": [
            1281.526139861424,
            1270.0297974547493,
            1399.4802505642526,
            1368.3446143747644,
            1291.6764008585435,
            1154.505683480695,
        ],
    },
}


def get_hf_token(user_token: str | None = None) -> str:
    """Retrieve Hugging Face token with precedence: CLI arg > HF_TOKEN env > default."""
    token = user_token or os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN
    return token.strip()


def resolve_default_checkpoint() -> Path:
    """Locate the default checkpoint file relative to this script or current directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / DEFAULT_CHECKPOINT_NAME,
        Path.cwd() / "model" / DEFAULT_CHECKPOINT_NAME,
        Path.cwd() / DEFAULT_CHECKPOINT_NAME,
        script_dir / "Prithvi_100M.pt",
        Path.cwd() / "model" / "Prithvi_100M.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return script_dir / DEFAULT_CHECKPOINT_NAME


def import_prithvi_model_from_hf(
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    save_local_copy: bool = True,
) -> type[torch.nn.Module]:
    """Download and dynamically import the PrithviMAE model class from Hugging Face."""
    hf_token = get_hf_token(token)
    print(f"Fetching model definition 'prithvi_mae.py' from Hugging Face ({repo_id})...")

    try:
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename="prithvi_mae.py",
            token=hf_token,
        )
    except Exception as e:
        print(f"Warning: Failed to download 'prithvi_mae.py' from HF Hub: {e}")
        # Check if local fallback exists
        local_model_file = Path(__file__).resolve().parent / "prithvi_mae.py"
        if local_model_file.is_file():
            print(f"Using local cached copy: {local_model_file}")
            model_file = str(local_model_file)
        else:
            raise RuntimeError(
                f"Could not download prithvi_mae.py from Hugging Face and no local copy was found: {e}"
            ) from e

    # Dynamically load the module
    spec = importlib.util.spec_from_file_location("prithvi_mae", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module specification from {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["prithvi_mae"] = module
    spec.loader.exec_module(module)

    # Optionally persist a copy next to this script for offline reuse
    if save_local_copy:
        dest = Path(__file__).resolve().parent / "prithvi_mae.py"
        if not dest.is_file() and os.path.abspath(model_file) != str(dest):
            try:
                dest.write_text(Path(model_file).read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Cached local model definition at: {dest}")
            except Exception as e:
                print(f"Note: Could not cache local copy ({e})")

    if not hasattr(module, "PrithviMAE"):
        raise AttributeError("The imported module does not define 'PrithviMAE'.")

    return getattr(module, "PrithviMAE")


def load_model_config(
    config_path: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
) -> dict[str, Any]:
    """Load Prithvi model configuration from local file, Hugging Face Hub, or defaults."""
    if config_path and Path(config_path).is_file():
        print(f"Loading configuration from local path: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
            data = json.loads(raw) if config_path.endswith(".json") else yaml.safe_load(raw)
            return data.get("pretrained_cfg", data)

    # Attempt to fetch config.json from Hugging Face
    hf_token = get_hf_token(token)
    try:
        print(f"Fetching configuration 'config.json' from Hugging Face ({repo_id})...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=hf_token,
        )
        with open(downloaded, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("pretrained_cfg", data)
    except Exception as e:
        print(f"Notice: Could not fetch config from Hugging Face Hub ({e}). Using built-in default config.")
        return DEFAULT_CONFIG["pretrained_cfg"].copy()


def load_prithvi_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load model weights from .pt checkpoint, handling dynamic positional embeddings."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint weights file not found: {checkpoint_path}")

    print(f"Loading model weights from: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    # In case the checkpoint was wrapped in an outer dict
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Prithvi uses dynamic 3D positional embeddings based on input grid size and num_frames.
    # Fixed-size positional embeddings in state_dict are discarded so the architecture
    # calculates matching positional embeddings dynamically.
    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded successfully. (Missing dynamic keys: {len(missing)}, Unexpected keys: {len(unexpected)})")

    model.to(device)
    model.eval()
    return model


def read_geotiff(file_path: str | Path) -> Tuple[np.ndarray, dict, Any]:
    """Read all bands and metadata from a GeoTIFF raster."""
    with rasterio.open(str(file_path)) as src:
        img = src.read()
        meta = src.meta.copy()
        try:
            coords = src.lnglat()
        except Exception:
            coords = None
    return img, meta, coords


def save_geotiff(image: np.ndarray | torch.Tensor, output_path: str | Path, meta: dict) -> None:
    """Save multi-band raster data to a GeoTIFF file with preserved metadata."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    meta = meta.copy()
    meta.update({
        "driver": "GTiff",
        "count": image.shape[0],
        "height": image.shape[1],
        "width": image.shape[2],
        "dtype": image.dtype.name if hasattr(image.dtype, "name") else str(image.dtype),
    })

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with rasterio.open(str(output_path), "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)


def process_channel_group(
    orig_img: torch.Tensor,
    new_img: torch.Tensor,
    channels: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process original and reconstructed bands for RGB visualization.
    
    Rescales to original reflectance, removes outliers via percentile stretch,
    and scales to [0, 1].
    """
    mean_t = torch.tensor(np.asarray(mean)[:, None, None], dtype=torch.float32)
    std_t = torch.tensor(np.asarray(std)[:, None, None], dtype=torch.float32)

    orig_c = orig_img[list(channels), ...].float()
    new_c = new_img[list(channels), ...].float()

    valid_mask = orig_c != NO_DATA_FLOAT

    # Convert back to unnormalized reflectance values
    orig_unnorm = (orig_c * std_t[list(channels)]) + mean_t[list(channels)]
    new_unnorm = (new_c * std_t[list(channels)]) + mean_t[list(channels)]

    valid_vals = orig_unnorm[valid_mask]
    if valid_vals.numel() > 0:
        max_val = max(3000.0, float(np.percentile(valid_vals.cpu().numpy(), PERCENTILE)))
    else:
        max_val = 3000.0
    min_val = float(OFFSET)

    denom = max(max_val - min_val, 1e-5)
    orig_rgb = torch.clamp((orig_unnorm - min_val) / denom, 0.0, 1.0)
    new_rgb = torch.clamp((new_unnorm - min_val) / denom, 0.0, 1.0)

    orig_rgb[~valid_mask] = 0.0
    new_rgb[~valid_mask] = 0.0

    return orig_rgb, new_rgb


def load_input_dataset(
    file_paths: Sequence[str | Path],
    mean: Sequence[float],
    std: Sequence[float],
    indices: Sequence[int] | None = None,
) -> Tuple[np.ndarray, list[dict]]:
    """Load chronological images, normalize them, and collect GeoTIFF metadata.
    
    Returns:
        input_data: numpy array of shape (1, C, T, H, W)
        metas: list of rasterio metadata dicts for each time step
    """
    frames = []
    metas = []

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    for path in file_paths:
        img, meta, _ = read_geotiff(path)
        # Transpose from (C, H, W) to (H, W, C) for channel indexing/broadcasting
        img = np.moveaxis(img, 0, -1)

        if indices is not None:
            img = img[..., list(indices)]

        # Handle 3-band input gracefully by duplicating/padding to 6 bands if needed
        if img.shape[-1] == 3 and len(mean) == 6:
            print(f"Notice: {path} has 3 bands. Duplicating to 6 bands for Prithvi compatibility.")
            img = np.concatenate([img, img], axis=-1)

        if img.shape[-1] != len(mean):
            raise ValueError(
                f"File {path} has {img.shape[-1]} bands, but configuration requires {len(mean)} bands "
                f"(expected B02, B03, B04, B05, B06, B07)."
            )

        # Normalize with nodata preservation
        normalized = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean_arr) / std_arr)
        frames.append(normalized)
        metas.append(meta)

    # Stack along time axis: (T, H, W, C)
    stacked = np.stack(frames, axis=0)
    # Reorder to (C, T, H, W)
    stacked = np.moveaxis(stacked, -1, 0).astype(np.float32)
    # Add batch dimension: (1, C, T, H, W)
    batched = np.expand_dims(stacked, axis=0)

    return batched, metas


def run_sliding_window_inference(
    model: torch.nn.Module,
    input_data: np.ndarray,
    img_size: int = 224,
    batch_size: int = 4,
    mask_ratio: float = 0.75,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute tiled MAE inference over arbitrary image dimensions using sliding windows.
    
    Returns:
        rec_imgs_full: Reconstructed tensor of shape (1, C, T, H, W)
        mask_imgs_full: Mask tensor of shape (1, C, T, H, W)
        batch_full: Original input tensor of shape (1, C, T, H, W)
    """
    _, c, t, orig_h, orig_w = input_data.shape

    # Reflective padding so spatial dimensions are multiples of img_size
    pad_h = (img_size - (orig_h % img_size)) % img_size
    pad_w = (img_size - (orig_w % img_size)) % img_size

    padded_data = np.pad(
        input_data,
        ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )

    batch_tensor = torch.from_numpy(padded_data)

    # Unfold into spatial tiles of size (img_size, img_size)
    windows = batch_tensor.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h_tiles, w_tiles = windows.shape[3:5]

    # Rearrange to: (num_tiles, C, T, img_size, img_size)
    windows = rearrange(
        windows,
        "b c t h1 w1 h w -> (b h1 w1) c t h w",
        h=img_size,
        w=img_size,
    )

    num_tiles = windows.shape[0]
    num_batches = max(1, (num_tiles + batch_size - 1) // batch_size)
    tile_batches = torch.tensor_split(windows, num_batches, dim=0)

    print(f"Running inference on {num_tiles} tiles of size {img_size}x{img_size} across {num_batches} forward passes...")

    rec_list = []
    mask_list = []

    with torch.no_grad():
        for i, tile_batch in enumerate(tile_batches):
            x = tile_batch.to(device)
            # Forward pass through PrithviMAE
            _, pred, mask = model(x, mask_ratio=mask_ratio)

            # Unpatchify mask and predictions
            mask_unpatch = (
                model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
                .detach()
                .cpu()
            )
            pred_unpatch = model.unpatchify(pred).detach().cpu()

            # Merge visible and reconstructed patches
            tile_cpu = tile_batch.cpu().clone()
            tile_cpu[mask_unpatch == 1] = pred_unpatch[mask_unpatch == 1]

            # Invert mask representation so masked patches visually stand out
            mask_vis = (~(mask_unpatch.to(torch.bool))).to(torch.float32)

            rec_list.append(tile_cpu)
            mask_list.append(mask_vis)

    rec_concat = torch.cat(rec_list, dim=0)
    mask_concat = torch.cat(mask_list, dim=0)

    # Reconstruct full image grid from patches
    rec_reconstructed = rearrange(
        rec_concat,
        "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
        h=img_size,
        w=img_size,
        b=1,
        c=c,
        t=t,
        h1=h_tiles,
        w1=w_tiles,
    )
    mask_reconstructed = rearrange(
        mask_concat,
        "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
        h=img_size,
        w=img_size,
        b=1,
        c=c,
        t=t,
        h1=h_tiles,
        w1=w_tiles,
    )

    # Crop out reflective padding back to original spatial dimensions
    rec_full = rec_reconstructed[..., :orig_h, :orig_w]
    mask_full = mask_reconstructed[..., :orig_h, :orig_w]
    batch_full = batch_tensor[..., :orig_h, :orig_w]

    return rec_full, mask_full, batch_full


def save_inference_outputs(
    rec_img: torch.Tensor,
    mask_img: torch.Tensor,
    orig_img: torch.Tensor,
    metas: list[dict],
    output_dir: Path,
    bands: list[str],
    mean: list[float],
    std: list[float],
    rgb_outputs: bool = True,
) -> None:
    """Export reconstructed GeoTIFF rasters and RGB preview visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_timestamps = rec_img.shape[1]

    mean_t = torch.tensor(np.asarray(mean)[:, None, None], dtype=torch.float32)
    std_t = torch.tensor(np.asarray(std)[:, None, None], dtype=torch.float32)

    # Identify RGB channels for visualization (B04=Red, B03=Green, B02=Blue)
    rgb_channel_names = ["B04", "B03", "B02"]
    if all(name in bands for name in rgb_channel_names):
        rgb_indices = [bands.index(b) for b in rgb_channel_names]
    else:
        rgb_indices = [0, 1, 2] if len(bands) >= 3 else [0]

    for t in range(num_timestamps):
        meta = metas[t].copy() if t < len(metas) else metas[0].copy()

        # Unnormalize back to standard surface reflectance values (int16)
        rec_unnorm = ((rec_img[:, t, :, :] * std_t) + mean_t).clamp(-32768, 32767).to(torch.int16)
        mask_t = mask_img[:, t, :, :].to(torch.int16)

        pred_tif_path = output_dir / f"reconstructed_t{t}.tif"
        mask_tif_path = output_dir / f"mask_t{t}.tif"

        save_geotiff(rec_unnorm.numpy(), pred_tif_path, meta)
        save_geotiff(mask_t.numpy(), mask_tif_path, meta)
        print(f"Saved GeoTIFF output: {pred_tif_path}")

        # RGB Visualizations
        if rgb_outputs and len(rgb_indices) == 3:
            rgb_orig, rgb_pred = process_channel_group(
                orig_img=orig_img[:, t, :, :],
                new_img=rec_img[:, t, :, :],
                channels=rgb_indices,
                mean=mean,
                std=std,
            )
            # Masked input for visualization
            rgb_masked = mask_img[rgb_indices, t, :, :] * rgb_orig

            rgb_meta = meta.copy()
            rgb_meta.update(count=3, dtype="uint8", compress="lzw", nodata=0)

            def to_uint8(tensor: torch.Tensor) -> np.ndarray:
                return (tensor.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

            save_geotiff(to_uint8(rgb_orig), output_dir / f"rgb_original_t{t}.tif", rgb_meta)
            save_geotiff(to_uint8(rgb_masked), output_dir / f"rgb_masked_t{t}.tif", rgb_meta)
            save_geotiff(to_uint8(rgb_pred), output_dir / f"rgb_reconstructed_t{t}.tif", rgb_meta)
            print(f"Saved RGB previews for timestamp t={t}: (original, masked, reconstructed)")


def download_official_examples(
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
) -> list[str]:
    """Download official example HLS GeoTIFFs from Hugging Face for quick verification."""
    hf_token = get_hf_token(token)
    example_filenames = [
        "examples/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
        "examples/HLS.L30.T13REN.2018029T172738.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
        "examples/HLS.L30.T13REN.2018061T172724.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
    ]
    downloaded_paths = []
    print("Downloading official HLS GeoTIFF examples from Hugging Face...")
    for fn in example_filenames:
        path = hf_hub_download(repo_id=repo_id, filename=fn, token=hf_token)
        downloaded_paths.append(path)
        print(f"  Downloaded: {Path(path).name}")
    return downloaded_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure CLI options."""
    parser = argparse.ArgumentParser(
        description="Run Prithvi-EO-1.0-100M MAE inference using Hugging Face model definition and local weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-files",
        "-i",
        nargs="+",
        type=str,
        default=None,
        help="Path(s) to chronological multi-band GeoTIFFs.",
    )
    parser.add_argument(
        "--checkpoint",
        "-w",
        type=str,
        default=str(resolve_default_checkpoint()),
        help="Path to Prithvi_EO_V1_100M.pt or Prithvi_100M.pt weights checkpoint.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config.json. If omitted, automatically fetched from Hugging Face.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=DEFAULT_HF_TOKEN,
        help="Hugging Face User Access Token.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID containing the Prithvi model definition.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("prithvi_inference_output"),
        help="Directory to save reconstructed GeoTIFFs and visualizations.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
        help="Percentage of patches to mask during inference (0.75 = 75%%).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of 224x224 spatial tiles processed simultaneously.",
    )
    parser.add_argument(
        "--input-indices",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based indices to select and reorder bands to B02, B03, B04, B05, B06, B07.",
    )
    parser.add_argument(
        "--rgb-outputs",
        action="store_true",
        default=True,
        help="Generate high-contrast RGB visual GeoTIFFs for easy inspection.",
    )
    parser.add_argument(
        "--download-examples",
        action="store_true",
        help="Download and run inference on official HLS GeoTIFF examples from Hugging Face.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for computation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine compute device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)

    print(f"Selected device: {device}")

    # Determine input files
    data_files = args.data_files
    if args.download_examples or not data_files:
        if not data_files:
            print("No --data-files provided. Fetching official test examples from Hugging Face...")
        data_files = download_official_examples(repo_id=args.repo_id, token=args.hf_token)

    print(f"\nProcessing {len(data_files)} chronological image(s):")
    for f in data_files:
        print(f"  - {f}")

    # 1. Import model definition from Hugging Face
    PrithviMAE = import_prithvi_model_from_hf(
        repo_id=args.repo_id,
        token=args.hf_token,
        save_local_copy=True,
    )

    # 2. Load model configuration
    config = load_model_config(
        config_path=args.config,
        repo_id=args.repo_id,
        token=args.hf_token,
    )

    bands = config.get("bands", ["B02", "B03", "B04", "B05", "B06", "B07"])
    mean = config.get("mean", DEFAULT_CONFIG["pretrained_cfg"]["mean"])
    std = config.get("std", DEFAULT_CONFIG["pretrained_cfg"]["std"])
    img_size = config.get("img_size", 224)

    # 3. Instantiate model with current time-sequence length and channels
    model_kwargs = {
        "img_size": img_size,
        "patch_size": config.get("patch_size", [1, 16, 16]),
        "num_frames": len(data_files),
        "in_chans": len(bands),
        "embed_dim": config.get("embed_dim", 768),
        "depth": config.get("depth", 12),
        "num_heads": config.get("num_heads", 12),
        "decoder_embed_dim": config.get("decoder_embed_dim", 512),
        "decoder_depth": config.get("decoder_depth", 8),
        "decoder_num_heads": config.get("decoder_num_heads", 16),
        "mlp_ratio": config.get("mlp_ratio", 4),
    }

    print(f"\nInstantiating PrithviMAE architecture ({len(data_files)} frame(s), {len(bands)} bands)...")
    model = PrithviMAE(**model_kwargs)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # 4. Load weights from .pt checkpoint
    model = load_prithvi_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # 5. Load and preprocess input data
    print("\nReading and normalizing input data...")
    input_tensor_np, metas = load_input_dataset(
        file_paths=data_files,
        mean=mean,
        std=std,
        indices=args.input_indices,
    )
    print(f"Input tensor shape (B, C, T, H, W): {input_tensor_np.shape}")

    # 6. Run sliding window inference
    rec_full, mask_full, batch_full = run_sliding_window_inference(
        model=model,
        input_data=input_tensor_np,
        img_size=img_size,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        device=device,
    )
    print(f"Inference complete! Reconstructed shape: {tuple(rec_full.shape)}")

    # 7. Save reconstructed outputs
    print(f"\nSaving results to directory: {args.output_dir.resolve()}")
    save_inference_outputs(
        rec_img=rec_full[0],
        mask_img=mask_full[0],
        orig_img=batch_full[0],
        metas=metas,
        output_dir=args.output_dir,
        bands=bands,
        mean=mean,
        std=std,
        rgb_outputs=args.rgb_outputs,
    )

    print("\n[SUCCESS] Prithvi inference executed successfully!")


if __name__ == "__main__":
    main()