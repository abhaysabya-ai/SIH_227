"""Swin Transformer Tiny (microsoft/swin-tiny-patch4-window7-224) Inference Script.

This script:
1. Loads the model weights from the local .bin file ('model/pytorch_model.bin').
2. Imports the model architecture definition and feature processor from Hugging Face
   ('microsoft/swin-tiny-patch4-window7-224') using the Hugging Face API / token.
3. Supports two inference modes:
   - 'classification': Predicts Top-K ImageNet class labels and probabilities.
   - 'features': Extracts dense spatial feature maps (last_hidden_state, pooled embeddings)
     for integration into the Stage B Super Resolution pipeline (Hybrid Blocks / Conditioning).
4. Supports both standard images (JPG, PNG) and geospatial GeoTIFFs (TIF, TIFF).

Example usage:
--------------
# Run classification on an image:
python model/inference_swin.py --input dummy_input.jpg

# Extract feature embeddings for downstream super-resolution:
python model/inference_swin.py --input dummy_input.jpg --mode features --output-dir swin_features

# Specify custom weights and top-k predictions:
python model/inference_swin.py \
    --weights model/pytorch_model.bin \
    --input path/to/image.png \
    --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

# Optional rasterio import for geospatial rasters
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Default configuration
DEFAULT_REPO_ID = "microsoft/swin-tiny-patch4-window7-224"
DEFAULT_HF_TOKEN = "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"
DEFAULT_WEIGHTS_NAME = "pytorch_model.bin"
DEFAULT_CONFIG_CACHE_NAME = "swin_config.json"


def get_hf_token(user_token: Optional[str] = None) -> str:
    """Retrieve Hugging Face token with precedence: CLI arg > HF_TOKEN env > default."""
    token = user_token or os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN
    return token.strip()


def resolve_default_weights() -> Path:
    """Locate the default pytorch_model.bin checkpoint."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / DEFAULT_WEIGHTS_NAME,
        Path.cwd() / "model" / DEFAULT_WEIGHTS_NAME,
        Path.cwd() / DEFAULT_WEIGHTS_NAME,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return script_dir / DEFAULT_WEIGHTS_NAME


def fetch_swin_config(
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Fetch and cache config.json for Swin Transformer from Hugging Face."""
    hf_token = get_hf_token(token)
    target_cache = (cache_dir or Path(__file__).resolve().parent) / DEFAULT_CONFIG_CACHE_NAME

    if target_cache.is_file():
        return target_cache

    print(f"Fetching '{repo_id}' config from Hugging Face...")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=hf_token,
        )
        shutil.copy(downloaded, target_cache)
        print(f"Cached local Swin config to: {target_cache}")
        return target_cache
    except Exception as e:
        if target_cache.is_file():
            print(f"Warning: Failed to download config ({e}), using cached copy.")
            return target_cache
        raise RuntimeError(f"Could not retrieve config for {repo_id}: {e}") from e


def load_image_as_rgb(image_path: Union[str, Path]) -> Image.Image:
    """Load standard image or GeoTIFF file as a PIL RGB Image."""
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    # Handle GeoTIFF files
    if path.suffix.lower() in [".tif", ".tiff"]:
        if HAS_RASTERIO:
            with rasterio.open(str(path)) as src:
                arr = src.read()  # (C, H, W)
                if arr.shape[0] >= 3:
                    # Use first 3 channels as RGB (or Red, Green, Blue)
                    rgb_arr = arr[:3]
                else:
                    rgb_arr = np.repeat(arr[:1], 3, axis=0)

                # Normalize to 0-255 uint8
                rgb_arr = rgb_arr.astype(np.float32)
                min_v, max_v = rgb_arr.min(), rgb_arr.max()
                if max_v > min_v:
                    rgb_arr = (rgb_arr - min_v) / (max_v - min_v) * 255.0
                rgb_uint8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
                return Image.fromarray(np.moveaxis(rgb_uint8, 0, -1))

    # Standard image formats (JPG, PNG, WebP, etc.)
    return Image.open(str(path)).convert("RGB")


def load_swin_model(
    weights_path: Union[str, Path],
    mode: str = "classification",
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """Load Swin Transformer with architecture from Hugging Face and weights from local .bin file.
    
    Args:
        weights_path: Path to the pytorch_model.bin checkpoint.
        mode: 'classification' or 'features'.
        repo_id: Hugging Face model repository.
        token: Hugging Face user access token.
        device: Target torch device.
    
    Returns:
        Tuple of (model, image_processor).
    """
    weights_path = Path(weights_path).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Swin model weights file not found: {weights_path}")

    hf_token = get_hf_token(token)
    config_file = fetch_swin_config(repo_id=repo_id, token=hf_token)

    print(f"Loading image processor from Hugging Face ({repo_id})...")
    processor = AutoImageProcessor.from_pretrained(repo_id, token=hf_token)

    print(f"Loading Swin Transformer weights ({weights_path.name})...")
    # Hugging Face transformers requires config.json and pytorch_model.bin to be co-located
    # in a directory to execute automated architectural weight mapping cleanly.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = os.path.join(tmpdir, "config.json")
        tmp_weights = os.path.join(tmpdir, "pytorch_model.bin")
        shutil.copy(str(config_file), tmp_cfg)

        # Create link or copy
        try:
            os.link(str(weights_path), tmp_weights)
        except Exception:
            shutil.copy(str(weights_path), tmp_weights)

        if mode == "features":
            model = AutoModel.from_pretrained(tmpdir)
        else:
            model = AutoModelForImageClassification.from_pretrained(tmpdir)

    model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model successfully loaded ({param_count:,} parameters) in '{mode}' mode.")
    return model, processor


def run_classification(
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    image: Image.Image,
    top_k: int = 5,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Any]]:
    """Execute classification inference and return Top-K predicted labels with probabilities."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[0]))

    results = []
    id2label = model.config.id2label
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        label = id2label.get(idx, f"Class {idx}")
        results.append({
            "class_id": idx,
            "label": label,
            "probability": prob,
            "percentage": f"{prob * 100:.2f}%",
        })
    return results


def extract_features(
    model: AutoModel,
    processor: AutoImageProcessor,
    image: Image.Image,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Extract deep spatial representations and pooled feature embeddings from Swin."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {
        "last_hidden_state": outputs.last_hidden_state.cpu(),  # [1, 49, 768] (7x7 spatial tokens)
        "pooler_output": outputs.pooler_output.cpu() if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else None,
    }
    if outputs.hidden_states is not None:
        results["stage_hidden_states"] = [h.cpu() for h in outputs.hidden_states]

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference using Microsoft Swin Transformer Tiny (microsoft/swin-tiny-patch4-window7-224).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="dummy_input.jpg",
        help="Path to input image file (JPG, PNG, GeoTIFF).",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=str(resolve_default_weights()),
        help="Path to pytorch_model.bin checkpoint.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["classification", "features"],
        default="classification",
        help="Inference mode: 'classification' for ImageNet labels or 'features' for feature extraction.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top class predictions to display (classification mode).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("swin_output"),
        help="Directory to save feature maps or predictions.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=DEFAULT_HF_TOKEN,
        help="Hugging Face user access token for configuration / model download.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)

    print(f"Selected computation device: {device}")

    # Check input file
    input_path = Path(args.input)
    if not input_path.is_file():
        # Fallback check in project root
        root_candidate = Path.cwd() / args.input
        if root_candidate.is_file():
            input_path = root_candidate
        else:
            raise FileNotFoundError(f"Input image not found: {args.input}")

    print(f"Loading input: {input_path}")
    image = load_image_as_rgb(input_path)
    print(f"Image loaded: size={image.size}, mode={image.mode}")

    # Load model and processor
    model, processor = load_swin_model(
        weights_path=args.weights,
        mode=args.mode,
        repo_id=args.repo_id,
        token=args.hf_token,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "classification":
        print(f"\n--- Top-{args.top_k} Predictions ---")
        predictions = run_classification(
            model=model,
            processor=processor,
            image=image,
            top_k=args.top_k,
            device=device,
        )
        for rank, res in enumerate(predictions, start=1):
            print(f"  {rank}. [{res['percentage']}] {res['label']} (ID: {res['class_id']})")

        out_json = args.output_dir / f"{input_path.stem}_predictions.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        print(f"\nSaved predictions to: {out_json}")

    elif args.mode == "features":
        print("\nExtracting feature maps...")
        features = extract_features(
            model=model,
            processor=processor,
            image=image,
            device=device,
        )
        last_hidden = features["last_hidden_state"]
        print(f"  - Last Hidden State shape: {tuple(last_hidden.shape)} [Batch, Tokens, Dim]")
        if features["pooler_output"] is not None:
            print(f"  - Pooler Output shape:     {tuple(features['pooler_output'].shape)}")

        out_pt = args.output_dir / f"{input_path.stem}_features.pt"
        torch.save(features, out_pt)
        print(f"\nSaved feature tensors to: {out_pt}")

    print("\n[SUCCESS] Swin inference completed successfully!")


if __name__ == "__main__":
    main()
