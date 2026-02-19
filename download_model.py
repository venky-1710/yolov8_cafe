"""
download_model.py
─────────────────
Downloads and validates the YOLOv8 model weights.
Run this once before starting the app:

    python download_model.py [--model yolov8m]

Available models (accuracy vs speed trade-off):
  yolov8n  – nano   (fastest,  ~3.2M params)
  yolov8s  – small  (~11M params)
  yolov8m  – medium (~25M params)  ← recommended
  yolov8l  – large  (~43M params)
  yolov8x  – xlarge (most accurate, ~68M params)
"""

import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download & validate YOLOv8 model")
    parser.add_argument("--model", default="yolov8m",
                        choices=["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"],
                        help="Model variant to download (default: yolov8m)")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save the model (default: models/)")
    args = parser.parse_args()

    model_name = args.model + ".pt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / model_name

    print("=" * 55)
    print(f"  YOLOv8 Model Downloader")
    print("=" * 55)

    # ── Import check ──────────────────────────────────────
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    # ── GPU info ──────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU detected: {gpu} ({vram:.1f} GB VRAM)")
        device = "cuda"
    else:
        print("\n⚠️  No GPU detected — using CPU (inference will be slower)")
        device = "cpu"

    # ── Download / load model ─────────────────────────────
    print(f"\n📥 Loading model: {model_name}")
    print(f"   Save path: {model_path.resolve()}")
    t0 = time.time()

    try:
        # YOLO auto-downloads if not found locally
        model = YOLO(model_name)

        # Move the downloaded .pt file to our models/ dir
        import shutil, os
        default_loc = Path(model_name)
        if default_loc.exists() and not model_path.exists():
            shutil.move(str(default_loc), str(model_path))
            print(f"   Moved to {model_path}")
        elif model_path.exists():
            print(f"   Already exists at {model_path}")

        elapsed = time.time() - t0
        print(f"   ✅ Model loaded in {elapsed:.1f}s")

    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # ── Validate model with a dummy inference ─────────────
    print(f"\n🔍 Running validation inference on {device.upper()}...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    t1 = time.time()
    results = model(dummy, device=device, verbose=False)
    inf_ms = (time.time() - t1) * 1000
    print(f"   ✅ Inference OK — {inf_ms:.1f} ms per frame")

    # ── Model info ────────────────────────────────────────
    print(f"\n📊 Model Info:")
    print(f"   Name      : {args.model}")
    info = model.info(verbose=False)
    print(f"   Parameters: {info[0]:,}")
    print(f"   GFLOPs    : {info[1]:.1f}")

    # ── Update detector.py MODEL_PATH hint ───────────────
    print(f"\n💡 To use this model, set in detector.py or via env var:")
    print(f"   MODEL_PATH = \"{model_path}\"")
    print(f"   or:  set YOLO_MODEL={model_path}")

    print("\n✅ Done! You can now run:  python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
