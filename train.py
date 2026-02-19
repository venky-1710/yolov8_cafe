"""
train.py
────────
Fine-tune YOLOv8 on a custom cafe/crowd dataset for higher accuracy.

Usage:
    python train.py [--model yolov8m] [--epochs 50] [--data data/cafe.yaml]

If you don't have a custom dataset, the script can download the
publicly available CrowdHuman dataset subset or use COCO person class.
"""

import argparse
import sys
import os
from pathlib import Path


# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for cafe occupancy")
parser.add_argument("--model",   default="yolov8m.pt",  help="Base model weights")
parser.add_argument("--data",    default="data/cafe.yaml", help="Dataset YAML path")
parser.add_argument("--epochs",  type=int, default=50,  help="Training epochs")
parser.add_argument("--imgsz",   type=int, default=640, help="Input image size")
parser.add_argument("--batch",   type=int, default=16,  help="Batch size (-1 = auto)")
parser.add_argument("--workers", type=int, default=4,   help="Dataloader workers")
parser.add_argument("--device",  default="",            help="cuda / cpu / 0,1 (auto if empty)")
parser.add_argument("--project", default="runs/train",  help="Output directory")
parser.add_argument("--name",    default="cafe_v1",     help="Run name")
parser.add_argument("--resume",  action="store_true",   help="Resume last training")
args = parser.parse_args()


def check_deps():
    try:
        from ultralytics import YOLO
        import torch
        return YOLO, torch
    except ImportError as e:
        print(f"[ERROR] {e}\nRun: pip install -r requirements.txt")
        sys.exit(1)


def get_device(torch, requested: str) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dataset(data_path: Path):
    """
    If the dataset YAML doesn't exist, offer to use COCO person-only subset.
    The user can also point --data at their own dataset.
    """
    if data_path.exists():
        return

    print(f"\n⚠️  Dataset config not found: {data_path}")
    print("Options:")
    print("  1. Use built-in COCO person class (downloads ~20 GB)")
    print("  2. Provide your own dataset (see data/cafe.yaml template)")
    choice = input("\nUse COCO person subset? [y/N]: ").strip().lower()

    if choice == "y":
        # Use Ultralytics built-in COCO dataset (person class only)
        # We'll create a filtered YAML
        data_path.parent.mkdir(parents=True, exist_ok=True)
        coco_yaml = data_path.parent / "coco_person.yaml"
        coco_yaml.write_text(
            "# COCO person-only subset\n"
            "path: ./datasets/coco\n"
            "train: images/train2017\n"
            "val:   images/val2017\n"
            "nc: 1\n"
            "names: ['person']\n"
            "download: https://ultralytics.com/assets/coco2017labels.zip\n"
        )
        print(f"Created: {coco_yaml}")
        return str(coco_yaml)
    else:
        print(f"\nPlease create your dataset and update: {data_path}")
        print("See data/cafe.yaml for the template format.")
        sys.exit(0)

    return str(data_path)


def main():
    YOLO, torch = check_deps()
    device = get_device(torch, args.device)

    print("=" * 60)
    print("  YOLOv8 Fine-Tuning — Cafe Occupancy")
    print("=" * 60)
    print(f"  Base model : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {device.upper()}")
    print(f"  Output     : {args.project}/{args.name}")
    print("=" * 60)

    # ── Dataset check ─────────────────────────────────────
    data_path = Path(args.data)
    data_yaml  = ensure_dataset(data_path) or str(data_path)

    # ── Load model ────────────────────────────────────────
    print(f"\n📥 Loading base model: {args.model}")
    model = YOLO(args.model)

    # ── Training ──────────────────────────────────────────
    print(f"\n🚀 Starting training...\n")
    results = model.train(
        data       = data_yaml,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        workers    = args.workers,
        device     = device,
        project    = args.project,
        name       = args.name,
        resume     = args.resume,

        # ── Augmentation (helps with crowd scenes) ────────
        augment    = True,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 5.0,
        translate  = 0.1,
        scale      = 0.5,
        shear      = 2.0,
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,

        # ── Optimizer ─────────────────────────────────────
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        # ── Logging ───────────────────────────────────────
        plots      = True,
        save       = True,
        save_period = 10,
        patience   = 20,   # early stopping
        verbose    = True,
    )

    # ── Results ───────────────────────────────────────────
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Best model : {best_model.resolve()}")
    print(f"   mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95   : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("=" * 60)

    # ── Validation on best weights ────────────────────────
    print("\n🔍 Validating best model...")
    best = YOLO(str(best_model))
    val_results = best.val(data=data_yaml, device=device, verbose=False)
    print(f"   Precision : {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall    : {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"   mAP50     : {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")

    # ── Export hint ───────────────────────────────────────
    print(f"\n💡 To use your trained model, update detector.py:")
    print(f"   MODEL_PATH = \"{best_model}\"")
    print(f"   or:  set YOLO_MODEL={best_model}")
    print("\nOr export to ONNX for faster CPU inference:")
    print(f"   python -c \"from ultralytics import YOLO; YOLO('{best_model}').export(format='onnx')\"")


if __name__ == "__main__":
    main()
