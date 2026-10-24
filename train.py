#!/usr/bin/env python3
"""Fine-tuning do YOLOv8 no dataset gerado por prepare_dataset.py.

Uso:
    python train.py
    python train.py --weights yolov8s.pt --epochs 100 --batch 16 --device 0
"""
import argparse

from ultralytics import YOLO

from config import BEST_WEIGHTS, DATA_YAML, RUN_NAME, RUN_PROJECT, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="ex.: 0 para GPU, cpu para CPU")
    ap.add_argument("--force", action="store_true",
                    help="retreina mesmo que ja exista um best.pt")
    args = ap.parse_args()

    if BEST_WEIGHTS.exists() and not args.force:
        print(f"[=] Modelo ja treinado em {BEST_WEIGHTS} (use --force para retreinar).")
        return

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} nao existe. Rode prepare_dataset.py primeiro.")

    model = YOLO(args.weights)
    model.train(data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, patience=20, seed=SEED,
                project=RUN_PROJECT, name=RUN_NAME, exist_ok=True)
    model.val(data=str(DATA_YAML), split="test")
    print("[+] Melhor modelo:", BEST_WEIGHTS)


if __name__ == "__main__":
    main()
