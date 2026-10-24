#!/usr/bin/env python3
"""Roda a rotina completa: prepare_dataset -> train -> predict.

Cada etapa e pulada se o resultado ja existir (dataset e best.pt), a menos que
voce use --force. Se qualquer etapa falhar, a rotina para.

Uso:
    python run_pipeline.py                                  # prepara e treina (se faltar)
    python run_pipeline.py --source foto.jpg --draw         # e ainda faz a inferencia
    python run_pipeline.py --extra-root "Corals Classification" --epochs 100 --device 0
    python run_pipeline.py --force                          # refaz tudo do zero
"""
import argparse
import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(f"[x] Etapa falhou (codigo {code}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-root", action="append", default=[])
    ap.add_argument("--full-image-boxes", action="store_true")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--source", help="imagem, video ou 0 (webcam); se omitido, nao roda predict")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--force", action="store_true", help="refaz dataset e treino")
    args = ap.parse_args()

    py = sys.executable

    cmd = [py, "prepare_dataset.py"]
    for root in args.extra_root:
        cmd += ["--extra-root", root]
    if args.full_image_boxes:
        cmd.append("--full-image-boxes")
    if args.force:
        cmd.append("--force")
    run(cmd)

    cmd = [py, "train.py", "--weights", args.weights,
           "--epochs", str(args.epochs), "--batch", str(args.batch)]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.force:
        cmd.append("--force")
    run(cmd)

    if args.source:
        cmd = [py, "predict.py", "--source", args.source, "--conf", str(args.conf)]
        if args.draw:
            cmd.append("--draw")
        run(cmd)

    print("\n[+] Rotina concluida.")


if __name__ == "__main__":
    main()
