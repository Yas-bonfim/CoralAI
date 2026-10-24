#!/usr/bin/env python3
"""Inferencia com o modelo treinado. Imprime um JSON por deteccao:

{"timestamp": 1729432800.125, "label": "bleached_coral", "confidence": 0.94, "bbox": [120, 85, 220, 185]}

Uso:
    python predict.py --source foto.jpg [--draw]
    python predict.py --source video.mp4 [--draw]
    python predict.py --source 0 [--draw]          # webcam, "q" para sair
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from config import BEST_WEIGHTS

OUT_DIR = Path("saida_anotada")


def detections_to_json(result, timestamp):
    out = []
    for b in result.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
        out.append({
            "timestamp": round(timestamp, 3),
            "label": result.names[int(b.cls[0])],
            "confidence": round(float(b.conf[0]), 2),
            "bbox": [x1, y1, x2, y2],
        })
    return out


def predict_image(model, args):
    import cv2
    res = model.predict(args.source, conf=args.conf, verbose=False)[0]
    for d in detections_to_json(res, time.time()):
        print(json.dumps(d))
    if args.draw:
        OUT_DIR.mkdir(exist_ok=True)
        dst = OUT_DIR / f"{Path(args.source).stem}_anotada.jpg"
        cv2.imwrite(str(dst), res.plot())
        print(f"[+] Imagem anotada: {dst}")


def predict_video(model, args):
    import cv2
    src = args.source
    live = src.isdigit()
    cap = cv2.VideoCapture(int(src) if live else src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0, idx, writer = time.time(), 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = time.time() if live else t0 + idx / fps
        res = model.predict(frame, conf=args.conf, verbose=False)[0]
        for d in detections_to_json(res, ts):
            print(json.dumps(d), flush=True)
        if args.draw:
            annotated = res.plot()
            if live:
                cv2.imshow("coral", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if writer is None:
                    OUT_DIR.mkdir(exist_ok=True)
                    hh, ww = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(OUT_DIR / f"{Path(src).stem}_anotado.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                writer.write(annotated)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[+] Video anotado salvo em {OUT_DIR}/")
    if live and args.draw:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default=str(BEST_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true",
                    help="desenha as boxes (arquivos em saida_anotada/; webcam abre janela)")
    args = ap.parse_args()

    model = YOLO(args.model)
    is_video = args.source.isdigit() or Path(args.source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    (predict_video if is_video else predict_image)(model, args)


if __name__ == "__main__":
    main()
