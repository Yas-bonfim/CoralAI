#!/usr/bin/env python3
"""Le as imagens das pastas locais, unifica as classes e gera o dataset no formato YOLO.

Os datasets sao de CLASSIFICACAO (sem bounding boxes). Aqui as boxes sao geradas
automaticamente (pseudo-labels) com YOLO-World ("coral"); se nada for detectado,
a box vira a imagem inteira com uma pequena margem.

Uso:
    python prepare_dataset.py
    python prepare_dataset.py --extra-root "Corals Classification"
    python prepare_dataset.py --full-image-boxes      # sem YOLO-World, mais rapido
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

from config import DATASET_DIRS, DATA_YAML, IMG_EXT, SEED, YOLO_DIR, detect_split, unify_label


def pseudo_box(world_model, img_path, w, h):
    """Retorna (x1, y1, x2, y2) em pixels."""
    if world_model is not None:
        r = world_model.predict(str(img_path), conf=0.10, verbose=False)[0]
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            return (float(xyxy[:, 0].min()), float(xyxy[:, 1].min()),
                    float(xyxy[:, 2].max()), float(xyxy[:, 3].max()))
    mx, my = 0.02 * w, 0.02 * h
    return (mx, my, w - mx, h - my)


def collect(roots):
    items, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[!] Pasta nao encontrada, ignorando: {root}")
            continue
        found = 0
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in seen:  # duplicata entre/dentro dos datasets
                continue
            seen.add(digest)
            rel = p.relative_to(root).parts[:-1]
            items.append({
                "path": p,
                "label": unify_label(p.parent.name),
                "split": detect_split(rel),
                "tag": root.name.replace(" ", "")[:12],
                "hash": digest[:8],
            })
            found += 1
        print(f"[+] {root}: {found} imagens")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-root", action="append", default=[],
                    help="pasta adicional de dataset (pode repetir)")
    ap.add_argument("--full-image-boxes", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recria o dataset mesmo que ele ja exista")
    args = ap.parse_args()

    # O data.yaml e escrito por ultimo, entao sua existencia indica dataset completo
    if DATA_YAML.exists() and not args.force:
        print(f"[=] Dataset ja existe em {YOLO_DIR}/ (use --force para recriar). Nada a fazer.")
        return

    random.seed(SEED)
    items = collect(list(DATASET_DIRS) + args.extra_root)
    if not items:
        raise SystemExit("Nenhuma imagem encontrada. Confira DATASET_DIRS em config.py.")

    # Imagens sem pasta de split (ex.: segundo dataset) sao divididas 80/10/10
    loose = [it for it in items if it["split"] is None]
    random.shuffle(loose)
    for i, it in enumerate(loose):
        frac = i / max(len(loose), 1)
        it["split"] = "train" if frac < 0.8 else "val" if frac < 0.9 else "test"

    labels = sorted({it["label"] for it in items})
    cls_id = {l: i for i, l in enumerate(labels)}
    print("[+] Classes unificadas:", cls_id)
    for split in ("train", "val", "test"):
        counts = {l: sum(1 for it in items if it["split"] == split and it["label"] == l) for l in labels}
        print(f"    {split}: {counts}")

    world = None
    if not args.full_image_boxes:
        from ultralytics import YOLOWorld
        world = YOLOWorld("yolov8s-worldv2.pt")
        world.set_classes(["coral"])

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for s in ("train", "val", "test"):
        (YOLO_DIR / "images" / s).mkdir(parents=True)
        (YOLO_DIR / "labels" / s).mkdir(parents=True)

    for n, it in enumerate(items, 1):
        name = f"{it['tag']}_{it['hash']}_{it['path'].stem}"[:90]
        try:
            with Image.open(it["path"]) as im:
                im = im.convert("RGB")
                w, h = im.size
                dst = YOLO_DIR / "images" / it["split"] / f"{name}.jpg"
                im.save(dst, quality=95)
        except Exception as e:
            print("   pulando", it["path"], e)
            continue
        x1, y1, x2, y2 = pseudo_box(world, dst, w, h)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        (YOLO_DIR / "labels" / it["split"] / f"{name}.txt").write_text(
            f"{cls_id[it['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        if n % 200 == 0:
            print(f"    {n}/{len(items)}")

    yaml = [f"path: {YOLO_DIR.resolve()}", "train: images/train", "val: images/val",
            "test: images/test", f"nc: {len(labels)}", "names:"]
    yaml += [f"  {i}: {l}" for l, i in cls_id.items()]
    DATA_YAML.write_text("\n".join(yaml) + "\n")
    print("[+] Pronto:", DATA_YAML)


if __name__ == "__main__":
    main()
