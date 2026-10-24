"""Configuracao compartilhada entre prepare_dataset.py, train.py e predict.py.

Rode tudo a partir da pasta do projeto (CoralAI), onde fica a pasta do dataset.
"""
from pathlib import Path

# Dataset "Bleached Corals and Healthy Corals Classification" (ja vem com
# Training / Validation / Testing, cada um com bleached_corals e healthy_corals)
DATASET_DIRS = [
    Path("Bleached Corals and Healthy Corals Classification"),
    # Segundo dataset (corals-classification): baixe, extraia e adicione a pasta aqui,
    # ou passe com --extra-root no prepare_dataset.py. Exemplo:
    # Path("Corals Classification"),
]

# Saida do prepare_dataset.py (formato YOLO)
YOLO_DIR = Path("coral_yolo_data")
DATA_YAML = YOLO_DIR / "data.yaml"

# Treino
RUN_PROJECT = "runs_coral"
RUN_NAME = "yolov8_coral"
BEST_WEIGHTS = Path(RUN_PROJECT) / RUN_NAME / "weights" / "best.pt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def unify_label(folder_name: str) -> str:
    """Normaliza o nome da pasta de classe (ex.: bleached_corals -> bleached_coral)."""
    n = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    if "bleach" in n:
        return "bleached_coral"
    if "health" in n:
        return "healthy_coral"
    n = n.rstrip("s") if n.endswith("corals") else n
    return n if n.endswith("coral") else f"{n}_coral"


def detect_split(relative_parts):
    """Descobre train/val/test pelo nome das pastas no caminho. None se nao houver."""
    for part in relative_parts:
        p = part.lower()
        if p.startswith("train"):
            return "train"
        if p.startswith("val"):
            return "val"
        if p.startswith("test"):
            return "test"
    return None
