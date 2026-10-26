"""Konfigurasi bersama. Path bisa diubah lewat variabel lingkungan EMO_DATA / EMO_OUT
(mis. di Kaggle: EMO_DATA=/kaggle/input/emotweetid, EMO_OUT=/kaggle/working)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EMO_DATA", ROOT / "data"))
OUT_DIR = Path(os.environ.get("EMO_OUT", ROOT))
RES = OUT_DIR / "results"
MODELS = OUT_DIR / "models"
RES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
E2I = {e: i for i, e in enumerate(EMOTIONS)}
K = len(EMOTIONS)
SEED = 42
BERT_MODEL = "indolem/indobertweet-base-uncased"  # Koto et al. (2021)
