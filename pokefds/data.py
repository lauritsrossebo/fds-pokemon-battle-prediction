"""Data loading helpers that work both on Kaggle and locally."""
import json
import os
from pathlib import Path

SLUG = "fds-pokemon-battles-prediction-2026"


def find_data_dir():
    """Return the folder that contains train.jsonl (Kaggle input or a local folder).

    Locally the competition files can be in data/, ../data/ or a folder named like the
    competition next to the repository. POKEFDS_DATA_DIR overrides the search."""
    candidates = [Path(f"/kaggle/input/competitions/{SLUG}"), Path(f"/kaggle/input/{SLUG}"),
                  Path("data"), Path("../data"), Path(SLUG), Path("..") / SLUG]
    if os.environ.get("POKEFDS_DATA_DIR"):
        candidates.insert(0, Path(os.environ["POKEFDS_DATA_DIR"]))
    for c in candidates:
        if (c / "train.jsonl").exists():
            return c
    root = Path("/kaggle/input")
    hits = list(root.rglob("train.jsonl")) if root.exists() else []
    if hits:
        return hits[0].parent
    raise FileNotFoundError("train.jsonl not found: attach the competition dataset.")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
