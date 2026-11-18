"""Carga los datos incluidos en la demostración y evalúa la partición reservada."""

import csv
import hashlib
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from preprocess import CLASSES, prepare


ROOT = Path(__file__).resolve().parent


def dataset_archive():
    configured = os.getenv("HAMBEE_DATASET")
    candidates = (Path(configured),) if configured else (
        ROOT / "data" / "Dataset.zip", ROOT.parent / "Dataset.zip", ROOT.parent.parent / "Dataset.zip",
    )
    return next((path for path in candidates if path.is_file()), None)


def evaluate_test(session):
    start = perf_counter()
    dataset = ROOT / "validacion"
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    errors, results = [], []
    for row in rows:
        path = dataset / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"La huella de {row['relative_path']} no coincide.")
        with Image.open(path) as image:
            batch = prepare(image)[None]
        predicted = CLASSES[int(session.run(None, {"image": batch})[0].argmax())]
        actual = row["class_label"]
        results.append({"ruta": row["relative_path"], "particion": "test",
                        "clase_real": actual, "clase_predicha": predicted})
        confusion[CLASSES.index(actual), CLASSES.index(predicted)] += 1
        if actual != predicted:
            errors.append({"image": path.name, "actual": actual, "predicted": predicted})
    correct = int(np.trace(confusion))
    f1 = 2 * np.diag(confusion) / (confusion.sum(axis=0) + confusion.sum(axis=1))
    return {"images": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "macro_f1": float(f1.mean()), "confusion_np_p": confusion.tolist(),
            "errors": errors, "elapsed_seconds": perf_counter() - start, "results": results}
