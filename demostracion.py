"""Carga los datos incluidos en la demostración y evalúa la partición reservada."""

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from dataset import Dataset
from preprocess import CLASSES


ROOT = Path(__file__).resolve().parent


def dataset_archive():
    configured = os.getenv("HAMBEE_DATASET")
    path = Path(configured) if configured else ROOT / "data" / "Dataset.zip"
    return path if path.is_file() else None


def evaluate_test(session, archive, expected_sha256):
    start = perf_counter()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    errors, results = [], []
    with Dataset(archive, expected_sha256) as dataset:
        rows = dataset.partition("test")
        for row in rows:
            path = row["relative_path"]
            batch = dataset.tensor(row)[None]
            predicted = CLASSES[int(session.run(None, {"image": batch})[0].argmax())]
            actual = row["class_label"]
            results.append({"ruta": path, "particion": "test",
                            "clase_real": actual, "clase_predicha": predicted})
            confusion[CLASSES.index(actual), CLASSES.index(predicted)] += 1
            if actual != predicted:
                errors.append({"image": Path(path).name, "actual": actual, "predicted": predicted})
    correct = int(np.trace(confusion))
    f1 = 2 * np.diag(confusion) / (confusion.sum(axis=0) + confusion.sum(axis=1))
    return {"images": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "macro_f1": float(f1.mean()), "confusion_np_p": confusion.tolist(),
            "errors": errors, "elapsed_seconds": perf_counter() - start, "results": results}
