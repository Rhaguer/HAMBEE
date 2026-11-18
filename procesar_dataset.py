"""Clasifica el dataset completo en lotes y exporta la auditoría en CSV y JSON."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import onnxruntime as ort

from app import MODELS, session_for
from dataset import Dataset
from preprocess import CLASSES


def batched(iterable, size):
    """Compatibilidad con Python 3.11 / 3.12 para la ejecución en Colab."""
    if size < 1:
        raise ValueError("El tamaño del lote debe ser >= 1")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch


def process(dataset_zip, output, batch_size):
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    model = MODELS / "abejas.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("La huella del modelo no coincide con sus metadatos.")
    session = session_for(model)
    counts = Counter()
    confusion = {label: dict.fromkeys(CLASSES, 0) for label in CLASSES}
    timings = []
    model_seconds = 0.0
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with Dataset(dataset_zip, metadata["dataset_manifest_sha256"]) as dataset, TemporaryDirectory(dir=output) as staging:
        rows = dataset.rows

        predictions_path = Path(staging) / "predicciones-dataset.csv"
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("ruta", "particion", "clase_real", "clase_predicha",
                             "puntaje_np", "puntaje_p", "lote"))
            for index, batch in enumerate(batched(rows, batch_size), start=1):
                inputs = np.stack([dataset.tensor(row) for row in batch])
                tick = perf_counter()
                logits = session.run(None, {"image": inputs})[0]
                seconds = perf_counter() - tick
                model_seconds += seconds
                if len(batch) == batch_size:
                    timings.append(seconds * 1000)
                weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                scores = weights / weights.sum(axis=1, keepdims=True)
                for row, score in zip(batch, scores, strict=True):
                    predicted = CLASSES[int(score.argmax())]
                    counts[row["split"]] += 1
                    if row["split"] == "test":
                        confusion[row["class_label"]][predicted] += 1
                    writer.writerow((row["relative_path"], row["split"],
                                     row["class_label"], predicted, *map(float, score), index))

        if dict(counts) != metadata["counts"]:
            raise ValueError("Los totales procesados difieren de los metadatos.")
        wall_seconds = perf_counter() - start
        correct = sum(confusion[label][label] for label in CLASSES)
        result = {
            "runtime": f"ONNX Runtime {ort.__version__} / CPUExecutionProvider",
            "platform": platform.platform(),
            "model_version": metadata["model_version"],
            "model_sha256": metadata["model_sha256"],
            "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
            "processed": len(rows),
            "partitions": dict(counts),
            "batch_size": batch_size,
            "full_batches": len(timings),
            "p95_full_batch_inference_ms": float(np.percentile(timings, 95)) if timings else None,
            "model_execution_seconds": model_seconds,
            "processing_seconds": wall_seconds,
            "processing_images_per_second": len(rows) / wall_seconds,
            "test": {"images": counts["test"], "correct": correct,
                     "accuracy": correct / counts["test"], "confusion": confusion},
        }
        summary_path = Path(staging) / "procesamiento-dataset.json"
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        predictions_path.replace(output / predictions_path.name)
        summary_path.replace(output / summary_path.name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="ZIP de las 714 imágenes originales.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, choices=range(1, 33), metavar="1-32")
    args = parser.parse_args()
    print(json.dumps(process(args.dataset, args.output, args.batch_size), indent=2))
