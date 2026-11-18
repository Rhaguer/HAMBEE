"""Mide lotes fijos en CPU como referencia para evaluar un agrupador dinámico."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from PIL import Image

from app import MODELS, session_for
from preprocess import prepare


BATCH_SIZES = (1, 2, 4, 8, 16)


def evaluate(dataset, repeats):
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    inputs = []
    for row in rows:
        path = dataset / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Archivo de prueba alterado: {path}")
        with Image.open(path) as image:
            inputs.append(prepare(image))
    if len(inputs) < max(BATCH_SIZES):
        raise ValueError("Se requieren al menos 16 imágenes de prueba.")

    images = np.stack(inputs)
    model = MODELS / "abejas.onnx"
    session = session_for(model)
    reference = np.array([session.run(None, {"image": image[None]})[0].argmax()
                          for image in images])
    batches = {
        size: [(start, images[start:start + size]) for start in range(0, len(images), size)]
        for size in BATCH_SIZES
    }
    results = {size: [] for size in BATCH_SIZES}
    for size, items in batches.items():
        predictions = np.concatenate([
            session.run(None, {"image": batch})[0].argmax(axis=1) for _, batch in items
        ])
        np.testing.assert_array_equal(predictions, reference)
        for _ in range(4):
            session.run(None, {"image": items[0][1]})

    # Alternar el orden reduce el efecto de calentamiento y cambios de frecuencia.
    rng = np.random.default_rng(20261117)
    for _ in range(repeats):
        for size in rng.permutation(BATCH_SIZES):
            for _, inputs in batches[size]:
                start = perf_counter()
                session.run(None, {"image": inputs})
                results[size].append((len(inputs), (perf_counter() - start) * 1000))

    metrics = []
    for size, values in results.items():
        full = [elapsed for count, elapsed in values if count == size]
        metrics.append({
            "batch_size": size,
            "full_batch_samples": len(full),
            "p50_full_batch_ms": float(np.median(full)),
            "p95_full_batch_ms": float(np.percentile(full, 95)),
            "images_per_second": sum(count for count, _ in values) * 1000 /
                                 sum(elapsed for _, elapsed in values),
        })
    return {
        "platform": platform.platform(), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "images": len(images), "repeats": repeats, "identical_predictions": True,
        "batch_metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats debe ser positivo")
    args.output.write_text(json.dumps(evaluate(args.dataset, args.repeats), indent=2) + "\n",
                           encoding="utf-8")
