"""Clasifica el dataset completo en lotes y exporta la auditoría en CSV y JSON."""

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
from itertools import batched
import json
from pathlib import Path, PurePosixPath
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort
from PIL import Image

from app import MODELS, session_for
from preprocess import CLASSES, prepare


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
    with ZipFile(dataset_zip) as archive, TemporaryDirectory(dir=output) as staging:
        manifests = [name for name in archive.namelist() if PurePosixPath(name).name == "manifest.csv"]
        if len(manifests) != 1:
            raise ValueError("Se esperaba un manifiesto en el ZIP.")
        manifest = manifests[0]
        data = archive.read(manifest)
        if hashlib.sha256(data).hexdigest() != metadata["dataset_manifest_sha256"]:
            raise ValueError("El manifiesto no coincide con el dataset de la EVA4.")
        root = PurePosixPath(manifest).parent
        rows = list(csv.DictReader(data.decode("utf-8-sig").splitlines()))

        predictions_path = Path(staging) / "predicciones-dataset.csv"
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("ruta", "particion", "variante", "clase_real", "clase_predicha",
                             "puntaje_np", "puntaje_p", "lote"))
            for index, batch in enumerate(batched(rows, batch_size), start=1):
                tensors = []
                for row in batch:
                    image_bytes = archive.read((root / row["relative_path"]).as_posix())
                    if hashlib.sha256(image_bytes).hexdigest() != row["sha256"]:
                        raise ValueError(f"Imagen alterada: {row['relative_path']}")
                    with Image.open(BytesIO(image_bytes)) as image:
                        tensors.append(prepare(image))
                inputs = np.stack(tensors)
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
                    writer.writerow((row["relative_path"], row["split"], row["variant"],
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
    parser.add_argument("dataset", type=Path, help="ZIP original del dataset de la EVA4.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, choices=range(1, 33), metavar="1-32")
    args = parser.parse_args()
    print(json.dumps(process(args.dataset, args.output, args.batch_size), indent=2))
