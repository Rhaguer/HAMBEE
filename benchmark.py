import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image
import psutil

from app import MODELS, session_for
from preprocess import CLASSES, prepare

LEVELS = {"sin_optimizacion": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
          "optimizado": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def measure(dataset, level):
    with (dataset / "manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    inputs = []
    for row in rows:
        path = dataset / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Archivo de prueba alterado: {path}")
        with Image.open(path) as image:
            inputs.append(prepare(image)[None])
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    startup = perf_counter()
    session = session_for(MODELS / "abejas.onnx", LEVELS[level])
    startup_ms = (perf_counter() - startup) * 1000
    for image in inputs[:20]:
        session.run(None, {"image": image})
    predictions, timings = [], []
    for _ in range(5):
        for image in inputs:
            start = perf_counter()
            output = session.run(None, {"image": image})[0]
            timings.append((perf_counter() - start) * 1000)
            predictions.append(CLASSES[int(output.argmax())])
    predictions = predictions[:len(rows)]
    confusion = [[0, 0], [0, 0]]
    for row, prediction in zip(rows, predictions):
        confusion[CLASSES.index(row["class_label"])][CLASSES.index(prediction)] += 1
    f1 = [2 * confusion[i][i] / (sum(confusion[i]) + sum(row[i] for row in confusion))
          for i in range(len(CLASSES))]
    return {"startup_ms": startup_ms, "p50_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95)), "samples": len(timings),
            "rss_after_mib": process.memory_info().rss / 2**20,
            "rss_delta_mib": (process.memory_info().rss - baseline_rss) / 2**20,
            "accuracy": sum(confusion[i][i] for i in range(len(CLASSES))) / len(rows),
            "macro_f1": statistics.mean(f1), "confusion_np_p": confusion,
            "predictions": predictions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--level", choices=LEVELS)
    parser.add_argument("--output", type=Path, default=MODELS.parent / "evidencia" / "benchmark.json")
    args = parser.parse_args()
    if args.level:
        print(json.dumps(measure(args.dataset, args.level)))
    else:
        result = {"platform": platform.platform(),
                  "cpu": os.getenv("PROCESSOR_IDENTIFIER") or platform.processor(),
                  "ort": ort.__version__, "model_bytes": (MODELS / "abejas.onnx").stat().st_size}
        for level in LEVELS:
            command = [sys.executable, __file__, str(args.dataset), "--level", level]
            result[level] = json.loads(subprocess.check_output(command, text=True))
        result["identical_predictions"] = (
            result["optimizado"].pop("predictions") == result["sin_optimizacion"].pop("predictions"))
        path = args.output
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path.read_text(encoding="utf-8"))

