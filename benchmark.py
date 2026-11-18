import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import onnxruntime as ort
import psutil

from app import MODELS, session_for
from dataset import Dataset
from preprocess import CLASSES

LEVELS = {"sin_optimizacion": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
          "optimizado": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(dataset, level, model=MODELS / "abejas.onnx", split="test"):
    manifest_hash = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))["dataset_manifest_sha256"]
    with Dataset(dataset, manifest_hash) as data:
        rows = data.partition(split)
        inputs = [data.tensor(row)[None] for row in rows]
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    startup = perf_counter()
    session = session_for(model, LEVELS[level])
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
    final_rss = process.memory_info().rss
    return {"startup_ms": startup_ms, "p50_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95)), "samples": len(timings),
            "rss_after_mib": final_rss / 2**20,
            "rss_delta_mib": (final_rss - baseline_rss) / 2**20,
            "accuracy": sum(confusion[i][i] for i in range(len(CLASSES))) / len(rows),
            "macro_f1": statistics.mean(f1), "confusion_np_p": confusion,
            "predictions": predictions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--level", choices=LEVELS)
    parser.add_argument("--output", type=Path, default=MODELS.parent / "evidencia" / "benchmark.json")
    parser.add_argument("--model", type=Path, default=MODELS / "abejas.onnx")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    args = parser.parse_args()
    if args.level:
        print(json.dumps(measure(args.dataset, args.level, args.model, args.split)))
    else:
        if args.split != "test" or args.model != MODELS / "abejas.onnx":
            parser.error("--model y --split requieren --level")
        result = {"platform": platform.platform(),
                  "cpu": cpu_name(),
                  "ort": ort.__version__, "model_bytes": (MODELS / "abejas.onnx").stat().st_size,
                  "model_sha256": hashlib.sha256((MODELS / "abejas.onnx").read_bytes()).hexdigest()}
        for level in LEVELS:
            command = [sys.executable, __file__, str(args.dataset), "--level", level]
            result[level] = json.loads(subprocess.check_output(command, text=True))
        result["identical_predictions"] = (
            result["optimizado"].pop("predictions") == result["sin_optimizacion"].pop("predictions"))
        path = args.output
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path.read_text(encoding="utf-8"))

