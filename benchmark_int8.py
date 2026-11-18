"""Ensayo FP32 vs. INT8 en validación. Calibra exclusivamente con entrenamiento."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from onnxruntime.quantization.shape_inference import quant_pre_process

from app import MODELS, ROOT
from benchmark import cpu_name
from dataset import Dataset


class Calibration(CalibrationDataReader):
    def __init__(self, tensors):
        self.tensors = tensors
        self.rewind()

    def get_next(self):
        return next(self.inputs, None)

    def rewind(self):
        self.inputs = ({"image": tensor[None]} for tensor in self.tensors)


def compare(archive):
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    with Dataset(archive, metadata["dataset_manifest_sha256"]) as dataset:
        training = dataset.partition("train")
        calibration = [dataset.tensor(row) for label in ("NP", "P")
                       for row in [r for r in training if r["class_label"] == label][::4][:64]]
    model = MODELS / "abejas.onnx"
    with TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "candidate-int8.onnx"
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(model, prepared, skip_optimization=True)
        quantize_static(prepared, candidate, Calibration(calibration), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
        result = {"split": "val", "calibration_split": "train", "calibration_images": len(calibration),
                  "model_sha256": metadata["model_sha256"], "cpu": cpu_name()}
        for label, artifact in (("fp32", model), ("int8", candidate)):
            command = [sys.executable, str(ROOT / "benchmark.py"), str(archive),
                       "--level", "optimizado", "--model", str(artifact), "--split", "val"]
            stats = json.loads(subprocess.check_output(command, text=True))
            stats.pop("predictions")
            result[label] = {"model_bytes": artifact.stat().st_size, **stats}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.dataset), indent=2))
