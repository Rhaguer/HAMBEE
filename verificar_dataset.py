"""Verifica el conjunto original y su vínculo con el modelo exportado."""
import argparse
import hashlib
import json
from pathlib import Path

from dataset import Dataset

MODELS = Path(__file__).resolve().parent / "models"


def verify(archive):
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    model_hash = hashlib.sha256((MODELS / "abejas.onnx").read_bytes()).hexdigest()
    if model_hash != metadata["model_sha256"]:
        raise ValueError("El SHA-256 del modelo no coincide.")
    with Dataset(archive, metadata["dataset_manifest_sha256"]) as dataset:
        for path in dataset.by_path:
            dataset.image_bytes(path)
        groups = {split: len(values) for split, values in dataset.groups.items()}
        if groups != metadata["groups"] or dataset.counts != metadata["counts"]:
            raise ValueError("Las particiones no coinciden con los metadatos.")
        return {
            "model_version": metadata["model_version"], "model_sha256": model_hash,
            "dataset_manifest_sha256": dataset.manifest_sha256, "images_verified": len(dataset.rows),
            "counts": dict(dataset.counts), "groups": groups, "overlapping_origin_groups": 0,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset), indent=2))
