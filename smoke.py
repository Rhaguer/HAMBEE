"""Verifica FastAPI y la demostración con el conjunto de prueba empaquetado."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import statistics
from time import perf_counter

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import MAX_BYTES, MODELS, ROOT, app
from dataset import Dataset

SAMPLE = "test/P/P10109-127r.jpg"


def check(client, archive, panel, http_repeats=0):
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    with Dataset(archive, metadata["dataset_manifest_sha256"]) as dataset:
        sample = dataset.image_bytes(SAMPLE)
        dataset_paths = set(dataset.by_path)
    health = client.get("/health")
    health.raise_for_status()
    assert health.json() == {"status": "ready", "provider": "CPUExecutionProvider",
                             "model_version": metadata["model_version"]}
    client.get("/docs").raise_for_status()
    assert "/predict" in client.get("/openapi.json").json()["paths"]
    started = perf_counter()
    response = client.post("/predict", files={"file": ("sample.jpg", sample, "image/jpeg")})
    request_ms = (perf_counter() - started) * 1000
    response.raise_for_status()
    prediction = response.json()
    scores = prediction["probabilities"]
    assert set(scores) == {"NP", "P"} and all(0 <= score <= 1 for score in scores.values())
    assert abs(sum(scores.values()) - 1) < 1e-6
    assert prediction["label"] == max(scores, key=scores.get)
    assert prediction["confidence"] == max(scores.values())
    assert prediction["model_version"] == metadata["model_version"]
    assert prediction["inference_ms"] > 0
    missing = client.post("/predict").status_code
    invalid = client.post("/predict", files={"file": ("bad.jpg", b"broken")}).status_code
    too_large = client.post("/predict", files={"file": ("large.jpg", b"0" * (MAX_BYTES + 1))}).status_code
    with BytesIO() as buffer:
        Image.new("RGB", (4001, 3000)).save(buffer, format="PNG")
        too_many_pixels = client.post("/predict", files={"file": ("large.png", buffer.getvalue())}).status_code
    assert (missing, invalid, too_large, too_many_pixels) == (422, 415, 413, 413)
    result = {"health": health.json(), "sample": SAMPLE, "prediction": prediction,
              "request_ms": request_ms, "validated_status_codes": [response.status_code, too_large, invalid, missing],
              "resolution_limit_status": too_many_pixels}
    if http_repeats:
        with Dataset(archive, metadata["dataset_manifest_sha256"]) as dataset:
            images = [dataset.image_bytes(row["relative_path"]) for row in dataset.partition("test")]
        timings = []
        for _ in range(http_repeats):
            for image in images:
                started = perf_counter()
                client.post("/predict", files={"file": ("sample.jpg", image, "image/jpeg")}).raise_for_status()
                timings.append((perf_counter() - started) * 1000)
        result["http_sequential"] = {
            "samples": len(timings), "p50_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95)),
        }
    if panel:
        assert "HAMBEE" in client.get("/").raise_for_status().text
        client.get("/panel/panel.js").raise_for_status()
        stored = client.get("/demo/evidence").raise_for_status().json()
        assert sha256((MODELS / "abejas.onnx").read_bytes()).hexdigest() == stored["model"]["model_sha256"]
        evaluation = client.post("/demo/evaluate").raise_for_status().json()
        full, test = evaluation["full"], evaluation["test"]
        assert full["processed"] == len(dataset_paths)
        assert full["partitions"] == metadata["counts"]
        assert test["images"] == metadata["counts"]["test"]
        assert test["correct"] == full["test"]["correct"] == stored["full"]["test"]["correct"]
        assert test["confusion_np_p"] == stored["benchmark"]["optimizado"]["confusion_np_p"]
        artifact = client.get("/demo/predictions.csv").raise_for_status().content
        rows = list(csv.DictReader(StringIO(artifact.decode("utf-8-sig"))))
        assert len(rows) == len(dataset_paths) == len({row["ruta"] for row in rows})
        assert {row["ruta"] for row in rows} == dataset_paths
        sample_row = next(row for row in rows if row["ruta"] == SAMPLE)
        assert sample_row["clase_predicha"] == prediction["label"]
        page = client.get("/demo/images", params={"partition": "all"}).raise_for_status().json()
        assert page["total"] == len(dataset_paths)
        preview = client.get(f"/demo/images/{page['images'][0]['id']}").raise_for_status()
        with Dataset(archive, metadata["dataset_manifest_sha256"]) as dataset:
            assert preview.content == dataset.image_bytes(page["images"][0]["ruta"])
        result["panel"] = {"test": test, "full": full, "csv_rows": len(rows), "gallery_images": page["total"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "Dataset.zip")
    parser.add_argument("--url", help="URL local de la instancia Uvicorn o Docker.")
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--http-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.http_repeats < 0:
        parser.error("--http-repeats debe ser mayor o igual a cero")
    client = httpx.Client(base_url=args.url, timeout=30, trust_env=False) if args.url else TestClient(app)
    with client:
        print(json.dumps(check(client, args.dataset, args.panel, args.http_repeats), ensure_ascii=False, indent=2))
