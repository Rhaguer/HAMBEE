import argparse
import json
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient
import httpx

from app import app


def verify(client, sample):
    health = client.get("/health")
    health.raise_for_status()
    assert health.json()["status"] == "ready"
    assert health.json()["provider"] == "CPUExecutionProvider"
    payload = {"file": ("sample.jpg", sample.read_bytes(), "image/jpeg")}
    start = perf_counter()
    response = client.post("/predict", files=payload)
    request_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    prediction = response.json()
    scores = prediction["probabilities"]
    assert set(scores) == {"NP", "P"}
    assert abs(sum(scores.values()) - 1) < 1e-6
    assert prediction["label"] == max(scores, key=scores.get)
    assert prediction["confidence"] == max(scores.values())
    assert prediction["model_version"] == health.json()["model_version"]
    assert client.post("/predict").status_code == 422
    assert client.post("/predict", files={"file": ("bad.jpg", b"broken")}).status_code == 415
    assert client.post("/predict", files={"file": ("large.jpg", b"0" * 5_242_881)}).status_code == 413
    return {"health": health.json(), "sample": sample.name, "prediction": prediction,
            "request_ms": request_ms, "validated_status_codes": [200, 413, 415, 422]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--url")
    args = parser.parse_args()
    client = httpx.Client(base_url=args.url, timeout=20) if args.url else TestClient(app)
    with client:
        print(json.dumps(verify(client, args.sample), indent=2))

