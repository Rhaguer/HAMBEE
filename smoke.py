import argparse
import json
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient
import httpx

from app import app


def verify(client, sample, panel=False):
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
    result = {"health": health.json(), "sample": sample.name, "prediction": prediction,
              "request_ms": request_ms, "validated_status_codes": [200, 413, 415, 422]}
    if panel:
        page = client.get("/")
        page.raise_for_status()
        assert "HAMBEE" in page.text
        client.get("/panel/panel.js").raise_for_status()
        metrics = client.get("/demo/evidence")
        metrics.raise_for_status()
        assert metrics.json()["model"]["best_epoch"] == 37
        evaluation = client.post("/demo/evaluate", timeout=160)
        evaluation.raise_for_status()
        result["panel"] = evaluation.json()
        assert result["panel"]["test"]["images"] == 71
        assert result["panel"]["test"]["correct"] == 70
        gallery = client.get("/demo/images")
        gallery.raise_for_status()
        page = gallery.json()
        assert page["total"] == 71 and len(page["images"]) == 8
        image = client.get(f"/demo/images/{page['images'][0]['id']}")
        image.raise_for_status()
        assert image.headers["content-type"].startswith("image/")
        if result["panel"]["full"]:
            csv_file = client.get("/demo/predictions.csv")
            csv_file.raise_for_status()
            assert len(csv_file.content.splitlines()) == 7141
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    client = httpx.Client(base_url=args.url, timeout=20) if args.url else TestClient(app)
    with client:
        print(json.dumps(verify(client, args.sample, args.panel), indent=2))

