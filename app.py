from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from preprocess import CLASSES, SIZE, prepare

MODELS = Path(__file__).resolve().parent / "models"
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000


def session_for(model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


@asynccontextmanager
async def lifespan(app):
    path = MODELS / "abejas.onnx"
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise RuntimeError("El modelo no coincide con la versión declarada.")
    app.state.session = session_for(path)
    app.state.session.run(None, {"image": np.zeros((1, 3, SIZE[1], SIZE[0]), np.float32)})
    app.state.version = metadata["model_version"]
    yield


app = FastAPI(title="Clasificador de abejas P/NP", version="1.0.0", lifespan=lifespan)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    inference_ms: float


@app.get("/health")
def health():
    return {"status": "ready", "model_version": app.state.version,
            "provider": app.state.session.get_providers()[0]}


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "El archivo excede 5 MiB.")
    try:
        with Image.open(BytesIO(data), formats=["JPEG", "PNG"]) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "La imagen excede 12 megapíxeles.")
            inputs = prepare(image)[None]
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "La imagen excede el límite de píxeles.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(415, "Se requiere una imagen JPEG o PNG válida.") from exc

    start = perf_counter()
    logits = app.state.session.run(None, {"image": inputs})[0][0]
    inference_ms = (perf_counter() - start) * 1000
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    winner = int(probabilities.argmax())
    return Prediction(label=CLASSES[winner], confidence=float(probabilities[winner]),
                      probabilities=dict(zip(CLASSES, map(float, probabilities))),
                      model_version=app.state.version, inference_ms=inference_ms)

