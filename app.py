from contextlib import asynccontextmanager
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal
from zipfile import BadZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from dataset import Dataset
from preprocess import CLASSES, SIZE, prepare

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
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
    app.state.metadata = metadata
    app.state.demo_lock = Lock()
    app.state.demo_images = ([], None)
    with TemporaryDirectory(prefix="hambee-demo-") as output:
        app.state.demo_output = Path(output)
        yield


app = FastAPI(
    title="HAMBEE · API de inferencia",
    description=(
        "Clasifica imágenes de abejas en **P (con polen)** o **NP (sin polen)** "
        "mediante un modelo ONNX ejecutado en CPU. Usa **GET /health** para "
        "comprobar que el modelo está listo y **POST /predict** para cargar una imagen."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


class Health(BaseModel):
    status: Literal["ready"] = Field(description="El modelo se cargó y puede recibir solicitudes.")
    model_version: str = Field(examples=["1.2.0"])
    provider: str = Field(examples=["CPUExecutionProvider"])


class Prediction(BaseModel):
    label: Literal["NP", "P"] = Field(description="P: con polen. NP: sin polen.")
    confidence: float = Field(description="Puntaje softmax de la clase predicha, entre 0 y 1.")
    probabilities: dict[str, float] = Field(
        description="Puntajes softmax para NP y P. Suman aproximadamente 1."
    )
    model_version: str = Field(description="Versión del modelo cargado.")
    inference_ms: float = Field(
        description="Tiempo de ejecución de ONNX en milisegundos. Excluye carga, preprocesamiento y HTTP."
    )


@app.get(
    "/health", response_model=Health, tags=["Estado"], summary="Comprobar disponibilidad del modelo",
    responses={200: {"content": {"application/json": {"example": {
        "status": "ready", "model_version": "1.2.0", "provider": "CPUExecutionProvider",
    }}}}},
)
def health():
    return {"status": "ready", "model_version": app.state.version,
            "provider": app.state.session.get_providers()[0]}


@app.post(
    "/predict",
    response_model=Prediction,
    tags=["Inferencia"],
    summary="Clasificar una imagen de abeja",
    description=(
        "En **Try it out**, selecciona un JPEG o PNG de hasta **5 MiB** y "
        "**12 megapíxeles**. Para reproducir el ejemplo, usa la imagen incluida "
        "`test/P/P10109-127r.jpg` del dataset. El tiempo varía según el equipo. "
        "Si falta el campo `file`, la API responde con 422."
    ),
    responses={
        200: {"description": "Imagen clasificada.", "content": {"application/json": {"example": {
            "label": "P",
            "confidence": 0.9999,
            "probabilities": {"NP": 0.0001, "P": 0.9999},
            "model_version": "1.2.0",
            "inference_ms": 1.0,
        }}}},
        413: {"description": "El archivo o la imagen supera los límites permitidos.",
              "content": {"application/json": {"examples": {
                  "archivo": {"summary": "Archivo demasiado grande",
                              "value": {"detail": "El archivo excede 5 MiB."}},
                  "resolucion": {"summary": "Resolución demasiado alta",
                                 "value": {"detail": "La imagen excede 12 megapíxeles."}},
              }}}},
        415: {"description": "Formato incompatible o imagen dañada.",
              "content": {"application/json": {"example": {
                  "detail": "Se requiere una imagen JPEG o PNG válida.",
              }}}},
    },
)
def predict(file: Annotated[UploadFile, File(description="Imagen JPEG o PNG de una abeja.")]):
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


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "panel" / "index.html")


@app.get("/demo/evidence", include_in_schema=False)
def evidence():
    from demostracion import dataset_archive

    files = {key: json.loads((ROOT / "evidencia" / filename).read_text(encoding="utf-8"))
             for key, filename in (("benchmark", "benchmark.json"), ("batches", "lotes-cpu.json"),
                                   ("full", "procesamiento-dataset.json"))}
    return {**files, "model": app.state.metadata, "dataset_available": dataset_archive() is not None}


@app.post("/demo/evaluate", include_in_schema=False)
def evaluate_demo():
    from demostracion import dataset_archive, evaluate_test
    from procesar_dataset import process

    if not app.state.demo_lock.acquire(blocking=False):
        raise HTTPException(409, "Ya hay una evaluación en curso.")
    try:
        archive = dataset_archive()
        if archive is None:
            raise HTTPException(404, "Configura el ZIP de las 714 imágenes para evaluar.")
        try:
            test = evaluate_test(app.state.session, archive, app.state.metadata["dataset_manifest_sha256"])
            test.pop("results")
            full = process(archive, app.state.demo_output, batch_size=8)
            with (app.state.demo_output / "predicciones-dataset.csv").open(
                    encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (BadZipFile, KeyError, ValueError, OSError) as error:
            raise HTTPException(422, "El dataset no coincide con la versión del modelo.") from error
        app.state.demo_images = (rows, archive)
        return {"test": test, "full": full}
    finally:
        app.state.demo_lock.release()


@app.get("/demo/predictions.csv", include_in_schema=False)
def download_predictions():
    path = app.state.demo_output / "predicciones-dataset.csv"
    if not path.is_file():
        raise HTTPException(404, "Ejecute la evaluación completa para generar el CSV.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/demo/images", include_in_schema=False)
def list_images(partition: Literal["test", "val", "train", "all"] = "test", search: str = "",
                page: int = 1):
    rows, _ = app.state.demo_images
    found = [{"id": index, **row} for index, row in enumerate(rows)
             if (partition == "all" or row["particion"] == partition)
             and search.casefold() in row["ruta"].casefold()]
    total_pages = max(1, (len(found) + 7) // 8)
    page = min(total_pages, max(1, page))
    return {"images": found[(page - 1) * 8:page * 8], "page": page,
            "pages": total_pages, "total": len(found)}


@app.get("/demo/images/{image_id}", include_in_schema=False)
def image_preview(image_id: int):
    rows, archive = app.state.demo_images
    if not 0 <= image_id < len(rows):
        raise HTTPException(404, "Imagen no encontrada.")
    path = rows[image_id]["ruta"]
    with Dataset(archive, app.state.metadata["dataset_manifest_sha256"]) as dataset:
        data = dataset.image_bytes(path)
    return Response(data, media_type="image/jpeg")


app.mount("/panel", StaticFiles(directory=ROOT / "panel", check_dir=False), name="panel")

