"""Registra una validación real y sus artefactos; no vuelve a entrenar el modelo."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mlflow

ROOT = Path(__file__).resolve().parent


def register(evidence, output):
    metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    benchmark = json.loads((evidence / "benchmark-ci.json").read_text(encoding="utf-8"))
    docker = json.loads((evidence / "api-docker-ci.json").read_text(encoding="utf-8"))
    model = ROOT / "models/abejas.onnx"
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    with ZipFile(ROOT / "data/Dataset.zip") as archive:
        manifest = archive.read("manifest.csv")
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    if not (model_hash == metadata["model_sha256"] == benchmark["model_sha256"] == docker["model_sha256"]
            and manifest_hash == metadata["dataset_manifest_sha256"] == docker["dataset_manifest_sha256"]):
        raise ValueError("La evidencia no corresponde a este modelo y dataset.")
    output.mkdir(parents=True, exist_ok=True)
    # SQLite y artefactos locales: el job puede archivarlos sin un servicio externo.
    database = (output / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri("sqlite:///" + database)
    experiment_name = "HAMBEE-EVA4"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=(output / "artifacts").resolve().as_uri())
    else:
        experiment_id = experiment.experiment_id
    commit = os.getenv("GITHUB_SHA") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"validacion-{metadata['model_version']}-{commit[:8]}") as run:
        mlflow.set_tags({"stage": "validacion_preliberacion", "git_commit": commit,
                         "model_sha256": model_hash, "dataset_manifest_sha256": manifest_hash,
                         "training_history_origin": "historial incluido; no es un nuevo entrenamiento",
                         "github_run": docker.get("github_run", "local")})
        mlflow.log_params({key: metadata[key] for key in (
            "model_version", "architecture", "best_epoch", "seed", "batch_size", "learning_rate", "opset")})
        mlflow.log_params({"train_images": metadata["counts"]["train"],
                           "val_images": metadata["counts"]["val"],
                           "test_images": metadata["counts"]["test"], "provider": "CPUExecutionProvider"})
        optimized = benchmark["optimizado"]
        mlflow.log_metrics({"test_accuracy": optimized["accuracy"], "test_macro_f1": optimized["macro_f1"],
                            "runtime_p95_ms": optimized["p95_ms"], "runtime_rss_mib": optimized["rss_after_mib"],
                            "http_p95_ms": docker["http_sequential"]["p95_ms"],
                            "model_bytes": model.stat().st_size})
        for item in metadata["history"]:
            mlflow.log_metrics({"historical_train_loss": item["train_loss"],
                                "historical_val_loss": item["val_loss"],
                                "historical_val_accuracy": item["val_accuracy"]}, step=item["epoch"])
        mlflow.log_artifact(str(model), "model")
        mlflow.log_artifact(str(ROOT / "models/metadata.json"), "model")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_bytes(manifest)
            mlflow.log_artifact(str(path), "data")
        for path in sorted(evidence.glob("*-ci.json")):
            mlflow.log_artifact(str(path), "validation")
        run_id = run.info.run_id
    # Leer del almacén comprueba que el registro persistió los parámetros y métricas.
    saved = mlflow.MlflowClient().get_run(run_id)
    if saved.info.status != "FINISHED" or saved.data.tags["model_sha256"] != model_hash:
        raise RuntimeError("El registro MLflow no quedó finalizado.")
    result = {"run_id": run_id, "experiment": experiment_name, "status": saved.info.status,
              "git_commit": commit, "model_sha256": model_hash, "dataset_manifest_sha256": manifest_hash,
              "metrics": saved.data.metrics, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "github_run": docker.get("github_run"),
              "tracking_backend": "SQLite; descargar evidencia-mlops para inspección local"}
    (output / "registro-mlflow.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidencia", type=Path, default=ROOT)
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados-mlops")
    args = parser.parse_args()
    register(args.evidencia, args.salida)
