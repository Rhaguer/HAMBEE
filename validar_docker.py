"""Construye o verifica la imagen HAMBEE y registra las pruebas HTTP."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import monotonic, sleep

import httpx

from smoke import check

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "metadata.json"
DATASET = ROOT / "data" / "Dataset.zip"


def docker(*args, capture=True, timeout=300):
    return subprocess.run(
        ["docker", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE if capture else sys.stderr, timeout=timeout,
    ).stdout


def wait_healthy(container):
    deadline = monotonic() + 120
    while monotonic() < deadline:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", container))
        if state.get("Health", {}).get("Status") == "healthy":
            return
        if not state["Running"] or state.get("Health", {}).get("Status") == "unhealthy":
            break
        sleep(2)
    raise RuntimeError("El contenedor no alcanzó el estado healthy.")


def validate(image):
    metadata = json.loads(MODEL.read_text(encoding="utf-8"))
    info = json.loads(docker("image", "inspect", image))[0]
    container = docker("run", "-d", "-p", "127.0.0.1::8000", image).strip()
    try:
        wait_healthy(container)
        uid = int(docker("exec", container, "id", "-u").strip())
        if uid == 0:
            raise RuntimeError("El contenedor ejecuta la aplicación como root.")
        model_hash = docker("exec", container, "sha256sum", "/srv/models/abejas.onnx").split()[0]
        manifest_hash = docker(
            "exec", container, "python", "-c",
            "from hashlib import sha256; from zipfile import ZipFile; "
            "print(sha256(ZipFile('/srv/data/Dataset.zip').read('manifest.csv')).hexdigest())",
        ).strip()
        if (model_hash, manifest_hash) != (metadata["model_sha256"], metadata["dataset_manifest_sha256"]):
            raise RuntimeError("Las huellas del modelo o del dataset en Docker difieren de la entrega.")
        address = docker("port", container, "8000/tcp").strip()
        with httpx.Client(base_url=f"http://{address}", timeout=300, trust_env=False) as client:
            report = check(client, DATASET, panel=True, http_repeats=2)
        report.update({
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "docker_engine": docker("version", "--format", "{{.Server.Version}}").strip(),
            "docker_image": image,
            "docker_image_id": info["Id"],
            "docker_image_bytes": info["Size"],
            "docker_platform": f"{info['Os']}/{info['Architecture']}",
            "docker_container_uid": uid,
            "docker_healthcheck": "healthy",
            "model_sha256": model_hash,
            "dataset_manifest_sha256": manifest_hash,
        })
        if os.getenv("GITHUB_RUN_ID"):
            report["github_commit"] = os.environ["GITHUB_SHA"]
            report["github_run"] = (
                f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            )
        return report
    except Exception:
        subprocess.run(["docker", "logs", container], cwd=ROOT, stdout=sys.stderr, check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", container], cwd=ROOT, stdout=subprocess.DEVNULL, check=False)


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False, mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(output)


def main():
    metadata = json.loads(MODEL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", default=f"hambee:{metadata['model_version']}")
    parser.add_argument("--solo-probar", action="store_true", help="Usa una imagen ya construida.")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados-docker" / "api-docker.json")
    args = parser.parse_args()
    if not args.solo_probar:
        docker("build", "-t", args.imagen, ".", capture=False, timeout=1200)
    report = validate(args.imagen)
    write_report(args.salida, report)
    print(f"Prueba Docker aprobada. Evidencia: {args.salida}")


if __name__ == "__main__":
    main()
