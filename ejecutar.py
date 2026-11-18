"""Prepara el entorno local y abre la demostración HAMBEE."""
import argparse
import hashlib
from pathlib import Path
import socket
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parent


def show_when_ready(url, open_browser):
    opener = build_opener(ProxyHandler({}))
    deadline = monotonic() + 60
    while monotonic() < deadline:
        try:
            with opener.open(url + "/health", timeout=1):
                break
        except OSError:
            sleep(0.25)
    else:
        print("El servicio no respondió en /health. Revisa el registro de Uvicorn.", flush=True)
        return

    panel = url + "/?ejecutar=1"
    print(f"Panel HAMBEE: {panel}", flush=True)
    print(f"Documentación de la API: {url}/docs", flush=True)
    if open_browser:
        try:
            opened = webbrowser.open(panel)
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            print("Abre manualmente la dirección del panel en tu navegador.", flush=True)


def prepare_environment():
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print("Creando el entorno virtual en este equipo.", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / "hambee-requirements.sha256"
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != digest:
        print("Instalando dependencias de inferencia.", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_text(digest, encoding="ascii")
    return python


def serve(port, open_browser):
    import uvicorn

    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise SystemExit(f"No se pudo abrir el puerto {port}: {error}") from error
        address, actual_port = listener.getsockname()
        server = uvicorn.Server(uvicorn.Config("app:app", host=address, port=actual_port))
        Thread(
            target=show_when_ready,
            args=(f"http://{address}:{actual_port}", open_browser),
            daemon=True,
        ).start()
        server.run(sockets=[listener])
        if not server.started:
            raise SystemExit("El servicio HAMBEE no pudo iniciar. Revisa el registro de Uvicorn.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int, default=0, help="Puerto local. 0 elige uno disponible.")
    parser.add_argument("--sin-navegador", action="store_true", help="Solo muestra la URL en terminal.")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.puerto <= 65535:
        parser.error("--puerto debe estar entre 0 y 65535.")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Ejecuta HAMBEE con Python 3.12. En VS Code selecciona ese intérprete.")
    if args.serve:
        serve(args.puerto, not args.sin_navegador)
        return

    try:
        python = prepare_environment()
        command = [str(python), str(Path(__file__).resolve()), "--serve", *sys.argv[1:]]
        completed = subprocess.run(command, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"No se pudo preparar el entorno de HAMBEE: {error}") from error
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
