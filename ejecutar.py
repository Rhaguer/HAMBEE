"""Arranca la demostración en el navegador desde el botón Ejecutar de VS Code."""

import hashlib
from pathlib import Path
import socket
import subprocess
import sys
from threading import Thread
from time import sleep
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent


def open_when_ready(url):
    for _ in range(120):
        try:
            with urlopen(url + "/health", timeout=1):
                webbrowser.open(url + "/?ejecutar=1")
                return
        except (OSError, URLError):
            sleep(0.25)


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Selecciona Python 3.12 como intérprete en VS Code.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / "hambee-requirements.sha256"
    if not marker.is_file() or marker.read_text(encoding="ascii") != digest:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_text(digest, encoding="ascii")
    if "--serve" not in sys.argv:
        subprocess.run([str(python), str(Path(__file__).resolve()), "--serve"], cwd=ROOT, check=True)
        return

    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address, port = listener.getsockname()
        server = uvicorn.Server(uvicorn.Config("app:app", host=address, port=port))
        Thread(target=open_when_ready, args=(f"http://{address}:{port}",), daemon=True).start()
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
