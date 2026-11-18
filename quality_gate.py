import json
from pathlib import Path
import sys


def check(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    optimized = report["optimizado"]
    requirements = (
        (report["identical_predictions"], "La optimización cambió alguna clase."),
        (optimized["samples"] == 355, "Se esperaban 355 mediciones de 71 imágenes."),
        (optimized["macro_f1"] >= 0.94, "Macro-F1 inferior a 0,94."),
        (optimized["p95_ms"] <= 25, "p95 de inferencia superior a 25 ms."),
    )
    failures = [message for passed, message in requirements if not passed]
    if failures:
        raise SystemExit("\n".join(failures))
    print("Validación de calidad, paridad y latencia aprobada.")


if __name__ == "__main__":
    check(Path(sys.argv[1]))

