"""Genera el panel de cuatro gráficos a partir de las métricas de una ejecución."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def render(results, output, panels=None):
    model = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    benchmark = json.loads((results / "benchmark.json").read_text(encoding="utf-8"))
    batches = json.loads((results / "lotes-cpu.json").read_text(encoding="utf-8"))
    panels = panels or ("latencia", "matriz", "lotes", "historial")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), layout="constrained") if len(panels) == 4 else (
        plt.subplots(1, len(panels), figsize=(5.9 * len(panels), 3.45), layout="constrained"))
    green, gray, accent = "#236A51", "#81968D", "#A0600C"
    for key, ax in zip(panels, np.asarray(axes).ravel(), strict=True):
        if key == "latencia":
            positions = np.arange(2)
            for offset, label, color, data in (
                (-.18, "Sin optimización", gray, benchmark["sin_optimizacion"]),
                (.18, "Grafo optimizado", green, benchmark["optimizado"]),
            ):
                bars = ax.bar(positions + offset, [data["p50_ms"], data["p95_ms"]], .31,
                              color=color, label=label)
                ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.set_xticks(positions, ["p50", "p95"])
            ax.set_ylim(0, 1.30 * max(benchmark[name]["p95_ms"]
                                     for name in ("optimizado", "sin_optimizacion")))
            ax.set(title="Latencia de ONNX en CPU", ylabel="Milisegundos")
            ax.legend(frameon=False)
        elif key == "matriz":
            matrix = np.asarray(benchmark["optimizado"]["confusion_np_p"])
            ax.imshow(matrix, cmap="Greens", vmin=0, vmax=max(1, matrix.max()))
            ax.set_xticks([0, 1], ["NP", "P"])
            ax.set_yticks([0, 1], ["NP", "P"])
            ax.set(title="Prueba reservada", xlabel="Clase predicha", ylabel="Clase real")
            for index, value in np.ndenumerate(matrix):
                ax.text(index[1], index[0], str(value), ha="center", va="center",
                        color="white" if value > matrix.max() / 2 else "#182A23", fontsize=13)
        elif key == "lotes":
            sizes = [row["batch_size"] for row in batches["batch_metrics"]]
            rates = [row["images_per_second"] for row in batches["batch_metrics"]]
            ax.plot(sizes, rates, "o-", color=green)
            ax.set(title="Capacidad por tamaño de lote", xlabel="Imágenes por lote",
                   ylabel="Imágenes/s", xticks=sizes)
            ax.grid(axis="y", alpha=.2)
        else:
            history = model["history"]
            best = next(row for row in history if row["epoch"] == model["best_epoch"])
            ax.semilogy([row["epoch"] for row in history], [row["val_loss"] for row in history],
                        color=green)
            ax.scatter(best["epoch"], best["val_loss"], color=accent,
                       label=f"Época {best['epoch']}", zorder=3)
            ax.set(title="Pérdida de validación", xlabel="Época", ylabel="Entropía cruzada")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=.2)
    fig.savefig(output, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.results, args.output)
