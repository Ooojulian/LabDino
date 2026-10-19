"""Evidencias del entrenamiento: curvas de pérdida y resumen comparativo."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labdino.training import TrainResult


def plot_curve(ax, name: str, res: TrainResult) -> None:
    epochs = range(1, res.epochs_run + 1)
    ax.plot(epochs, res.train_losses, label="entrenamiento")
    ax.plot(epochs, res.val_losses, label="validación")
    ax.axvline(res.best_epoch, color="gray", ls="--", lw=1, label=f"mejor época ({res.best_epoch})")
    ax.set(title=name, xlabel="época", ylabel="pérdida (entropía cruzada, sin <PAD>)")
    ax.legend()
    ax.grid(alpha=0.3)


def save_curves(results: dict[str, TrainResult], out_dir: Path) -> None:
    """Una imagen por configuración + una comparativa con ambas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_curve(ax, f"Config {name}", res)
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_curve_{name}.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), sharey=True)
    for ax, (name, res) in zip(axes, results.items()):
        plot_curve(ax, f"Config {name}", res)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves_comparison.png", dpi=130)
    plt.close(fig)


def save_summary(rows: list[dict], winner: str, criterion: str, out_dir: Path) -> None:
    """training_summary.json + training_summary.md con la tabla comparativa."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_summary.json").write_text(
        json.dumps({"criterion": criterion, "winner": winner, "configs": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cols = [
        ("config", "Config"), ("architecture", "Arquitectura"), ("num_layers", "Capas"),
        ("hidden_size", "Hidden"), ("dropout", "Dropout"), ("epochs_run", "Épocas corridas"),
        ("max_epochs", "Máx. épocas"), ("patience", "Paciencia"), ("loss", "Pérdida"),
        ("optimizer", "Optimizador"), ("lr", "lr"), ("batch_size", "Batch"),
        ("n_params", "Parámetros"), ("best_val_loss", "Mejor val loss"), ("best_epoch", "Época del mejor"),
    ]
    lines = ["# Resumen de entrenamiento", "", f"**Criterio de elección (fijado antes de entrenar):** {criterion}", "",
             f"**Configuración ganadora:** {winner}", "",
             "| Métrica | " + " | ".join(r["config"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for key, label in cols[1:]:
        vals = [f"{r[key]:.4f}" if isinstance(r[key], float) and key == "best_val_loss" else str(r[key]) for r in rows]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "Notas: la pérdida de entrenamiento es el promedio por token durante la época (con dropout activo); "
              "la de validación se calcula en modo eval sin gradientes. El checkpoint guardado es el de la mejor "
              "época de validación, no el último.", "",
              "![comparativa](loss_curves_comparison.png)", ""]
    (out_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")
