"""Calculo de metricas e geracao dos artefatos de avaliacao."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import ensure_directory, save_json, save_text


def calculate_metrics(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """Calcula metricas principais para classificacao binaria."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def save_classification_report(y_true, y_pred, output_path: Path) -> None:
    """Salva o classification_report em arquivo texto."""
    report = classification_report(y_true, y_pred, zero_division=0)
    save_text(output_path, report)


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """Gera e salva o grafico da matriz de confusao."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predito 0", "Predito 1"],
        yticklabels=["Real 0", "Real 1"],
    )
    plt.title("Matriz de Confusao")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_and_save_results(
    y_true,
    y_pred,
    y_proba,
    output_dir: Path | str,
    model_name: str,
) -> dict[str, Any]:
    """Calcula metricas e salva todos os arquivos esperados em outputs/."""
    output_path = Path(output_dir)
    ensure_directory(output_path)

    metrics = calculate_metrics(y_true, y_pred, y_proba)
    metrics["model"] = model_name

    save_json(output_path / "metrics.json", metrics)
    save_classification_report(y_true, y_pred, output_path / "classification_report.txt")
    save_confusion_matrix(y_true, y_pred, output_path / "confusion_matrix.png")

    return metrics
