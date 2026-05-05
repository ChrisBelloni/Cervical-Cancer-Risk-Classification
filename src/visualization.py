"""Geração de gráficos e artefatos de saída."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_score, title: str, output_path: Path) -> None:
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(best_model, feature_names, output_path: Path) -> pd.DataFrame:
    model_step = best_model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importance_values = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importance_values = abs(model_step.coef_[0])
    else:
        return pd.DataFrame(columns=["variavel", "importancia"])

    importances = (
        pd.DataFrame({"variavel": feature_names, "importancia": importance_values})
        .sort_values("importancia", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importances, x="importancia", y="variavel")
    plt.title("Principais variáveis para o modelo escolhido")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return importances
