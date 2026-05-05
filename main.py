"""Execução principal do projeto Tech Challenge - Fase 1.

O script treina modelos de classificação para apoiar a triagem de risco de
câncer do colo do útero usando a variável alvo `Biopsy`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.data_processing import clean_data, load_data, prepare_features, split_train_validation_test
from src.modeling import evaluate_model, train_and_select_model
from src.visualization import save_confusion_matrix, save_feature_importance, save_roc_curve

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Executa o pipeline completo: carga, limpeza, treino, avaliação e outputs."""
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    X, y = prepare_features(df_clean)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _X_trainval,
        _y_trainval,
    ) = split_train_validation_test(X, y)

    best_name, best_model, validation_metrics, _fitted_models = train_and_select_model(
        X_train, y_train, X_val, y_val
    )
    validation_metrics.to_csv(OUTPUT_DIR / "metrics_validation.csv", index=False)

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = pd.DataFrame(
        [{"modelo_escolhido": best_name, **evaluate_model(best_model, X_test, y_test)}]
    )
    test_metrics.to_csv(OUTPUT_DIR / "metrics_test.csv", index=False)

    with open(OUTPUT_DIR / "classification_report_test.txt", "w", encoding="utf-8") as file:
        file.write(f"Modelo escolhido: {best_name}\n\n")
        file.write(classification_report(y_test, y_test_pred, zero_division=0))

    save_confusion_matrix(
        y_test,
        y_test_pred,
        f"Matriz de Confusão - Teste ({best_name})",
        OUTPUT_DIR / "confusion_matrix_test.png",
    )
    save_roc_curve(
        y_test,
        y_test_proba,
        f"Curva ROC - Teste ({best_name})",
        OUTPUT_DIR / "roc_curve_test.png",
    )
    importances = save_feature_importance(
        best_model, X.columns, OUTPUT_DIR / "feature_importance.png"
    )
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    print("Projeto executado com sucesso.")
    print("Melhor modelo:", best_name)
    print("Métricas de validação:")
    print(validation_metrics)
    print("Métricas de teste:")
    print(test_metrics)


if __name__ == "__main__":
    main()
