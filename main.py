"""Pipeline principal para classificacao de risco de cancer cervical.

Execute com:
    python main.py
"""
from __future__ import annotations

from pathlib import Path

from src.evaluate import evaluate_and_save_results
from src.preprocessing import (
    load_data,
    prepare_features,
    split_data,
    standardize_column_names,
)
from src.train import train_random_forest
from src.utils import ensure_directory, log_message


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cervical_cancer.csv"
FALLBACK_DATA_PATH = BASE_DIR / "data" / "kag_risk_factors_cervical_cancer.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TARGET_COLUMN = "Biopsy"
RANDOM_STATE = 42


def main() -> None:
    """Executa carga, preprocessamento, treino, avaliacao e gravacao de outputs."""
    ensure_directory(OUTPUT_DIR)

    data_path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    log_message(f"Carregando dataset: {data_path}")
    df = load_data(data_path)
    df = standardize_column_names(df)

    log_message("Separando variaveis preditoras e alvo.")
    X, y = prepare_features(df, target_column=TARGET_COLUMN)

    log_message("Dividindo dados em treino e teste.")
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        scale_features=False,
    )

    log_message("Treinando RandomForestClassifier.")
    model = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)

    log_message("Gerando previsoes e metricas.")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = evaluate_and_save_results(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        output_dir=OUTPUT_DIR,
        model_name="RandomForestClassifier",
    )

    log_message("Execucao finalizada com sucesso.")
    log_message(f"Metricas salvas em: {OUTPUT_DIR / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
