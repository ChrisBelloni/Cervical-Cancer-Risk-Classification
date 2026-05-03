from pathlib import Path
import joblib

from src.utils import ensure_directories
from src.preprocessing import load_dataset, prepare_data, split_data
from src.train import train_models
from src.evaluate import (
    evaluate_models,
    plot_confusion_matrix,
    plot_feature_importance
)

DATA_PATH = Path("data/cervical_cancer.csv")

def main():
    ensure_directories()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset não encontrado. Baixe a base do Kaggle e salve como: "
            "data/cervical_cancer.csv"
        )

    print("Carregando dataset...")
    df = load_dataset(DATA_PATH)

    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")
    print("\nDistribuição da variável alvo Biopsy:")
    print(df["Biopsy"].value_counts(dropna=False))

    print("\nPreparando dados...")
    X, y, feature_names, imputer, scaler = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTreinando modelos...")
    models = train_models(X_train, y_train)

    print("\nAvaliando modelos...")
    results = evaluate_models(models, X_test, y_test)

    print("\nTabela final de métricas:")
    print(results.round(4))

    results.to_csv("outputs/metrics.csv", index=False)

    best_model_name = results.sort_values(
        by=["Recall", "F1-score"],
        ascending=False
    ).iloc[0]["Modelo"]

    best_model = models[best_model_name]
    print(f"\nMelhor modelo considerando Recall e F1-score: {best_model_name}")

    plot_confusion_matrix(
        best_model,
        X_test,
        y_test,
        best_model_name,
        "outputs/figures/matriz_confusao.png"
    )

    if "Random Forest" in models:
        plot_feature_importance(
            models["Random Forest"],
            feature_names,
            "outputs/figures/importancia_variaveis.png"
        )

    joblib.dump(best_model, "outputs/models/melhor_modelo.pkl")

    print("\nExecução concluída!")
    print("Arquivos gerados:")
    print("- outputs/metrics.csv")
    print("- outputs/figures/matriz_confusao.png")
    print("- outputs/figures/importancia_variaveis.png")
    print("- outputs/models/melhor_modelo.pkl")

if __name__ == "__main__":
    main()
