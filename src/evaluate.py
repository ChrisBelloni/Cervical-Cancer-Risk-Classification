import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

def evaluate_models(models, X_test, y_test):
    """Avalia os modelos e retorna uma tabela de métricas."""
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0)
        })

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(classification_report(y_test, y_pred, zero_division=0))

    return pd.DataFrame(results)

def plot_confusion_matrix(model, X_test, y_test, model_name, output_path):
    """Gera e salva matriz de confusão."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negativo", "Positivo"],
        yticklabels=["Negativo", "Positivo"]
    )
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    """Gera e salva gráfico de importância das variáveis para Random Forest."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Top 15 Variáveis Mais Importantes - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
