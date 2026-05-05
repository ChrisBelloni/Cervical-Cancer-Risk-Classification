"""Funcoes de treinamento de modelos."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Treina um RandomForestClassifier reprodutivel."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
