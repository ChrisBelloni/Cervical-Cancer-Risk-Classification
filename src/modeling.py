"""Modelagem, avaliação e seleção de modelos."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def make_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, X, y) -> dict:
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }


def train_and_select_model(X_train, y_train, X_val, y_val) -> Tuple[str, Pipeline, pd.DataFrame, Dict[str, Pipeline]]:
    results = []
    fitted_models = {}

    for name, base_model in build_models().items():
        pipeline = make_pipeline(base_model)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        row = {"modelo": name, **evaluate_model(pipeline, X_val, y_val)}
        results.append(row)

    metrics = pd.DataFrame(results).sort_values(
        ["recall", "f1", "roc_auc"], ascending=False
    )
    best_name = str(metrics.iloc[0]["modelo"])
    return best_name, fitted_models[best_name], metrics, fitted_models
