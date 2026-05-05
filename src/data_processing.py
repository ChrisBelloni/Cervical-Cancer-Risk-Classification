"""Funções de carregamento, limpeza e preparação dos dados."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Biopsy"
AUXILIARY_TARGETS = ["Hinselmann", "Schiller", "Citology"]
DATA_PATH = Path("data/kag_risk_factors_cervical_cancer.csv")
RAW_URL = (
    "https://raw.githubusercontent.com/ChrisBelloni/"
    "Cervical-Cancer-Risk-Classification/main/data/"
    "kag_risk_factors_cervical_cancer.csv"
)


def load_data(data_path: Path = DATA_PATH, raw_url: str = RAW_URL) -> pd.DataFrame:
    """Carrega o dataset localmente ou tenta baixar do repositório GitHub."""
    if data_path.exists():
        return pd.read_csv(data_path)

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(raw_url)
        df.to_csv(data_path, index=False)
        return df
    except Exception as exc:
        raise FileNotFoundError(
            "Dataset não encontrado. Adicione o arquivo "
            f"'{data_path.as_posix()}' ou execute o notebook no Colab com upload manual."
        ) from exc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Converte '?' para NaN e força colunas para formato numérico."""
    df_clean = df.replace("?", np.nan).copy()
    for column in df_clean.columns:
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
    return df_clean


def prepare_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo, removendo variáveis diagnósticas auxiliares."""
    if TARGET not in df_clean.columns:
        raise ValueError(f"A coluna alvo '{TARGET}' não foi encontrada no dataset.")

    columns_to_drop = [TARGET] + [c for c in AUXILIARY_TARGETS if c in df_clean.columns]
    X = df_clean.drop(columns=columns_to_drop)
    y = df_clean[TARGET].astype(int)
    return X, y


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    """Divide a base em treino (60%), validação (20%) e teste (20%)."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval
