"""Funcoes de carregamento e preprocessamento dos dados."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


AUXILIARY_TARGET_COLUMNS = ["Hinselmann", "Schiller", "Citology"]


def load_data(data_path: Path | str) -> pd.DataFrame:
    """Carrega o CSV do projeto e retorna um DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            "Dataset nao encontrado. Coloque o arquivo em "
            f"'{path.as_posix()}' e execute novamente."
        )
    return pd.read_csv(path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove espacos extras dos nomes de colunas sem alterar o significado."""
    clean_df = df.copy()
    clean_df.columns = [str(column).strip() for column in clean_df.columns]
    return clean_df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Converte marcadores de ausencia para NaN e colunas para numerico."""
    clean_df = df.replace("?", np.nan).copy()
    for column in clean_df.columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
    return clean_df


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "Biopsy",
    drop_auxiliary_targets: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Trata dados faltantes e separa X e y."""
    clean_df = clean_missing_values(df)

    if target_column not in clean_df.columns:
        raise ValueError(f"A coluna alvo '{target_column}' nao foi encontrada no dataset.")

    columns_to_drop = [target_column]
    if drop_auxiliary_targets:
        columns_to_drop.extend(
            column for column in AUXILIARY_TARGET_COLUMNS if column in clean_df.columns
        )

    X = clean_df.drop(columns=columns_to_drop)
    y = clean_df[target_column]

    if y.isna().any():
        raise ValueError("A coluna alvo contem valores ausentes e precisa ser revisada.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X_imputed, y.astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = False,
):
    """Divide os dados em treino e teste, com estratificacao quando possivel."""
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if not scale_features:
        return X_train, X_test, y_train, y_test

    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica StandardScaler usando apenas os dados de treino para ajuste."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled
