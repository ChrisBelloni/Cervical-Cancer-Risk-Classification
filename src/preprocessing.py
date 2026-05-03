import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Biopsy"

def load_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset e trata símbolos comuns de valores ausentes."""
    df = pd.read_csv(path)
    df = df.replace("?", pd.NA)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def prepare_data(df: pd.DataFrame):
    """Prepara X e y para modelagem."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"A coluna alvo '{TARGET_COLUMN}' não foi encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.copy()
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X = X.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, X.columns.tolist(), imputer, scaler

def split_data(X, y):
    """Separa dados em treino e teste com estratificação."""
    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
