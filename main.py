import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import joblib

DATA_PATH = 'data/kag_risk_factors_cervical_cancer.csv'
RAW_URL = 'https://raw.githubusercontent.com/ChrisBelloni/Cervical-Cancer-Risk-Classification/main/data/kag_risk_factors_cervical_cancer.csv'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(RAW_URL)
        os.makedirs('data', exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df


def prepare_data(df):
    df = df.replace('?', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    target = 'Biopsy'
    auxiliary_targets = ['Hinselmann', 'Schiller', 'Citology']
    X = df.drop(columns=[target] + [c for c in auxiliary_targets if c in df.columns])
    y = df[target].astype(int)
    return X, y


def main():
    df = load_data()
    X, y = prepare_data(df)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    fitted_models = {}
    reports = []

    for name, model in models.items():
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        pred = pipe.predict(X_val)
        proba = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, 'predict_proba') else pred
        results.append({
            'modelo': name,
            'accuracy': accuracy_score(y_val, pred),
            'precision': precision_score(y_val, pred, zero_division=0),
            'recall': recall_score(y_val, pred, zero_division=0),
            'f1': f1_score(y_val, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, proba)
        })
        reports.append(f'\n===== {name} - Validation =====\n')
        reports.append(classification_report(y_val, pred, zero_division=0))

    metrics = pd.DataFrame(results).sort_values(['recall', 'f1', 'roc_auc'], ascending=False)
    metrics.to_csv(os.path.join(OUTPUT_DIR, 'metrics_validation.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'classification_reports_validation.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(reports))

    best_name = metrics.iloc[0]['modelo']
    best_model = fitted_models[best_name]
    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = pd.DataFrame([{
        'modelo_escolhido': best_name,
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'f1': f1_score(y_test, test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, test_proba)
    }])
    test_metrics.to_csv(os.path.join(OUTPUT_DIR, 'metrics_test.csv'), index=False)
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.joblib'))

    ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
    plt.title(f'Matriz de Confusão - Teste ({best_name})')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png'), bbox_inches='tight')
    plt.close()

    RocCurveDisplay.from_predictions(y_test, test_proba)
    plt.title(f'Curva ROC - Teste ({best_name})')
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_test.png'), bbox_inches='tight')
    plt.close()

    print('Projeto executado com sucesso.')
    print(metrics)
    print(test_metrics)


if __name__ == '__main__':
    main()
