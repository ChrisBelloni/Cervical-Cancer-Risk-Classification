import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'kag_risk_factors_cervical_cancer.csv')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Load Kaggle CSV (missing values in this dataset are encoded as '?')
df = pd.read_csv(DATA_PATH)
df = df.replace('?', np.nan)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Target
TARGET = 'Biopsy'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# Save profile summaries
summary = {
    'rows': df.shape[0],
    'columns': df.shape[1],
    'target_negative': int((y == 0).sum()),
    'target_positive': int((y == 1).sum()),
    'missing_cells': int(df.isna().sum().sum()),
    'missing_columns': int((df.isna().sum() > 0).sum())
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'dataset_summary.csv'), index=False)
df.isna().sum().sort_values(ascending=False).to_csv(os.path.join(OUT_DIR, 'missing_values.csv'))
df.describe().T.to_csv(os.path.join(OUT_DIR, 'descriptive_statistics.csv'))

# EDA figures
plt.figure(figsize=(6,4))
y.value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição da variável alvo - Biopsy')
plt.xlabel('Biopsy (0 = negativo, 1 = positivo)')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'target_distribution.png'), dpi=180)
plt.close()

plt.figure(figsize=(8,4.5))
df['Age'].dropna().hist(bins=25)
plt.title('Distribuição de idade das pacientes')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'age_distribution.png'), dpi=180)
plt.close()

miss = df.isna().sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(9,5))
miss.sort_values().plot(kind='barh')
plt.title('Top 15 variáveis com valores ausentes')
plt.xlabel('Quantidade de valores ausentes')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'missing_values_top15.png'), dpi=180)
plt.close()

corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(12)
plt.figure(figsize=(9,5))
corr.sort_values().plot(kind='barh')
plt.title('Variáveis com maior correlação absoluta com Biopsy')
plt.xlabel('|Correlação|')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_target_top12.png'), dpi=180)
plt.close()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

numeric_features = X.columns.tolist()
preprocessor_scaled = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features)
])
preprocessor_tree = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features)
])

models = {
    'Regressao_Logistica': Pipeline([
        ('prep', preprocessor_scaled),
        ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ]),
    'Random_Forest': Pipeline([
        ('prep', preprocessor_tree),
        ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
    ]),
    'Gradient_Boosting': Pipeline([
        ('prep', preprocessor_tree),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
}

metrics = []
reports = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics.append({
        'Modelo': name.replace('_', ' '),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
    })
    reports.append(f"\n{name.replace('_',' ')}\n" + classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
    disp.plot(values_format='d')
    plt.title(f'Matriz de Confusão - {name.replace("_", " ")}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'cm_{name}.png'), dpi=180)
    plt.close()

# Metrics table
metrics_df = pd.DataFrame(metrics).sort_values(['Recall','F1-score'], ascending=False)
metrics_df.to_csv(os.path.join(OUT_DIR, 'metrics.csv'), index=False)
with open(os.path.join(OUT_DIR, 'classification_reports.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(reports))

# ROC Curve combined
plt.figure(figsize=(7,5))
ax = plt.gca()
for name, pipe in models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name.replace('_', ' '))
plt.title('Curvas ROC dos modelos')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'roc_curve.png'), dpi=180)
plt.close()

# Feature importance: Random Forest impurity and permutation on best recall model
rf = models['Random_Forest'].named_steps['model']
rf_importance = pd.Series(rf.feature_importances_, index=numeric_features).sort_values(ascending=False)
rf_importance.to_csv(os.path.join(OUT_DIR, 'feature_importance_random_forest.csv'))
plt.figure(figsize=(9,6))
rf_importance.head(12).sort_values().plot(kind='barh')
plt.title('Top 12 Importâncias - Random Forest')
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_random_forest.png'), dpi=180)
plt.close()

best_name = metrics_df.iloc[0]['Modelo'].replace(' ', '_')
best_pipe = models[best_name]
perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=15, random_state=42, scoring='recall')
perm_df = pd.DataFrame({'feature': numeric_features, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std})
perm_df = perm_df.sort_values('importance_mean', ascending=False)
perm_df.to_csv(os.path.join(OUT_DIR, 'permutation_importance_recall.csv'), index=False)
plt.figure(figsize=(9,6))
plot_df = perm_df.head(12).sort_values('importance_mean')
plt.barh(plot_df['feature'], plot_df['importance_mean'])
plt.title(f'Importância por Permutação - {best_name.replace("_", " ")} (Recall)')
plt.xlabel('Redução média no recall')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'permutation_importance_recall.png'), dpi=180)
plt.close()

print(metrics_df)
print(summary)
