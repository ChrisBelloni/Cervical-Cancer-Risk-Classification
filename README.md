# Sistema de Apoio ao Diagnóstico de Câncer do Colo do Útero com Machine Learning

Projeto final para o **Tech Challenge - Fase 1**, com foco em IA aplicada à saúde da mulher.

## Objetivo

Construir uma solução inicial de Machine Learning para apoiar a triagem de risco de câncer do colo do útero, usando fatores clínicos e comportamentais para prever a variável `Biopsy`.

> Importante: o modelo não substitui diagnóstico médico. Ele é uma ferramenta de apoio à triagem e a decisão final deve permanecer com profissionais de saúde.

## Dataset

- Nome: Cervical Cancer Risk Classification
- Fonte: Kaggle / UCI Machine Learning Repository
- Arquivo esperado: `data/kag_risk_factors_cervical_cancer.csv`
- Variável alvo: `Biopsy`


## Estrutura do projeto

```text
.
├── data/
│   └── kag_risk_factors_cervical_cancer.csv
├── notebooks/
│   └── tech_challenge_final_colab.ipynb
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── Dockerfile
├── README.md
├── ROTEIRO_VIDEO_DEMONSTRACAO.md
├── Relatorio_Tecnico_Tech_Challenge_Final.pdf
├── main.py
└── requirements.txt
```

## Como rodar no Google Colab

1. Abra `notebooks/tech_challenge_final_colab.ipynb`.
2. Execute as células em ordem.
3. Caso o download automático do CSV falhe, faça upload manual do arquivo para `data/kag_risk_factors_cervical_cancer.csv`.
4. Ao final, os gráficos, métricas e o modelo serão gerados na pasta `outputs/`.

## Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
python main.py
```

## Como rodar com Docker

```bash
docker build -t cervical-cancer-ml .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" cervical-cancer-ml
```

No Windows PowerShell, use:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/outputs:/app/outputs" cervical-cancer-ml
```

## Técnicas implementadas

### Exploração de dados

- Dimensão e amostras da base.
- Estatísticas descritivas.
- Análise de valores ausentes.
- Distribuição da variável alvo `Biopsy`.
- Histogramas de variáveis relevantes.
- Mapa de correlação e correlação com a variável alvo.

### Pré-processamento

- Conversão de `?` para `NaN`.
- Conversão das colunas para formato numérico.
- Remoção de variáveis diagnósticas auxiliares (`Hinselmann`, `Schiller`, `Citology`) para evitar vazamento de informação.
- Imputação pela mediana dentro do pipeline.
- Padronização com `StandardScaler` dentro do pipeline.
- Separação em treino, validação e teste.

### Modelos avaliados

- Regressão Logística.
- Árvore de Decisão.
- Random Forest.
- Gradient Boosting.

### Métricas

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Matriz de confusão.
- Curva ROC.

No contexto médico, o **recall** é priorizado, pois falsos negativos são críticos: uma paciente com risco positivo não identificada pode deixar de ser encaminhada para investigação clínica.

### Explicabilidade

- Feature importance / coeficientes absolutos.
- SHAP no notebook, quando disponível no ambiente de execução.

## Entregáveis incluídos

- Código-fonte completo.
- Notebook final para demonstração.
- README com instruções.
- Dockerfile.
- Relatório técnico em PDF.
- Roteiro para vídeo de demonstração.

## Saídas esperadas em `outputs/`

Após a execução, o projeto gera:

- `metrics_validation.csv`
- `metrics_test.csv`
- `classification_report_test.txt`
- `confusion_matrix_test.png`
- `roc_curve_test.png`
- `feature_importance.png`
- `feature_importance.csv`
- `best_model.joblib`

## Observação ética

Este projeto tem finalidade acadêmica. O modelo deve ser usado apenas como apoio à triagem e precisa de validação clínica antes de qualquer uso real.
