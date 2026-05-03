# Sistema de Apoio ao Diagnóstico de Câncer do Colo do Útero com Machine Learning

Projeto desenvolvido para o Tech Challenge - Fase 1, utilizando a base pública **Cervical Cancer Risk Classification** do Kaggle.

## Objetivo
Construir uma solução inicial de Machine Learning para apoiar a triagem de risco de câncer do colo do útero, usando fatores clínicos e comportamentais para prever a variável `Biopsy`.

## Estrutura

```text
projeto_kaggle_cancer_colo_utero/
├── data/
│   └── kag_risk_factors_cervical_cancer.csv
├── outputs/
│   ├── metrics.csv
│   ├── classification_reports.txt
│   ├── *.png
├── run_project.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Como executar localmente

```bash
pip install -r requirements.txt
python run_project.py
```

Os resultados serão salvos na pasta `outputs/`.

## Modelos utilizados

- Regressão Logística
- Random Forest

git config --global user.name "Christinne Belloni"
git config --global user.email "seu_email@exemplo.com"
- Gradient Boosting

## Métricas utilizadas

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

No contexto médico, o recall foi priorizado porque falsos negativos são críticos: uma paciente com risco positivo não identificada pode deixar de ser priorizada para investigação clínica.

## Observação ética

O modelo não substitui diagnóstico médico. Ele deve ser interpretado apenas como ferramenta de apoio à triagem e decisão clínica.
