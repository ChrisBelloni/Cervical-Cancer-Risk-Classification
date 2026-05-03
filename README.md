# Sistema de Apoio ao Diagnóstico de Câncer do Colo do Útero com Machine Learning

Projeto final para o Tech Challenge - Fase 1.

## Objetivo
Construir uma solução inicial de Machine Learning para apoiar a triagem de risco de câncer do colo do útero, usando fatores clínicos e comportamentais para prever a variável `Biopsy`.

## Dataset
- Nome: Cervical Cancer Risk Classification
- Fonte: Kaggle / UCI
- Arquivo esperado: `data/kag_risk_factors_cervical_cancer.csv`
- Variável alvo: `Biopsy`

## Como rodar no Google Colab
1. Abra `notebooks/tech_challenge_final_colab.ipynb`.
2. Execute as células em ordem.
3. O notebook tenta baixar automaticamente o CSV do repositório GitHub.
4. Caso o download falhe, faça upload manual do arquivo CSV na pasta `/content/data/`.

## Como rodar localmente
```bash
pip install -r requirements.txt
python main.py
```

## Entregáveis incluídos
- Notebook final organizado para apresentação.
- Relatório técnico em PDF.
- Roteiro de vídeo de demonstração.
- Código Python executável.
- README e requirements.

## Modelos utilizados
- Regressão Logística
- Árvore de Decisão
- Random Forest
- Gradient Boosting

## Métricas utilizadas
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

No contexto médico, o recall é priorizado, pois falsos negativos são críticos: uma paciente com risco positivo não identificada pode deixar de ser encaminhada para investigação clínica.

## Observação ética
O modelo não substitui diagnóstico médico. Ele deve ser usado apenas como ferramenta de apoio à triagem, mantendo a decisão final com profissionais de saúde.
