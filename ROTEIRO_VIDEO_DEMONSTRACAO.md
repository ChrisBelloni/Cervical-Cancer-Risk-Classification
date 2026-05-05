# Roteiro do vídeo de demonstração - até 15 minutos

## 1. Abertura - 1 min

Apresentar o projeto: sistema de apoio à triagem de risco de câncer do colo do útero usando Machine Learning.
Explicar que o foco é saúde da mulher e que o modelo não substitui diagnóstico médico.

## 2. Problema e dataset - 2 min

Mostrar o dataset Cervical Cancer Risk Classification, a variável alvo `Biopsy` e o objetivo de classificar risco a partir de fatores clínicos e comportamentais.

## 3. Estrutura do repositório - 2 min

Mostrar pastas `data`, `notebooks`, `src`, `outputs`, além de `main.py`, `README.md`, `requirements.txt` e `Dockerfile`.

## 4. Notebook e análise exploratória - 3 min

Executar ou apresentar as células de carregamento, tratamento de valores ausentes, distribuição da variável alvo, estatísticas descritivas e correlação.

## 5. Modelagem - 3 min

Explicar a divisão treino/validação/teste, o pipeline de imputação e padronização, e os modelos testados: Regressão Logística, Árvore de Decisão, Random Forest e Gradient Boosting.

## 6. Métricas e escolha do modelo - 2 min

Mostrar accuracy, precision, recall, F1-score, ROC-AUC, matriz de confusão e curva ROC. Explicar por que o recall é priorizado em saúde.

## 7. Explicabilidade e conclusão - 2 min

Mostrar feature importance ou SHAP. Encerrar destacando limitações, uso prático como apoio à triagem e decisão final do profissional de saúde.
