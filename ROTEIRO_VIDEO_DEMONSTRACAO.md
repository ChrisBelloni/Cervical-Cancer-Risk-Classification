# Roteiro do vídeo de demonstração - até 15 minutos

## 1. Abertura - 1 min
Apresentar o projeto: sistema de apoio ao diagnóstico de câncer do colo do útero com Machine Learning, voltado à saúde da mulher.

## 2. Problema e objetivo - 2 min
Explicar que o objetivo é apoiar a triagem inicial de risco, prevendo a variável `Biopsy` a partir de dados clínicos e comportamentais. Reforçar que o modelo não substitui o médico.

## 3. Dataset - 2 min
Mostrar a origem pública do dataset, o arquivo CSV e a variável alvo. Explicar que valores `?` foram tratados como ausentes.

## 4. Análise exploratória - 3 min
Mostrar:
- dimensões da base;
- distribuição da variável alvo;
- valores ausentes;
- estatísticas descritivas;
- correlação com `Biopsy`.

## 5. Pré-processamento - 2 min
Explicar:
- conversão para numérico;
- imputação pela mediana;
- padronização;
- remoção de variáveis diagnósticas auxiliares para evitar vazamento;
- separação treino, validação e teste.

## 6. Modelagem - 2 min
Mostrar os modelos:
- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- Gradient Boosting.

Explicar que o recall foi priorizado por causa do risco de falsos negativos em saúde.

## 7. Resultados e interpretabilidade - 2 min
Mostrar tabela de métricas, matriz de confusão, curva ROC, feature importance e/ou SHAP.

## 8. Conclusão - 1 min
Finalizar explicando que o sistema pode apoiar a triagem, mas precisa de validação clínica e decisão final médica.
