# Cervical Cancer Risk Classification

Projeto academico de Machine Learning para classificar risco de cancer do colo do utero usando a variavel alvo `Biopsy`.

> Este projeto tem finalidade educacional. O modelo nao substitui diagnostico medico e deve ser entendido apenas como apoio experimental a triagem.

## Objetivo

Construir um pipeline reprodutivel que carrega o dataset, trata valores faltantes, separa treino e teste, treina um modelo de classificacao e gera metricas claras para apresentacao.

## Dataset

O projeto usa o dataset de fatores de risco para cancer cervical, conhecido em bases publicas como Kaggle/UCI Cervical Cancer Risk Factors.

- Arquivo principal esperado: `data/cervical_cancer.csv`
- Variavel alvo: `Biopsy`
- Tipo do problema: classificacao binaria

Algumas colunas do dataset possuem o valor `?` para indicar ausencia de informacao. O codigo converte esses valores para `NaN`, transforma as colunas em numericas e aplica imputacao pela mediana.

As colunas diagnosticas auxiliares `Hinselmann`, `Schiller` e `Citology` sao removidas das features quando presentes, para reduzir risco de vazamento de informacao em relacao ao alvo `Biopsy`.

## Estrutura

```text
Cervical-Cancer-Risk-Classification/
├── data/
│   └── cervical_cancer.csv
├── notebooks/
│   └── tech_challenge_final_colab.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── main.py
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```

## Instalacao

Crie e ative um ambiente virtual, se desejar, e instale as dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principais:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Execucao local

Com o arquivo `data/cervical_cancer.csv` no projeto, execute:

```bash
python main.py
```

Se o arquivo `data/cervical_cancer.csv` nao existir, o codigo tenta usar `data/kag_risk_factors_cervical_cancer.csv` para manter compatibilidade com a versao anterior do projeto. Se nenhum dos arquivos existir, uma mensagem clara indica onde colocar o CSV.

## Execucao no Google Colab

O codigo usa caminhos relativos com `pathlib.Path`, entao pode ser executado no Colab apos enviar ou clonar o projeto.

Fluxo recomendado:

```bash
pip install -r requirements.txt
python main.py
```

No Colab, garanta que o CSV esteja em `data/cervical_cancer.csv` ou `data/kag_risk_factors_cervical_cancer.csv`.

## Modelo

O pipeline principal treina um `RandomForestClassifier` com:

- `random_state=42` para reprodutibilidade
- `class_weight="balanced"` para lidar melhor com desbalanceamento de classes
- imputacao por mediana antes do treino

Random Forest foi escolhido por ser robusto para dados tabulares, lidar bem com relacoes nao lineares e exigir pouca preparacao manual das variaveis.

## Saidas geradas

A pasta `outputs/` e criada automaticamente. A execucao de `python main.py` gera:

- `outputs/metrics.json`: metricas principais em formato JSON
- `outputs/classification_report.txt`: relatorio completo do scikit-learn por classe
- `outputs/confusion_matrix.png`: grafico da matriz de confusao

## Metricas

As metricas calculadas sao:

- Accuracy: proporcao geral de acertos.
- Precision: entre os casos previstos como positivos, quantos eram positivos de fato.
- Recall: entre os positivos reais, quantos o modelo conseguiu identificar.
- F1-score: media harmonica entre precision e recall.
- ROC-AUC: capacidade de separacao entre as classes quando probabilidades estao disponiveis.

Em um problema de apoio a triagem em saude, o recall merece atencao especial, pois falsos negativos podem representar casos de risco nao identificados.

## Docker

Tambem e possivel executar com Docker:

```bash
docker build -t cervical-cancer-ml .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" cervical-cancer-ml
```

No Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/outputs:/app/outputs" cervical-cancer-ml
```

## Observacao academica

Este repositorio foi organizado para entrega academica e demonstracao em video. Os resultados numericos podem variar conforme versao das bibliotecas, particionamento dos dados e alteracoes futuras no dataset.
