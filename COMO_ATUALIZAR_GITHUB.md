# Como atualizar o GitHub com esta versão corrigida

1. Baixe e extraia o arquivo `github_100_corrigido.zip`.
2. Copie todos os arquivos extraídos para a raiz do seu repositório local `Cervical-Cancer-Risk-Classification`.
3. Confirme que o dataset esteja em `data/kag_risk_factors_cervical_cancer.csv` ou faça upload manual no Colab.
4. Execute localmente:

```bash
pip install -r requirements.txt
python main.py
```

5. Faça commit e push:

```bash
git add .
git commit -m "Corrige projeto final do Tech Challenge Fase 1"
git push origin main
```

6. Depois de gravar o vídeo, coloque o link no README.
