## Objetivo

Prever, em dias, quanto tempo levará para um pedido chegar à casa do
cliente, usando features extraídas e combinadas a partir das diversas
tabelas do dataset da Olist. É um problema de **regressão**: cada amostra
representa um pedido entregue e a variável alvo é
`order_delivered_customer_date - order_purchase_timestamp`.

Mais detalhes sobre o contexto do problema, as tabelas utilizadas e o
desafio real envolvido estão em
[`docs/01_definicao_do_problema.md`](docs/01_definicao_do_problema.md).

## Dataset

[Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Os CSVs não ficam versionados no repositório (ver `.gitignore`). Para
reproduzir o projeto localmente:

1. Baixe o dataset do Kaggle (é necessária uma conta).
2. Extraia os arquivos `.csv` para `data/raw/`.

## Estrutura do repositório

```
.
├── docs/                         # roteiro/documentação de cada etapa do trabalho
│   ├── 01_definicao_do_problema.md
│   ├── 02_preprocessamento.md
│   ├── 03_eda.md
│   ├── 04_treinamento.md                 # documentação técnica completa do treinamento
│   ├── 04_treinamento_apresentacao.md    # guia enxuto e simples para os slides/apresentação
│   └── figures/                  # figuras (PNG) e tabelas (CSV) usadas nos slides
│       ├── 03_eda/
│       └── 04_treinamento/
├── data/
│   ├── raw/                      # dataset original (não versionado)
│   └── processed/                # dados tratados, modelos e predições gerados pelo pipeline
├── notebooks/                    # exploração e prototipação
└── src/                          # código do pipeline (pré-processamento, treino, avaliação)
    ├── preprocessing.py          # engenharia de atributos, limpeza e ColumnTransformer
    ├── eda.py                    # análise exploratória: gera figuras e tabelas
    └── train.py                  # treinamento, validação cruzada e ajuste de hiperparâmetros
```

## Como executar o pré-processamento

```bash
pip install -r requirements.txt
python src/preprocessing.py     # lê data/raw/, grava em data/processed/
```

Detalhes das decisões em [`docs/02_preprocessamento.md`](docs/02_preprocessamento.md).


## Como executar a análise exploratória (EDA)

```bash
python src/eda.py               # lê data/raw/, grava em docs/figures/03_eda/
```

Roteiro dos slides e principais achados em [`docs/03_eda.md`](docs/03_eda.md).


## Como executar o treinamento e ajuste de hiperparâmetros

```bash
python src/train.py             # lê data/processed/, grava em docs/figures/04_treinamento/ e data/processed/models/
```

Roteiro dos slides, hiperparâmetros e resultados em [`docs/04_treinamento.md`](docs/04_treinamento.md).
