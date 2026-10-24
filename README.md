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
│   └── 01_definicao_do_problema.md
├── data/
│   ├── raw/                      # dataset original (não versionado)
│   └── processed/                # dados tratados/gerados pelo pipeline
├── notebooks/                    # exploração e prototipação
└── src/                          # código do pipeline (pré-processamento, treino, avaliação)
```

