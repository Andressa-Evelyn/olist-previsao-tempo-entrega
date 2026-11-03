## Pré-processamento dos Dados

> Projeto: Previsão do tempo de entrega de pedidos — Olist
> Código: [`src/preprocessing.py`](../src/preprocessing.py)

Este documento é o roteiro da parte "Pré-processamento dos Dados" do trabalho.
Cada seção abaixo corresponde a um slide. Os números abaixo vêm da execução no dataset real (`python src/preprocessing.py`).

O enunciado pede **pelo menos 4 itens**; aqui são cobertos **os 5**.

---

## Visão geral do pipeline

```
CSVs (data/raw) ──► engenharia de atributos ──► limpeza ──► split treino/teste
                    (1 linha = 1 pedido)                          │
                                                                  ▼
                              ColumnTransformer ajustado SÓ no treino
                        (imputação → escalonamento → codificação)
```

**Bullets:**
- Unidade de análise: **um pedido entregue** (`order_status == 'delivered'`).
- O **split treino/teste (80/20, `random_state=42`) acontece antes** de ajustar
  imputadores, escalonadores e encoders, evitando *data leakage*.
- Todo o tratamento é um único `ColumnTransformer` (`make_preprocessor()`),
  que pode ser colocado dentro de um `Pipeline` e usado com validação cruzada
  e `GridSearchCV` no Tópico 4 sem vazamento.

---

## 1. Engenharia de atributos

**Regra de ouro:** só entra como feature o que **já é conhecido no momento da
compra**.

| Fica de fora | Motivo |
|---|---|
| `order_approved_at`, `order_delivered_carrier_date` | Acontecem **depois** da compra, já dentro do processo de entrega (vazamento) |
| `order_delivered_customer_date` | É o próprio alvo |
| Reviews (nota/comentário) | Ocorrem após a entrega (já definido no Tópico 1; só na EDA) |

**Features criadas** (agregações em nível de pedido):
- **Itens:** `n_itens`, `n_produtos`, `n_vendedores`, `preco_total`, `frete_total`,
  `razao_frete_preco`, `peso_total_g`, `volume_total_cm3` (comp × alt × larg).
- **Geografia:** coordenadas do cliente e do vendedor (mediana por prefixo de CEP)
  e `distancia_media_km` / `distancia_max_km` (**haversine** vendedor→cliente);
  `mesmo_estado`.
- **Tempo:** `prazo_estimado_dias` (estimada − compra, prometido no checkout),
  `prazo_postagem_dias` (limite de postagem − compra), `compra_dia_semana`,
  `compra_mes`, `compra_hora`, `compra_fim_semana`.
- **Produto/pagamento:** `categoria_produto` (do item de maior valor, traduzida
  para inglês), `tipo_pagamento` predominante, `parcelas_max`.
- **Indicador de ausência:** `sem_info_produto` (peso/dimensões faltavam).

---

## 2. Filtragem e limpeza

| Problema | Tratamento |
|---|---|
| Pedidos não entregues / sem data de entrega | Removidos (sem alvo) |
| Pedidos duplicados | Removidos |
| Coordenadas fora do território brasileiro (lat/lng inválidas) | Descartadas antes de calcular a mediana por CEP |
| Peso, volume e preço ≤ 0 | Viram **ausente** (valor fisicamente impossível) |
| Alvo ≤ 0 (entrega antes da compra) | Linha removida |
| Prazo estimado ≤ 0 | Vira ausente |
| **Outliers extremos do alvo** (> Q3 + 3·IQR) | Linha removida — **1.370 pedidos (≈1,4%)**, limite de **42,58 dias** |

> O corte do outlier usa 3·IQR (extremo), e não o 1,5·IQR clássico, para não
> descartar entregas lentas porém **legítimas** (ex.: regiões distantes), que
> são justamente o que o modelo precisa aprender. O multiplicador é
> configurável (`--iqr-k`).

---

## 3. Tratamento de valores faltantes

| Variável | Estratégia | Justificativa |
|---|---|---|
| Numéricas (peso, volume, distâncias, coordenadas, razão, prazos…) | `SimpleImputer(strategy="median")` | Mediana é robusta às caudas longas |
| Binárias | `SimpleImputer("most_frequent")` | Moda |
| Categóricas | `SimpleImputer("constant", "desconhecido")` | Ausência vira uma categoria própria |
| Peso/dimensão ausentes | Flag `sem_info_produto` **antes** de imputar | Preserva a informação de que o dado faltava |

Os imputadores são ajustados **apenas no treino**. Tabela de % de faltantes por
feature: `data/processed/resumo_features.csv` (coluna gerada na execução).

---

## 4. Normalização / Padronização

| Grupo | Variáveis | Técnica |
|---|---|---|
| Assimétricas (cauda longa) | `preco_total`, `frete_total`, `peso_total_g`, `volume_total_cm3`, `distancia_*_km`, `razao_frete_preco` | **`log1p` + `StandardScaler`** |
| Escala moderada / com outliers | contagens, prazos, parcelas, hora, lat/lng | **`RobustScaler`** (mediana e IQR) |
| Binárias | `mesmo_estado`, `compra_fim_semana`, `sem_info_produto` | Sem escalonamento |

Árvores e ensembles não exigem escalonamento, mas **Ridge/Lasso/KNN exigem** —
como serão comparados no Tópico 4, o mesmo conjunto serve para todos.

---

## 5. Tratamento de variáveis categóricas

| Variável | Cardinalidade | Técnica |
|---|---|---|
| `cliente_estado`, `vendedor_estado`, `tipo_pagamento`, `compra_dia_semana`, `compra_mes` | Baixa (≤ 27) | **`OneHotEncoder`** (categorias raras < 1% agrupadas; categorias novas no teste são toleradas) |
| `categoria_produto` | Alta (~70) | **`TargetEncoder`** com *cross-fitting* (5-fold) — evita explodir dimensionalidade e vazar o alvo no treino |

---

## Resumo das features selecionadas

27 features de entrada → **65 colunas** após a codificação (treino: 76.080 linhas;
teste: 19.020 linhas; 0 valores faltantes restantes).

| Grupo | Qtde | Features |
|---|---|---|
| Numérica (log1p + Standard) | 7 | preco_total, frete_total, peso_total_g, volume_total_cm3, distancia_media_km, distancia_max_km, razao_frete_preco |
| Numérica (Robust) | 11 | n_itens, n_produtos, n_vendedores, prazo_estimado_dias, prazo_postagem_dias, parcelas_max, compra_hora, cliente_lat/lng, vendedor_lat/lng |
| Binária | 3 | mesmo_estado, compra_fim_semana, sem_info_produto |
| Categórica (OneHot) | 5 | cliente_estado, vendedor_estado, tipo_pagamento, compra_dia_semana, compra_mes |
| Categórica (TargetEncoder) | 1 | categoria_produto |

A tabela completa, com descrição de cada feature, é gerada em
`data/processed/resumo_features.csv`.

---

## Onde cada requisito do enunciado é atendido

| Requisito (peso 30%) | Seção |
|---|---|
| Engenharia de atributos | 1 |
| Filtragem e limpeza dos dados | 2 |
| Tratamento de valores faltantes | 3 |
| Normalização/Padronização | 4 |
| Tratamento de variáveis categóricas | 5 |
| Resumo claro das features selecionadas | Resumo das features |

---

## Como reproduzir

```bash
pip install -r requirements.txt
# coloque os CSVs do Kaggle em data/raw/
python src/preprocessing.py
```

Saídas em `data/processed/`: `olist_dataset.csv` (base em nível de pedido, para a
EDA), `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
`preprocessor.joblib` e `resumo_features.csv`.
