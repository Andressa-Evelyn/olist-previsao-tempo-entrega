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
teste: 19.020 linhas; 0 valores faltantes restantes). A coluna **Tratamento** indica
o que é aplicado a cada feature depois de criada.

### Itens do pedido (agregados por pedido)

| Feature | O que representa | Origem | Tratamento |
|---|---|---|---|
| `n_itens` | Quantidade de itens | items | mediana + RobustScaler |
| `n_produtos` | Produtos distintos | items | mediana + RobustScaler |
| `n_vendedores` | Vendedores distintos | items | mediana + RobustScaler |
| `preco_total` | Soma dos preços (R$) | items | mediana + log1p + StandardScaler |
| `frete_total` | Soma do frete (R$) | items | mediana + log1p + StandardScaler |
| `razao_frete_preco` | frete_total / preco_total | derivada | mediana + log1p + StandardScaler |
| `peso_total_g` | Soma do peso (g) | products | mediana + log1p + StandardScaler |
| `volume_total_cm3` | Soma do volume (comp × alt × larg) | products | mediana + log1p + StandardScaler |
| `sem_info_produto` | 1 se peso ou dimensões estavam ausentes | derivada | moda, sem escala |
| `categoria_produto` | Categoria do item de maior valor | products + translation | "desconhecido" + TargetEncoder |

### Geografia e logística

| Feature | O que representa | Origem | Tratamento |
|---|---|---|---|
| `distancia_media_km` | Distância haversine média vendedor→cliente | geolocation | mediana + log1p + StandardScaler |
| `distancia_max_km` | Maior distância entre os itens | geolocation | mediana + log1p + StandardScaler |
| `cliente_lat`, `cliente_lng` | Coordenadas do cliente (mediana por CEP) | customers + geolocation | mediana + RobustScaler |
| `vendedor_lat`, `vendedor_lng` | Coordenadas do vendedor principal | sellers + geolocation | mediana + RobustScaler |
| `mesmo_estado` | 1 se vendedor e cliente estão na mesma UF | derivada | moda, sem escala |
| `cliente_estado` | UF do cliente | customers | "desconhecido" + One-Hot |
| `vendedor_estado` | UF do vendedor principal | sellers | "desconhecido" + One-Hot |

### Prazos

| Feature | O que representa | Origem | Tratamento |
|---|---|---|---|
| `prazo_estimado_dias` | Prazo prometido no checkout (estimada − compra) | orders | mediana + RobustScaler |
| `prazo_postagem_dias` | Prazo limite para o vendedor postar (shipping_limit − compra) | items | mediana + RobustScaler |

### Data da compra

| Feature | O que representa | Origem | Tratamento |
|---|---|---|---|
| `compra_hora` | Hora (0–23) | orders | mediana + RobustScaler |
| `compra_dia_semana` | 0 = segunda … 6 = domingo | orders | "desconhecido" + One-Hot |
| `compra_mes` | 1 a 12 | orders | "desconhecido" + One-Hot |
| `compra_fim_semana` | 1 se sábado ou domingo | derivada | moda, sem escala |

### Pagamento

| Feature | O que representa | Origem | Tratamento |
|---|---|---|---|
| `tipo_pagamento` | Forma de pagamento predominante (maior valor) | payments | "desconhecido" + One-Hot |
| `parcelas_max` | Maior número de parcelas | payments | mediana + RobustScaler |

### Contagem por tratamento

| Tratamento | Qtde |
|---|---|
| log1p + StandardScaler | 7 |
| RobustScaler | 11 |
| Binárias, sem escala | 3 |
| One-Hot | 5 |
| TargetEncoder | 1 |
| **Total** | **27** |

### O que ficou de fora

- **Alvo:** `tempo_entrega_dias`, a variável prevista.
- **Identificadores:** `order_id` e `data_compra` (só rastreio, não entram no modelo).
- **Informações posteriores à compra** (risco de vazamento): `order_approved_at`,
  data de postagem à transportadora e reviews.

A versão completa, com o % de faltantes de cada feature, é gerada em
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
