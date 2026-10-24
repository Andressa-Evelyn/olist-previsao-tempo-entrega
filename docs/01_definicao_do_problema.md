# 1. Definição do Problema (Peso 10%)

> Projeto: Previsão do tempo de entrega de pedidos — Olist
> Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
> https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Este documento é o roteiro da parte "Definição do Problema" do trabalho. Cada
seção abaixo corresponde a um slide. Ao final há uma tabela mostrando
explicitamente onde cada requisito pedido no enunciado é atendido.

---

## Slide 1 — O que é a Olist

**Bullets para o slide:**
- Olist é uma **startup brasileira de marketplace** (fundada em 2015, sede em
  Curitiba/PR).
- Ela **não vende produtos próprios**: conecta pequenos e médios lojistas de
  todo o Brasil aos grandes marketplaces (Mercado Livre, Amazon, Americanas,
  Shopee etc.) através de uma única plataforma.
- A Olist centraliza para o lojista: cadastro de produtos, gestão de pedidos,
  pagamentos e, principalmente, **a ponte logística** entre o vendedor e o
  cliente final.
- O dataset público cobre **~100 mil pedidos** feitos entre **2016 e 2018**,
  em múltiplos marketplaces, com informações de status do pedido, preço,
  pagamento, frete, avaliação do cliente e localização geográfica de
  clientes e vendedores.

**Fala sugerida:**
"A Olist é uma espécie de 'operadora logística e comercial' para pequenos
lojistas brasileiros: ela não tem produto próprio, ela dá a esses lojistas
acesso aos grandes marketplaces e cuida de todo o processo entre a compra e
a entrega. Como ela está no meio dessa cadeia — entre o vendedor, a
transportadora e o cliente — ela carrega a responsabilidade (e o risco) de
cumprir o prazo prometido ao cliente, mesmo não controlando diretamente
quem fabrica ou quem transporta o produto."

---

## Slide 2 — O problema de regressão

**O que representa cada amostra (linha do dataset):**
- Cada amostra é **um pedido (`order_id`)** já **entregue** ao cliente
  (`order_status == 'delivered'`), com todas as datas do ciclo de vida
  preenchidas.
- Um pedido pode conter mais de um item/produto (tabela
  `olist_order_items_dataset`); quando isso acontece, as features desses
  itens são **agregadas em nível de pedido** (ex.: número de itens, peso
  total, valor total, número de vendedores distintos), pois a entrega
  acontece para o pedido como um todo, não por item isoladamente.

**Variável alvo (target):**
- **Tempo de entrega, em dias**, calculado como:

  ```
  tempo_entrega_dias = order_delivered_customer_date - order_purchase_timestamp
  ```

- Ou seja: quantos dias se passam **entre o momento em que o cliente faz a
  compra e o momento em que o produto chega fisicamente à casa dele.**
- É um problema de **regressão** porque o alvo é uma variável numérica
  contínua (dias, podendo usar frações/horas), e não uma categoria.
- *(Observação para o time)*: também é possível calcular, como variável de
  apoio, `atraso = order_delivered_customer_date - order_estimated_delivery_date`,
  útil na EDA para entender atrasos, mas **o alvo do modelo é o tempo de
  entrega em si**, conforme pedido no enunciado.

**Fala sugerida:**
"Definimos que cada linha do nosso dataset final representa um pedido
entregue. A variável que queremos prever — o nosso 'y' — é quantos dias se
passam entre a compra e a entrega na casa do cliente. É uma regressão
porque esse número pode assumir qualquer valor contínuo, diferente de, por
exemplo, prever se o pedido vai atrasar ou não, que seria classificação."

---

## Slide 3 — Estrutura do dataset e tabelas utilizadas

O dataset é composto por **9 tabelas relacionais**, conectadas por chaves
como `order_id`, `customer_id`, `product_id`, `seller_id` e código postal
(`zip_code_prefix`):

| Tabela | Papel no problema | Por que usar |
|---|---|---|
| `olist_orders_dataset` | **Tabela central.** Traz `order_id`, status e todas as datas (compra, aprovação, envio à transportadora, entrega, prazo estimado). | É dela que extraímos a variável alvo e o recorte temporal do pedido. |
| `olist_order_items_dataset` | Itens de cada pedido: `product_id`, `seller_id`, preço, frete, prazo limite de envio. | Permite calcular nº de itens, valor total, frete total e nº de vendedores por pedido — fatores que afetam logística. |
| `olist_customers_dataset` | Localização do cliente (`customer_zip_code_prefix`, cidade, estado). | Necessária para calcular a distância vendedor→cliente, forte candidata a variável explicativa do tempo de entrega. |
| `olist_sellers_dataset` | Localização do vendedor (zip code, cidade, estado). | Mesma razão acima: origem do envio. |
| `olist_geolocation_dataset` | Latitude/longitude por prefixo de CEP. | Permite transformar CEP em coordenadas e calcular distância geográfica real (haversine) entre vendedor e cliente. |
| `olist_products_dataset` | Categoria, peso e dimensões do produto. | Peso/volume impactam modal de transporte e prazo. |
| `olist_order_payments_dataset` | Tipo de pagamento, parcelas, valor pago. | Forma de pagamento pode se relacionar indiretamente ao tempo de processamento do pedido. |
| `olist_order_reviews_dataset` | Nota e comentário da avaliação pós-entrega. | Uso **apenas na EDA** (ex.: correlação entre atraso e nota), não como feature de entrada, pois a review ocorre **depois** da entrega (vazamento de dados/*data leakage*). |
| `product_category_name_translation` | Tradução de categoria PT→EN. | Apoio para padronizar/legibilidade das categorias de produto. |

**Relacionamento entre as tabelas (chaves):**
```
orders (order_id) ──< order_items >── products (product_id)
   │                      │
   │                      └── sellers (seller_id) ── geolocation (seller_zip_code_prefix)
   │
   ├── customers (customer_id) ── geolocation (customer_zip_code_prefix)
   ├── order_payments (order_id)
   └── order_reviews (order_id)   [somente para EDA — não entra como feature]
```

**Fala sugerida:**
"O dataset é dividido em 9 tabelas que se conectam como um banco relacional.
A tabela `orders` é o centro de tudo, porque é dela que vem o nosso alvo.
As demais tabelas entram para enriquecer o pedido com informações de
distância geográfica, características do produto, forma de pagamento e
quantidade de itens — tudo que pode influenciar quanto tempo uma entrega
demora. A tabela de reviews nós deixamos de fora do modelo preditivo porque
ela só existe depois que o pedido já foi entregue — usá-la como feature
seria vazamento de informação do futuro."

---

## Slide 4 — Desafio real e impacto para a Olist

**O desafio:**
- A Olist **não controla diretamente a logística de ponta a ponta**: depende
  de milhares de vendedores (postagem) e de transportadoras terceirizadas
  (entrega), espalhados por um país continental, com infraestrutura de
  transporte muito desigual entre regiões (ex.: Sudeste vs. Norte/Nordeste).
- Isso cria **alta variância no tempo de entrega**, difícil de estimar com
  uma regra fixa — exatamente o tipo de problema onde aprendizado de
  máquina agrega valor sobre uma estimativa genérica.

**Impacto de prever bem (ou mal) o prazo:**
- **Experiência do cliente**: a data estimada de entrega é uma promessa feita
  no checkout; estimativas ruins geram expectativa quebrada, cancelamentos e
  avaliações negativas (há evidência no próprio dataset de correlação entre
  atraso e nota de review baixa).
- **Planejamento operacional**: prever melhor o prazo ajuda a Olist a
  negociar SLAs com transportadoras, alocar estoque/centros de distribuição
  e priorizar vendedores/regiões problemáticas.
- **Vantagem competitiva**: em marketplace, prazo de entrega é um dos
  principais fatores de decisão de compra — uma estimativa de prazo mais
  precisa e confiável (nem otimista demais, nem exageradamente conservadora)
  é diferencial frente a outros marketplaces.

**Fala sugerida:**
"O grande desafio aqui é que a Olist é uma intermediária: ela promete um
prazo de entrega sem controlar totalmente nem quem despacha, nem quem
transporta o produto. Isso faz o tempo de entrega variar bastante de pedido
para pedido. Prever esse prazo com mais precisão tem impacto direto na
satisfação do cliente — que é o que vemos nas notas de avaliação — e também
na forma como a empresa planeja sua operação logística."

---

## Checklist — onde cada requisito do enunciado é atendido

| Requisito do enunciado | Onde está neste documento |
|---|---|
| Apresentar o contexto da empresa Olist | **Slide 1** — modelo de negócio, o que é a Olist, escopo do dataset |
| Descrever claramente o problema de regressão (o que é cada amostra, qual a variável alvo) | **Slide 2** — amostra = pedido entregue; alvo = `order_delivered_customer_date − order_purchase_timestamp` (dias) |
| Identificar as principais tabelas utilizadas e justificar a escolha | **Slide 3** — tabela com as 9 tabelas, papel de cada uma e justificativa, + diagrama de relacionamento |
| Apontar o desafio real representado pelo problema e seu impacto | **Slide 4** — variância logística, falta de controle ponta a ponta, impacto em CX e operação |

Todos os 4 itens obrigatórios (★) estão contemplados.
