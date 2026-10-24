## Definição do Problema 

> Projeto: Previsão do tempo de entrega de pedidos — Olist
> Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
> https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Este documento é o roteiro da parte "Definição do Problema" do trabalho. Cada
seção abaixo corresponde a um slide. Ao final há uma tabela mostrando
explicitamente onde cada requisito pedido no enunciado é atendido.

---

## O que é a Olist

**Bullets:**
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

---

## Estrutura do dataset e tabelas utilizadas

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

---

## Desafio real e impacto para a Olist

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

