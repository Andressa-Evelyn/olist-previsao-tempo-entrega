"""Pré-processamento dos dados (Tópico 2) — Olist: previsão do tempo de entrega.

Fluxo:
    1. load_raw()               -> lê os CSVs de data/raw/
    2. build_order_dataset()    -> engenharia de atributos (1 linha = 1 pedido entregue)
    3. clean_dataset()          -> filtragem e limpeza (ruídos e inconsistências)
    4. split treino/teste       -> ANTES de ajustar imputadores/escalonadores
    5. make_preprocessor()      -> imputação, escalonamento e codificação (ColumnTransformer)
    6. salva dataset, treino/teste transformados, preprocessor e resumo das features

Uso:
    python src/preprocessing.py                  # lê data/raw, grava em data/processed
    python src/preprocessing.py --test-size 0.2 --seed 42

Decisão importante (data leakage): só entram como feature informações conhecidas
no momento da compra. `order_approved_at`, `order_delivered_carrier_date` e as
reviews ficam de fora, pois ocorrem depois da compra (ver docs/01).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "tempo_entrega_dias"
ID_COLS = ["order_id", "data_compra"]

# Limites aproximados do território brasileiro (descarta coordenadas inválidas)
BR_LAT = (-34.0, 6.0)
BR_LNG = (-74.0, -34.0)

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

# --------------------------------------------------------------------------- #
# Catálogo de features: grupo de tratamento -> {coluna: descrição}
# --------------------------------------------------------------------------- #
# Numéricas assimétricas (cauda longa): imputa mediana -> log1p -> StandardScaler
NUM_LOG = {
    "preco_total": "Soma dos preços dos itens do pedido (R$)",
    "frete_total": "Soma do frete dos itens do pedido (R$)",
    "peso_total_g": "Soma do peso dos produtos (g)",
    "volume_total_cm3": "Soma do volume (comp x alt x larg) dos produtos (cm³)",
    "distancia_media_km": "Distância haversine média vendedor→cliente (km)",
    "distancia_max_km": "Maior distância vendedor→cliente entre os itens (km)",
    "razao_frete_preco": "frete_total / preco_total",
}
# Numéricas com escala/outliers moderados: imputa mediana -> RobustScaler
NUM_ROBUST = {
    "n_itens": "Quantidade de itens no pedido",
    "n_produtos": "Quantidade de produtos distintos",
    "n_vendedores": "Quantidade de vendedores distintos",
    "prazo_estimado_dias": "Prazo prometido no checkout (estimada − compra), em dias",
    "prazo_postagem_dias": "Prazo limite de postagem do vendedor (shipping_limit − compra), em dias",
    "parcelas_max": "Maior número de parcelas do pagamento",
    "compra_hora": "Hora da compra (0–23)",
    "cliente_lat": "Latitude do cliente (mediana do prefixo de CEP)",
    "cliente_lng": "Longitude do cliente (mediana do prefixo de CEP)",
    "vendedor_lat": "Latitude do vendedor principal",
    "vendedor_lng": "Longitude do vendedor principal",
}
# Binárias: imputa moda, sem escalonamento
BINARY = {
    "mesmo_estado": "1 se vendedor principal e cliente estão no mesmo estado",
    "compra_fim_semana": "1 se a compra foi em sábado/domingo",
    "sem_info_produto": "1 se peso/dimensões do produto estavam ausentes (antes da imputação)",
}
# Categóricas de baixa cardinalidade: imputa "desconhecido" -> OneHotEncoder
CAT_ONEHOT = {
    "cliente_estado": "UF do cliente",
    "vendedor_estado": "UF do vendedor principal",
    "tipo_pagamento": "Forma de pagamento predominante",
    "compra_dia_semana": "Dia da semana da compra (0=seg … 6=dom)",
    "compra_mes": "Mês da compra (1–12)",
}
# Categórica de alta cardinalidade (~70 níveis): imputa -> TargetEncoder (cross-fitting)
CAT_TARGET = {
    "categoria_produto": "Categoria do item de maior valor do pedido",
}

FEATURE_GROUPS = {
    "numérica (log1p + StandardScaler)": NUM_LOG,
    "numérica (RobustScaler)": NUM_ROBUST,
    "binária (passthrough)": BINARY,
    "categórica (OneHotEncoder)": CAT_ONEHOT,
    "categórica (TargetEncoder)": CAT_TARGET,
}
ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


# --------------------------------------------------------------------------- #
# 1. Leitura
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em {raw_dir}: {missing}. "
            "Baixe o dataset do Kaggle e extraia os CSVs em data/raw/ (ver README)."
        )
    return {k: pd.read_csv(raw_dir / f) for k, f in RAW_FILES.items()}


# --------------------------------------------------------------------------- #
# 2. Engenharia de atributos
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Distância (km) entre dois pontos (vetorizado)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def build_geo_table(geo: pd.DataFrame) -> pd.DataFrame:
    """Uma coordenada (mediana) por prefixo de CEP, descartando pontos fora do Brasil."""
    geo = geo.rename(
        columns={
            "geolocation_zip_code_prefix": "zip",
            "geolocation_lat": "lat",
            "geolocation_lng": "lng",
        }
    )
    ok = geo["lat"].between(*BR_LAT) & geo["lng"].between(*BR_LNG)
    return geo[ok].groupby("zip", as_index=False)[["lat", "lng"]].median()


def build_order_dataset(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Monta 1 linha por pedido entregue, com target e features agregadas."""
    # ---- pedidos entregues com datas válidas + target --------------------- #
    orders = t["orders"].copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    orders = orders[orders["order_status"] == "delivered"]
    orders = orders.dropna(subset=date_cols).drop_duplicates("order_id")

    compra = orders["order_purchase_timestamp"]
    orders[TARGET] = (
        orders["order_delivered_customer_date"] - compra
    ).dt.total_seconds() / 86400
    orders["prazo_estimado_dias"] = (
        orders["order_estimated_delivery_date"] - compra
    ).dt.total_seconds() / 86400
    orders["data_compra"] = compra
    orders["compra_dia_semana"] = compra.dt.day_of_week
    orders["compra_mes"] = compra.dt.month
    orders["compra_hora"] = compra.dt.hour
    orders["compra_fim_semana"] = (compra.dt.day_of_week >= 5).astype(int)
    orders = orders[
        [
            "order_id",
            "customer_id",
            "data_compra",
            TARGET,
            "prazo_estimado_dias",
            "compra_dia_semana",
            "compra_mes",
            "compra_hora",
            "compra_fim_semana",
        ]
    ]

    # ---- cliente (UF + coordenadas) -------------------------------------- #
    geo = build_geo_table(t["geolocation"])
    cust = t["customers"][
        ["customer_id", "customer_zip_code_prefix", "customer_state"]
    ].merge(
        geo.rename(columns={"zip": "customer_zip_code_prefix", "lat": "cliente_lat", "lng": "cliente_lng"}),
        on="customer_zip_code_prefix",
        how="left",
    )
    cust = cust.rename(columns={"customer_state": "cliente_estado"}).drop(
        columns="customer_zip_code_prefix"
    )
    orders = orders.merge(cust, on="customer_id", how="left")

    # ---- itens: produto, vendedor, distância ------------------------------ #
    prod = t["products"].copy()
    prod["volume_cm3"] = (
        prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )
    # valores fisicamente impossíveis -> ausente
    prod["product_weight_g"] = prod["product_weight_g"].where(prod["product_weight_g"] > 0)
    prod["volume_cm3"] = prod["volume_cm3"].where(prod["volume_cm3"] > 0)
    trans = t["translation"]
    prod = prod.merge(trans, on="product_category_name", how="left")
    prod["categoria_produto"] = (
        prod["product_category_name_english"]
        .fillna(prod["product_category_name"])
        .fillna("desconhecido")
    )
    prod = prod[["product_id", "product_weight_g", "volume_cm3", "categoria_produto"]]

    sellers = t["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]].merge(
        geo.rename(columns={"zip": "seller_zip_code_prefix", "lat": "vendedor_lat", "lng": "vendedor_lng"}),
        on="seller_zip_code_prefix",
        how="left",
    )
    sellers = sellers.rename(columns={"seller_state": "vendedor_estado"}).drop(
        columns="seller_zip_code_prefix"
    )

    items = t["items"][t["items"]["order_id"].isin(orders["order_id"])].copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
    items = (
        items.merge(prod, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(
            orders[["order_id", "data_compra", "cliente_lat", "cliente_lng", "cliente_estado"]],
            on="order_id",
            how="left",
        )
    )
    items["distancia_km"] = haversine_km(
        items["vendedor_lat"], items["vendedor_lng"], items["cliente_lat"], items["cliente_lng"]
    )
    items["prazo_postagem_dias"] = (
        items["shipping_limit_date"] - items["data_compra"]
    ).dt.total_seconds() / 86400

    # agregação por pedido
    agg = items.groupby("order_id").agg(
        n_itens=("order_item_id", "count"),
        n_produtos=("product_id", "nunique"),
        n_vendedores=("seller_id", "nunique"),
        preco_total=("price", "sum"),
        frete_total=("freight_value", "sum"),
        peso_total_g=("product_weight_g", lambda s: s.sum(min_count=1)),
        volume_total_cm3=("volume_cm3", lambda s: s.sum(min_count=1)),
        distancia_media_km=("distancia_km", "mean"),
        distancia_max_km=("distancia_km", "max"),
        prazo_postagem_dias=("prazo_postagem_dias", "max"),
        algum_sem_info=("product_weight_g", lambda s: int(s.isna().any())),
    )
    agg = agg.rename(columns={"algum_sem_info": "sem_info_produto"})

    # item de maior valor = "item principal" (categoria, vendedor)
    principal = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[
            ["order_id", "categoria_produto", "vendedor_estado", "vendedor_lat", "vendedor_lng"]
        ]
    )

    # ---- pagamentos ------------------------------------------------------- #
    pay = t["payments"]
    pay_main = (
        pay.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "tipo_pagamento"})
    )
    pay_agg = (
        pay.groupby("order_id", as_index=False)["payment_installments"]
        .max()
        .rename(columns={"payment_installments": "parcelas_max"})
    )

    df = (
        orders.merge(agg, on="order_id", how="inner")
        .merge(principal, on="order_id", how="left")
        .merge(pay_main, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
    )

    # ---- features derivadas ---------------------------------------------- #
    df["preco_total"] = df["preco_total"].where(df["preco_total"] > 0)
    df["razao_frete_preco"] = df["frete_total"] / df["preco_total"]
    df["mesmo_estado"] = (df["vendedor_estado"] == df["cliente_estado"]).astype(int)
    df["tipo_pagamento"] = df["tipo_pagamento"].replace("not_defined", np.nan)

    cols = ID_COLS + [TARGET] + ALL_FEATURES
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Filtragem e limpeza
# --------------------------------------------------------------------------- #
def clean_dataset(df: pd.DataFrame, iqr_k: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Remove ruídos/inconsistências. Retorna (df limpo, relatório com contagens)."""
    report = {"linhas_iniciais": len(df)}
    df = df.drop_duplicates("order_id")
    report["duplicados_removidos"] = report["linhas_iniciais"] - len(df)

    # target impossível (entrega antes/no instante da compra)
    n = len(df)
    df = df[df[TARGET] > 0]
    report["target_nao_positivo"] = n - len(df)

    # prazo estimado / postagem inconsistentes -> ausente (imputado depois)
    inval = (df["prazo_estimado_dias"] <= 0).sum()
    df.loc[df["prazo_estimado_dias"] <= 0, "prazo_estimado_dias"] = np.nan
    report["prazo_estimado_invalido_p_nan"] = int(inval)

    # outliers EXTREMOS do target (acima de Q3 + k*IQR): entregas atípicas
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    limite = q3 + iqr_k * (q3 - q1)
    n = len(df)
    df = df[df[TARGET] <= limite]
    report["limite_target_dias"] = round(float(limite), 2)
    report["outliers_target_removidos"] = n - len(df)

    report["linhas_finais"] = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------- #
# 5. Imputação + escalonamento + codificação
# --------------------------------------------------------------------------- #
def _log1p() -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def make_preprocessor() -> ColumnTransformer:
    """ColumnTransformer sem vazamento: ajustar SOMENTE no treino (ou dentro do CV)."""
    num_log = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", _log1p()),
            ("scaler", StandardScaler()),
        ]
    )
    num_robust = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    )
    binary = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    cat_onehot = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False),
            ),
        ]
    )
    cat_target = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            (
                "target",
                TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42)),
            ),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num_log", num_log, list(NUM_LOG)),
            ("num_robust", num_robust, list(NUM_ROBUST)),
            ("binary", binary, list(BINARY)),
            ("cat_onehot", cat_onehot, list(CAT_ONEHOT)),
            ("cat_target", cat_target, list(CAT_TARGET)),
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def feature_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resumo claro das features selecionadas (grupo, descrição e % de faltantes)."""
    rows = []
    for tratamento, cols in FEATURE_GROUPS.items():
        for c, desc in cols.items():
            row = {"feature": c, "tratamento": tratamento, "descricao": desc}
            if df is not None:
                row["pct_faltantes"] = round(df[c].isna().mean() * 100, 2)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def main(raw_dir: Path, out_dir: Path, test_size: float, seed: int, iqr_k: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Lendo tabelas...")
    tables = load_raw(raw_dir)
    print("Construindo dataset em nível de pedido...")
    df = build_order_dataset(tables)
    print(f"  {len(df):,} pedidos entregues, {len(ALL_FEATURES)} features")

    df, report = clean_dataset(df, iqr_k=iqr_k)
    print("Limpeza:", report)
    df.to_csv(out_dir / "olist_dataset.csv", index=False)  # usado também na EDA (tópico 3)

    summary = feature_summary(df)
    summary.to_csv(out_dir / "resumo_features.csv", index=False)

    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    pre = make_preprocessor()
    Xt_tr = pre.fit_transform(X_tr, y_tr)  # ajusta SÓ no treino
    Xt_te = pre.transform(X_te)
    print(f"Treino: {Xt_tr.shape} | Teste: {Xt_te.shape} | NaN restantes: {int(Xt_tr.isna().sum().sum())}")

    Xt_tr.to_csv(out_dir / "X_train.csv", index=False)
    Xt_te.to_csv(out_dir / "X_test.csv", index=False)
    y_tr.to_csv(out_dir / "y_train.csv", index=False)
    y_te.to_csv(out_dir / "y_test.csv", index=False)
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    print(f"Arquivos gravados em {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iqr-k", type=float, default=3.0, help="multiplicador do IQR p/ outliers do target")
    a = p.parse_args()
    main(a.raw_dir, a.out_dir, a.test_size, a.seed, a.iqr_k)
