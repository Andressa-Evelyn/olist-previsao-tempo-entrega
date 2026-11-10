"""Análise Exploratória dos Dados (Tópico 3) — Olist: previsão do tempo de entrega.

Gera as figuras (PNG, para os slides) e tabelas (CSV) da EDA a partir da base em
nível de pedido construída no Tópico 2 (`src/preprocessing.py`).

Etapas:
    1. Estatísticas descritivas  -> média, mediana, desvio, quartis, assimetria...
    2. Distribuição das variáveis -> histogramas do alvo, numéricas e categóricas
    3. Outliers                   -> corte do alvo, boxplots, contagem por IQR, anomalias
    4. Correlações                -> heatmap Spearman, ranking com o alvo, dispersões
    5. Mapas geográficos          -> tempo de entrega e densidade por lat/lng
    Extra: atraso x nota da review (as reviews entram SÓ na EDA; ver docs/01)

Uso:
    python src/eda.py                       # lê data/raw, grava em docs/figures/03_eda

Observação: a EDA usa a base já LIMPA (mesma que alimenta o modelo); a etapa de
outliers usa também a base ANTES da limpeza, para mostrar o que foi cortado.
"""

from __future__ import annotations

import argparse
import locale
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # só gera arquivos, sem janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm

from preprocessing import (
    BINARY,
    NUM_LOG,
    NUM_ROBUST,
    RAW_DIR,
    ROOT,
    TARGET,
    build_geo_table,
    build_order_dataset,
    clean_dataset,
    load_raw,
)

OUT_DIR = ROOT / "docs" / "figures" / "03_eda"

NUMERICAS = list(NUM_LOG) + list(NUM_ROBUST)
COORDS = ["cliente_lat", "cliente_lng", "vendedor_lat", "vendedor_lng"]

# --------------------------------------------------------------------------- #
# Estilo dos gráficos (padrão único para todos os slides)
# --------------------------------------------------------------------------- #
COR = "#2a78d6"        # série principal (azul)
COR_DESTAQUE = "#eb6834"  # linhas de referência (média, cortes)
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e4e3df"
FUNDO = "#fcfcfb"

# sequencial (magnitude): um só matiz, claro -> escuro
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# divergente (correlação): azul (-) <-> cinza neutro (0) <-> vermelho (+)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "diverg", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a52a2a"]
)

DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

ROTULOS = {
    TARGET: "Tempo de entrega (dias)",
    "preco_total": "Preço total (R$)",
    "frete_total": "Frete total (R$)",
    "peso_total_g": "Peso total (g)",
    "volume_total_cm3": "Volume total (cm³)",
    "distancia_media_km": "Distância média (km)",
    "distancia_max_km": "Distância máxima (km)",
    "razao_frete_preco": "Frete / preço",
    "n_itens": "Nº de itens",
    "n_produtos": "Nº de produtos distintos",
    "n_vendedores": "Nº de vendedores",
    "prazo_estimado_dias": "Prazo estimado (dias)",
    "prazo_postagem_dias": "Prazo de postagem (dias)",
    "parcelas_max": "Parcelas (máx.)",
    "compra_hora": "Hora da compra",
    "cliente_lat": "Latitude do cliente",
    "cliente_lng": "Longitude do cliente",
    "vendedor_lat": "Latitude do vendedor",
    "vendedor_lng": "Longitude do vendedor",
    "mesmo_estado": "Mesmo estado (vend. = cliente)",
    "compra_fim_semana": "Compra no fim de semana",
    "sem_info_produto": "Produto sem peso/dimensões",
}


def rotulo(col: str) -> str:
    return ROTULOS.get(col, col)


BOX_STYLE = dict(
    orientation="horizontal",
    patch_artist=True,
    boxprops={"facecolor": "#b7d3f6", "edgecolor": COR},
    medianprops={"color": COR, "lw": 2},
    whiskerprops={"color": COR},
    capprops={"color": COR},
    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.25,
                "markerfacecolor": COR, "markeredgecolor": "none"},
)


def fmt_int(n: int) -> str:
    """12345 -> '12.345' (separador de milhar brasileiro)."""
    return f"{n:,}".replace(",", ".")


def fmt_dec(x: float, casas: int = 1) -> str:
    """42.58 -> '42,6'."""
    return f"{x:.{casas}f}".replace(".", ",")


def setup_style() -> None:
    # vírgula decimal nos eixos (se o sistema tiver o locale pt_BR; senão, segue com ponto)
    try:
        locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")
        plt.rcParams["axes.formatter.use_locale"] = True
    except locale.Error:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "figure.facecolor": FUNDO,
            "axes.facecolor": FUNDO,
            "savefig.facecolor": FUNDO,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXTO_2,
            "axes.edgecolor": TEXTO_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "xtick.color": TEXTO_2,
            "ytick.color": TEXTO_2,
            "text.color": TEXTO,
            "legend.frameon": False,
        }
    )


def salvar(fig: plt.Figure, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{nome}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {nome}.png")


def salvar_tabela(tab: pd.DataFrame, nome: str, index: bool = True) -> None:
    (OUT_DIR / "tabelas").mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "tabelas" / f"{nome}.csv", index=index)
    print(f"  tabela: tabelas/{nome}.csv")


def hist(ax: plt.Axes, s: pd.Series, bins: int = 50, **kw) -> None:
    ax.hist(s.dropna(), bins=bins, color=COR, edgecolor=FUNDO, linewidth=0.5, **kw)


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def carregar(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Retorna (base antes da limpeza, base limpa, tabelas brutas)."""
    tabelas = load_raw(raw_dir)
    bruto = build_order_dataset(tabelas)
    limpo, relatorio = clean_dataset(bruto)
    print(f"  {fmt_int(len(bruto))} pedidos antes da limpeza -> {fmt_int(len(limpo))} depois")
    print(f"  relatório de limpeza: {relatorio}")
    # reviews ficam fora do pipeline do modelo (ocorrem após a entrega); aqui só para a EDA
    tabelas["reviews"] = pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv")
    return bruto, limpo, tabelas


# --------------------------------------------------------------------------- #
# 1. Estatísticas descritivas
# --------------------------------------------------------------------------- #
def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Média, mediana, desvio, quartis, assimetria e curtose do alvo e numéricas."""
    cols = [TARGET] + NUMERICAS
    d = df[cols]
    tab = pd.DataFrame(
        {
            "n": d.count(),
            "faltantes_pct": d.isna().mean() * 100,
            "media": d.mean(),
            "mediana": d.median(),
            "desvio_padrao": d.std(),
            "min": d.min(),
            "q1": d.quantile(0.25),
            "q3": d.quantile(0.75),
            "max": d.max(),
            "assimetria": d.skew(),
            "curtose": d.kurt(),
        }
    ).round(3)
    tab.index.name = "variavel"
    return tab


def etapa_1_estatisticas(df: pd.DataFrame) -> None:
    print("1. Estatísticas descritivas")
    salvar_tabela(estatisticas_descritivas(df), "estatisticas_numericas")

    # alvo por UF do cliente
    por_uf = (
        df.groupby("cliente_estado")[TARGET]
        .agg(pedidos="count", media="mean", mediana="median", desvio_padrao="std",
             p90=lambda s: s.quantile(0.9))
        .sort_values("mediana")
        .round(2)
    )
    salvar_tabela(por_uf, "alvo_por_uf_cliente")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(por_uf.index, por_uf["mediana"], color=COR, edgecolor=FUNDO, linewidth=1)
    med = df[TARGET].median()
    ax.axhline(med, color=COR_DESTAQUE, lw=1.5, ls="--",
               label=f"mediana Brasil: {fmt_dec(med)} dias")
    ax.legend(loc="upper left")
    ax.set_ylabel("Mediana do tempo de entrega (dias)")
    ax.set_title("Tempo de entrega por UF do cliente")
    for i, v in enumerate(por_uf["mediana"]):
        ax.text(i, v + 0.3, f"{v:.0f}", ha="center", fontsize=7.5, color=TEXTO_2)
    salvar(fig, "1_alvo_por_uf")


# --------------------------------------------------------------------------- #
# 2. Distribuição das variáveis
# --------------------------------------------------------------------------- #
def etapa_2_distribuicoes(df: pd.DataFrame) -> None:
    print("2. Distribuição das variáveis")
    y = df[TARGET]

    # --- alvo: escala linear e log --------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    hist(ax, y, bins=60)
    for v, nome, ls in [(y.mean(), "média", "--"), (y.median(), "mediana", ":")]:
        ax.axvline(v, color=COR_DESTAQUE, lw=1.5, ls=ls, label=f"{nome}: {fmt_dec(v)} dias")
    ax.legend(loc="upper right")
    ax.set_title(f"Tempo de entrega — assimetria {fmt_dec(y.skew(), 2)}")
    ax.set_xlabel(rotulo(TARGET))
    ax.set_ylabel("Pedidos")

    ax = axes[1]
    hist(ax, np.log1p(y), bins=60)
    ax.set_title(f"log(1 + tempo de entrega) — assimetria {fmt_dec(np.log1p(y).skew(), 2)}")
    ax.set_xlabel("log(1 + dias)")
    ax.set_ylabel("Pedidos")
    salvar(fig, "2_dist_alvo")

    # --- numéricas assimétricas: bruto x log1p (justifica o Tópico 2) ---- #
    cols = list(NUM_LOG)
    fig, axes = plt.subplots(2, len(cols), figsize=(3 * len(cols), 5.6))
    for j, c in enumerate(cols):
        s = df[c].dropna()
        p99 = s.quantile(0.99)
        hist(axes[0, j], s[s <= p99], bins=40)
        axes[0, j].set_title(f"{rotulo(c)}\nassim. {fmt_dec(s.skew())}", fontsize=9)
        hist(axes[1, j], np.log1p(s), bins=40)
        axes[1, j].set_title(f"log1p — assim. {fmt_dec(np.log1p(s).skew())}", fontsize=9)
        for ax in axes[:, j]:
            ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Original (até p99)")
    axes[1, 0].set_ylabel("Após log1p")
    fig.suptitle("Numéricas de cauda longa: original × log1p", x=0.01, ha="left",
                 fontweight="bold")
    salvar(fig, "2_dist_numericas_log")

    # --- demais numéricas ------------------------------------------------- #
    cols = list(NUM_ROBUST)
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow))
    for ax, c in zip(axes.flat, cols):
        s = df[c].dropna()
        # cauda extrema esmaga o gráfico: exibe até p99,5 e informa o máximo
        corte = s.quantile(0.995)
        titulo = rotulo(c)
        sem_corte = c == "compra_hora" or c.endswith(("_lat", "_lng"))
        if s.max() > corte and not sem_corte:
            titulo += f"\n(até p99,5 = {corte:.0f}; máx. {fmt_int(round(s.max()))})"
            s = s[s <= corte]
        if s.nunique() <= 25:  # discreta -> barras por valor
            vc = s.value_counts().sort_index()
            ax.bar(vc.index, vc.values, color=COR, edgecolor=FUNDO, linewidth=0.5, width=0.85)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        else:
            hist(ax, s, bins=40)
        ax.set_title(titulo, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Demais numéricas", x=0.01, ha="left", fontweight="bold")
    salvar(fig, "2_dist_numericas_robust")

    # --- categóricas ------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    vc = df["compra_dia_semana"].value_counts().sort_index()
    axes[0, 0].bar([DIAS[i] for i in vc.index], vc.values, color=COR, edgecolor=FUNDO)
    axes[0, 0].set_title("Pedidos por dia da semana da compra")

    vc = df["compra_mes"].value_counts().sort_index()
    axes[0, 1].bar([MESES[i - 1] for i in vc.index], vc.values, color=COR, edgecolor=FUNDO)
    axes[0, 1].set_title("Pedidos por mês da compra (2016–2018 somados)")

    vc = df["tipo_pagamento"].fillna("desconhecido").value_counts()
    axes[1, 0].barh(vc.index[::-1], vc.values[::-1], color=COR, edgecolor=FUNDO)
    axes[1, 0].set_title("Forma de pagamento predominante")
    for i, v in enumerate(vc.values[::-1]):
        axes[1, 0].text(v, i, f" {fmt_dec(v / vc.sum() * 100)}%", va="center", fontsize=8,
                        color=TEXTO_2)

    vc = df["categoria_produto"].value_counts()
    top = vc.head(15)
    axes[1, 1].barh(top.index[::-1], top.values[::-1], color=COR, edgecolor=FUNDO)
    axes[1, 1].set_title(f"Top 15 categorias de produto (de {vc.size})")
    axes[1, 1].tick_params(axis="y", labelsize=8)

    for ax in axes[1]:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    for ax in axes.flat:
        ax.set_ylabel("Pedidos" if ax in axes[0] else "")
    salvar(fig, "2_dist_categoricas")


# --------------------------------------------------------------------------- #
# 3. Outliers
# --------------------------------------------------------------------------- #
def contar_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Quantos valores ficam fora de Q1 - k*IQR / Q3 + k*IQR (k = 1,5 e 3)."""
    linhas = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        linha = {"variavel": c, "q1": q1, "q3": q3, "iqr": iqr, "max": s.max()}
        for k in (1.5, 3.0):
            fora = (s < q1 - k * iqr) | (s > q3 + k * iqr)
            linha[f"outliers_{k}iqr"] = int(fora.sum())
            linha[f"pct_{k}iqr"] = round(fora.mean() * 100, 2)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("variavel").round(3)


def etapa_3_outliers(bruto: pd.DataFrame, df: pd.DataFrame) -> None:
    print("3. Outliers")
    y = bruto[TARGET]
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_15, lim_3 = q3 + 1.5 * iqr, q3 + 3 * iqr
    n_15, n_3 = int((y > lim_15).sum()), int((y > lim_3).sum())

    # --- alvo ANTES da limpeza: onde o corte foi feito --------------------- #
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3]})
    axes[0].boxplot(y, widths=0.6, **BOX_STYLE)
    axes[0].set_yticks([])
    axes[0].grid(False)
    axes[0].set_title(f"Tempo de entrega antes da limpeza — {fmt_int(len(y))} pedidos")
    hist(axes[1], y, bins=120)
    axes[1].set_yscale("log")
    axes[1].axvline(lim_15, color=COR_DESTAQUE, ls=":", lw=1.5,
                    label=f"Q3 + 1,5·IQR = {fmt_dec(lim_15)} dias ({fmt_int(n_15)} pedidos acima)")
    axes[1].axvline(lim_3, color=COR_DESTAQUE, ls="--", lw=1.8,
                    label=f"Q3 + 3·IQR = {fmt_dec(lim_3)} dias ({fmt_int(n_3)} removidos) — corte usado")
    for ax in axes:
        ax.axvline(lim_3, color=COR_DESTAQUE, ls="--", lw=1.8)
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel(rotulo(TARGET))
    axes[1].set_ylabel("Pedidos (escala log)")
    salvar(fig, "3_outliers_alvo")

    # --- contagem de outliers por variável (base limpa) -------------------- #
    cols = [c for c in NUMERICAS if c not in COORDS + ["compra_hora"]]
    salvar_tabela(contar_outliers(df, [TARGET] + cols), "outliers_por_variavel")

    # --- boxplots (pequenos múltiplos, cada um na sua escala) -------------- #
    cols_box = [TARGET] + list(NUM_LOG) + ["prazo_estimado_dias", "prazo_postagem_dias",
                                           "parcelas_max", "n_itens"]
    ncol = 4
    nrow = int(np.ceil(len(cols_box) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.1 * nrow))
    for ax, c in zip(axes.flat, cols_box):
        s = df[c].dropna()
        if c in NUM_LOG:
            s = s[s > 0]  # escala log não comporta zero (ex.: frete grátis)
        ax.boxplot(s, widths=0.55, **BOX_STYLE)
        if c in NUM_LOG:
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        titulo = rotulo(c) + (" — escala log" if c in NUM_LOG else "")
        if c == "prazo_postagem_dias":
            titulo += f" — máx. {fmt_int(int(s.max()))} (erro de cadastro)"
        ax.set_title(titulo, fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(cols_box):]:
        ax.set_visible(False)
    fig.suptitle("Boxplots — pontos = valores além de 1,5·IQR", x=0.01, ha="left",
                 fontweight="bold")
    salvar(fig, "3_boxplots")

    # --- anomalias (erros prováveis de cadastro) --------------------------- #
    max_post = fmt_int(round(df["prazo_postagem_dias"].max()))
    anom = pd.DataFrame(
        [
            ("prazo_postagem_dias > 30", int((df["prazo_postagem_dias"] > 30).sum()),
             f"prazo para postar de até {max_post} dias: erro de cadastro"),
            ("parcelas_max == 0", int((df["parcelas_max"] == 0).sum()),
             "pagamento com 0 parcelas é inválido"),
            ("prazo_estimado_dias > 60", int((df["prazo_estimado_dias"] > 60).sum()),
             "promessa de entrega acima de 2 meses"),
            ("frete_total == 0", int((df["frete_total"] == 0).sum()),
             "frete grátis (legítimo, mas distorce frete/preço)"),
            ("distancia_media_km == 0", int((df["distancia_media_km"] == 0).sum()),
             "vendedor e cliente no mesmo prefixo de CEP"),
        ],
        columns=["regra", "pedidos", "interpretacao"],
    )
    salvar_tabela(anom, "anomalias", index=False)


# --------------------------------------------------------------------------- #
# 4. Correlações
# --------------------------------------------------------------------------- #
def etapa_4_correlacoes(df: pd.DataFrame) -> None:
    print("4. Correlações")
    cols = [TARGET] + NUMERICAS + list(BINARY)
    d = df[cols]
    spear = d.corr(method="spearman")
    pear = d.corr(method="pearson")

    # --- heatmap (triângulo inferior) ------------------------------------- #
    rot = [rotulo(c) for c in cols]
    m = spear.values.copy()
    m[np.triu_indices_from(m, k=0)] = np.nan  # esconde diagonal (sempre 1) e triângulo superior
    m = m[1:, :-1]  # sem a diagonal, 1ª linha e última coluna ficariam vazias
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(m, cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols) - 1), rot[:-1], rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols) - 1), rot[1:], fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(i + 1):
            v = m[i, j]
            if abs(v) >= 0.1:
                ax.text(j, i, fmt_dec(v, 2), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else TEXTO)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Correlação de Spearman")
    ax.set_title("Matriz de correlação (Spearman) — valores |ρ| ≥ 0,10 anotados")
    salvar(fig, "4_correlacao_heatmap")

    # --- ranking: correlação de cada feature com o alvo (tabela) ---------- #
    rank = pd.DataFrame({"spearman": spear[TARGET], "pearson": pear[TARGET]}).drop(TARGET)
    rank = rank.reindex(rank["spearman"].abs().sort_values(ascending=False).index)
    salvar_tabela(rank.round(3), "correlacao_com_alvo")

    # --- dispersões: alvo x 2 features mais fortes ------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, c in zip(axes, ["distancia_media_km", "prazo_estimado_dias"]):
        sub = df[[c, TARGET]].dropna()
        sub = sub[sub[c] <= sub[c].quantile(0.995)]
        hb = ax.hexbin(sub[c], sub[TARGET], gridsize=55, cmap=CMAP_SEQ, mincnt=1,
                       norm=LogNorm(), linewidths=0)
        # mediana do alvo por faixa da feature
        faixas = pd.qcut(sub[c], 20, duplicates="drop")
        med = sub.groupby(faixas, observed=True).agg(x=(c, "median"), y=(TARGET, "median"))
        ax.plot(med["x"], med["y"], color=COR_DESTAQUE, lw=2, marker="o", ms=4,
                label="mediana por faixa")
        if c == "prazo_estimado_dias":
            lim = sub[c].max()
            ax.plot([0, lim], [0, lim], color=TEXTO_2, ls="--", lw=1,
                    label="entrega = prazo prometido")
        ax.set_xlabel(rotulo(c))
        ax.set_title(f"× {rotulo(c).lower()} — Spearman ρ = {fmt_dec(spear.loc[c, TARGET], 2)}",
                     fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(False)
    axes[0].set_ylabel(rotulo(TARGET))
    fig.suptitle("Tempo de entrega × as duas features mais correlacionadas", x=0.06, ha="left",
                 fontweight="bold")
    cb = fig.colorbar(hb, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("Pedidos (escala log)")
    fig.savefig(OUT_DIR / "4_dispersao_alvo.png", bbox_inches="tight")
    plt.close(fig)
    print("  figura: 4_dispersao_alvo.png")

    # --- multicolinearidade: pares de features muito correlacionados ------ #
    feats = spear.drop(index=TARGET, columns=TARGET)
    pares = (
        feats.where(np.triu(np.ones(feats.shape, dtype=bool), k=1))
        .stack()
        .rename("spearman")
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )
    pares = pares[pares["spearman"].abs() >= 0.7].sort_values("spearman", key=abs, ascending=False)
    salvar_tabela(pares.round(3), "multicolinearidade", index=False)


# --------------------------------------------------------------------------- #
# 5. Mapas geográficos
# --------------------------------------------------------------------------- #
def eixo_mapa(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-74.5, -34)
    ax.set_ylim(-34.5, 6)
    ax.grid(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def etapa_5_mapas(df: pd.DataFrame, tabelas: dict) -> None:
    print("5. Mapas geográficos")
    cli = df.dropna(subset=["cliente_lat", "cliente_lng"])

    # --- tempo de entrega pela localização do cliente --------------------- #
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.hexbin(cli["cliente_lng"], cli["cliente_lat"], gridsize=(70, 70),
              extent=(-74.5, -34, -34.5, 6), cmap=ListedColormap(["#d9d8d4"]), mincnt=1,
              linewidths=0)
    hb = ax.hexbin(cli["cliente_lng"], cli["cliente_lat"], C=cli[TARGET],
                   reduce_C_function=np.median, gridsize=(70, 70),
                   extent=(-74.5, -34, -34.5, 6), cmap=CMAP_SEQ, mincnt=5,
                   vmin=5, vmax=25, linewidths=0)
    eixo_mapa(ax)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02, extend="both")
    cb.set_label("Mediana do tempo de entrega (dias)")
    ax.set_title("Tempo de entrega pela localização do cliente\n"
                 "(hexágonos com ≥ 5 pedidos; cinza = menos de 5)")
    salvar(fig, "5_mapa_tempo_entrega")

    # --- onde estão clientes x vendedores --------------------------------- #
    geo = build_geo_table(tabelas["geolocation"])
    vend = tabelas["sellers"].merge(geo, left_on="seller_zip_code_prefix", right_on="zip")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    for ax, (x, y, titulo) in zip(
        axes,
        [
            (cli["cliente_lng"], cli["cliente_lat"], f"Clientes — {fmt_int(len(cli))} pedidos"),
            (vend["lng"], vend["lat"], f"Vendedores — {fmt_int(len(vend))} vendedores"),
        ],
    ):
        hb = ax.hexbin(x, y, gridsize=(70, 70), extent=(-74.5, -34, -34.5, 6),
                       cmap=CMAP_SEQ, norm=LogNorm(), mincnt=1, linewidths=0)
        eixo_mapa(ax)
        ax.set_title(titulo)
        cb = fig.colorbar(hb, ax=ax, shrink=0.65, pad=0.02)
        cb.set_label("Quantidade (escala log)")
    salvar(fig, "5_mapa_clientes_vendedores")

    # tabela de apoio: região do cliente
    regiao = {
        **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
        **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
        **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
        **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
        **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
    }
    tab = (
        df.assign(regiao=df["cliente_estado"].map(regiao))
        .groupby("regiao")
        .agg(pedidos=(TARGET, "count"), mediana_dias=(TARGET, "median"),
             media_dias=(TARGET, "mean"), distancia_mediana_km=("distancia_media_km", "median"),
             pct_mesmo_estado=("mesmo_estado", "mean"))
        .sort_values("mediana_dias")
    )
    tab["pct_mesmo_estado"] *= 100
    salvar_tabela(tab.round(2), "alvo_por_regiao")


# --------------------------------------------------------------------------- #
# Extra: atraso x nota da review (sustenta o "impacto" do Tópico 1)
# --------------------------------------------------------------------------- #
def etapa_extra_reviews(df: pd.DataFrame, reviews: pd.DataFrame) -> None:
    print("Extra: atraso x nota da review")
    nota = reviews.groupby("order_id", as_index=False)["review_score"].mean()
    d = df[["order_id", TARGET, "prazo_estimado_dias"]].merge(nota, on="order_id", how="inner")
    d["atraso_dias"] = d[TARGET] - d["prazo_estimado_dias"]  # > 0 = chegou depois do prometido

    faixas = [-np.inf, -10, -5, 0, 5, 10, np.inf]
    nomes = ["10+ dias\nadiantado", "5–10\nadiantado", "0–5\nadiantado",
             "0–5\natrasado", "5–10\natrasado", "10+ dias\natrasado"]
    d["faixa"] = pd.cut(d["atraso_dias"], faixas, labels=nomes)
    tab = d.groupby("faixa", observed=True).agg(
        pedidos=("review_score", "size"),
        nota_media=("review_score", "mean"),
        pct_nota_1_2=("review_score", lambda s: (s <= 2).mean() * 100),
    )
    tab.index = [n.replace("\n", " ") for n in tab.index]
    salvar_tabela(tab.round(2), "review_por_atraso")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(nomes, tab["nota_media"], color=COR, edgecolor=FUNDO, linewidth=1)
    for i, (v, n) in enumerate(zip(tab["nota_media"], tab["pedidos"])):
        ax.text(i, v + 0.06, f"{fmt_dec(v, 2)}\n({fmt_int(n)} pedidos)", ha="center",
                fontsize=8, color=TEXTO_2)
    ax.axvline(2.5, color=COR_DESTAQUE, ls="--", lw=1.5, label="prazo prometido")
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("Nota média da review (1–5)")
    rho = d["atraso_dias"].corr(d["review_score"], method="spearman")
    ax.set_title(f"Nota da review por atraso em relação ao prazo prometido "
                 f"— Spearman ρ = {fmt_dec(rho, 2)}")
    ax.legend(loc="upper right")
    salvar(fig, "6_review_por_atraso")


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def main(raw_dir: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    print("Carregando dados...")
    bruto, df, tabelas = carregar(raw_dir)

    etapa_1_estatisticas(df)
    etapa_2_distribuicoes(df)
    etapa_3_outliers(bruto, df)
    etapa_4_correlacoes(df)
    etapa_5_mapas(df, tabelas)
    etapa_extra_reviews(df, tabelas["reviews"])
    print(f"Saídas em {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    a = p.parse_args()
    main(a.raw_dir)
