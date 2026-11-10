"""Treinamento do Modelo (Tópico 4) — Olist: previsão do tempo de entrega.

Fluxo:
    1. Carrega X_train, X_test, y_train, y_test (gerados sem vazamento no Tópico 2)
       e recupera o prazo estimado original para o baseline de negócio da Olist.
    2. Define o esquema de validação cruzada: 5-Fold CV (shuffle=True, random_state=42)
       exclusivamente sobre o conjunto de treino (76.080 pedidos).
    3. Avalia os Baselines de referência (Prazo Estimado da Olist e Dummy Mediana).
    4. Avalia os 4 modelos em configuração default (5-Fold CV):
       - Linear: Ridge
       - Árvore de Decisão: DecisionTreeRegressor
       - Baseado em Instâncias: KNeighborsRegressor (KNN)
       - Ensemble: RandomForestRegressor
    5. Executa o ajuste de hiperparâmetros (GridSearchCV e RandomizedSearchCV):
       - Ridge (GridSearchCV + experimento alvo original vs. log1p)
       - DecisionTreeRegressor (GridSearchCV)
       - KNeighborsRegressor (RandomizedSearchCV)
       - RandomForestRegressor (RandomizedSearchCV)
    6. Gera curvas de validação, gráficos comparativos, tabelas CSV, salva os
       modelos otimizados e registra os resultados parciais no conjunto de teste.

Uso:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
import locale
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # gera arquivos sem abrir janela gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from preprocessing import PROCESSED_DIR, ROOT, TARGET

OUT_DIR = ROOT / "docs" / "figures" / "04_treinamento"
MODELS_DIR = PROCESSED_DIR / "models"

# --------------------------------------------------------------------------- #
# Estilo visual idêntico ao src/eda.py (padronização dos slides)
# --------------------------------------------------------------------------- #
COR = "#2a78d6"           # azul principal (modelos otimizados / série principal)
COR_CLARA = "#9bc2f2"     # azul claro (modelos default / treino)
COR_DESTAQUE = "#eb6834"  # laranja destaque (baselines / pontos ótimos / validação)
COR_VERDE = "#1f8a5b"     # verde apoio (melhor modelo / ganho)
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e4e3df"
FUNDO = "#fcfcfb"

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_dec(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def setup_style() -> None:
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


def salvar_tabela(tab: pd.DataFrame, nome: str, index: bool = False) -> None:
    (OUT_DIR / "tabelas").mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "tabelas" / f"{nome}.csv", index=index)
    print(f"  tabela: tabelas/{nome}.csv")


# --------------------------------------------------------------------------- #
# 1. Carregamento dos dados processados
# --------------------------------------------------------------------------- #
def carregar_dados(
    proc_dir: Path = PROCESSED_DIR, seed: int = 42, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Lê X_train, X_test, y_train, y_test e recupera prazo_estimado_dias bruto."""
    X_tr = pd.read_csv(proc_dir / "X_train.csv")
    X_te = pd.read_csv(proc_dir / "X_test.csv")
    y_tr = pd.read_csv(proc_dir / "y_train.csv").squeeze("columns")
    y_te = pd.read_csv(proc_dir / "y_test.csv").squeeze("columns")

    # Prazo estimado original (em dias) para o baseline de negócio da Olist
    df_full = pd.read_csv(proc_dir / "olist_dataset.csv")
    df_tr, df_te = train_test_split(df_full, test_size=test_size, random_state=seed)
    prazo_tr = df_tr["prazo_estimado_dias"].fillna(df_tr["prazo_estimado_dias"].median()).reset_index(drop=True)
    prazo_te = df_te["prazo_estimado_dias"].fillna(df_tr["prazo_estimado_dias"].median()).reset_index(drop=True)

    return X_tr, X_te, y_tr, y_te, prazo_tr, prazo_te


# --------------------------------------------------------------------------- #
# 2. Avaliação em Validação Cruzada (5-fold) e Teste
# --------------------------------------------------------------------------- #
def avaliar_baseline_olist(y: pd.Series, prazo: pd.Series, cv: KFold) -> dict:
    """Calcula métricas 5-fold do baseline atual da Olist (prever prazo_estimado_dias)."""
    rmses, maes, r2s = [], [], []
    y_arr = y.to_numpy()
    p_arr = prazo.to_numpy()
    for _, val_idx in cv.split(y_arr):
        yv, pv = y_arr[val_idx], p_arr[val_idx]
        rmses.append(root_mean_squared_error(yv, pv))
        maes.append(mean_absolute_error(yv, pv))
        r2s.append(r2_score(yv, pv))
    return {
        "modelo": "Prazo Estimado Olist",
        "familia": "Baseline (Negócio)",
        "estagio": "Baseline",
        "train_rmse_mean": round(float(root_mean_squared_error(y_arr, p_arr)), 4),
        "cv_rmse_mean": round(float(np.mean(rmses)), 4),
        "cv_rmse_std": round(float(np.std(rmses)), 4),
        "cv_mae_mean": round(float(np.mean(maes)), 4),
        "cv_mae_std": round(float(np.std(maes)), 4),
        "train_r2_mean": round(float(r2_score(y_arr, p_arr)), 4),
        "cv_r2_mean": round(float(np.mean(r2s)), 4),
        "cv_r2_std": round(float(np.std(r2s)), 4),
        "tempo_s": 0.0,
        "fold_rmses": rmses,
        "fold_r2s": r2s,
    }


def avaliar_cv(
    estimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: KFold,
    modelo: str,
    familia: str,
    estagio: str,
    n_jobs_cv: int = -1,
) -> dict:
    """Executa cross_validate (5-fold) retornando métricas de treino e validação."""
    t0 = time.perf_counter()
    res = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=n_jobs_cv,
    )
    dt = time.perf_counter() - t0
    rmses = (-res["test_rmse"]).tolist()
    maes = (-res["test_mae"]).tolist()
    r2s = res["test_r2"].tolist()
    return {
        "modelo": modelo,
        "familia": familia,
        "estagio": estagio,
        "train_rmse_mean": round(float((-res["train_rmse"]).mean()), 4),
        "cv_rmse_mean": round(float(np.mean(rmses)), 4),
        "cv_rmse_std": round(float(np.std(rmses)), 4),
        "cv_mae_mean": round(float(np.mean(maes)), 4),
        "cv_mae_std": round(float(np.std(maes)), 4),
        "train_r2_mean": round(float(res["train_r2"].mean()), 4),
        "cv_r2_mean": round(float(np.mean(r2s)), 4),
        "cv_r2_std": round(float(np.std(r2s)), 4),
        "tempo_s": round(dt, 2),
        "fold_rmses": rmses,
        "fold_r2s": r2s,
    }


def metricas_teste(
    nome: str, familia: str, estagio: str, y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    erro = y_pred - y_true
    return {
        "modelo": nome,
        "familia": familia,
        "estagio": estagio,
        "rmse_teste": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae_teste": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "medae_teste": round(float(median_absolute_error(y_true, y_pred)), 4),
        "r2_teste": round(float(r2_score(y_true, y_pred)), 4),
        "vies_medio_dias": round(float(np.mean(erro)), 4),
        "pct_erro_ate_3d": round(float((np.abs(erro) <= 3.0).mean() * 100), 2),
        "pct_erro_ate_5d": round(float((np.abs(erro) <= 5.0).mean() * 100), 2),
    }


# --------------------------------------------------------------------------- #
# 3. Experimento especial da EDA: Ridge com alvo original vs. log1p(alvo)
# --------------------------------------------------------------------------- #
def experimento_ridge_log(
    X_tr: np.ndarray, y_tr: np.ndarray, cv: KFold
) -> pd.DataFrame:
    """Testa a hipótese 2 da EDA: efeito de log1p(alvo) + expm1 no modelo Ridge."""
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    linhas = []
    warnings.filterwarnings("ignore", category=UserWarning)

    for escala in ["Original (dias)", "log1p(dias) -> expm1"]:
        for alpha in alphas:
            base = Ridge(alpha=alpha, random_state=42)
            if escala.startswith("log1p"):
                est = TransformedTargetRegressor(
                    regressor=base,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                )
            else:
                est = base
            res = cross_validate(
                est,
                X_tr,
                y_tr,
                cv=cv,
                scoring=SCORING,
                return_train_score=True,
                n_jobs=-1,
            )
            linhas.append(
                {
                    "escala_alvo": escala,
                    "alpha": alpha,
                    "train_rmse": round(float((-res["train_rmse"]).mean()), 4),
                    "cv_rmse": round(float((-res["test_rmse"]).mean()), 4),
                    "cv_rmse_std": round(float((-res["test_rmse"]).std()), 4),
                    "cv_mae": round(float((-res["test_mae"]).mean()), 4),
                    "cv_r2": round(float(res["test_r2"].mean()), 4),
                }
            )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# 4. Figuras para os slides
# --------------------------------------------------------------------------- #
def plot_1_default_vs_tuned(df_cv: pd.DataFrame) -> None:
    """Comparação de RMSE e MAE (5-fold CV) entre Baselines, Default e Tuned."""
    modelos_ordem = ["Ridge", "KNN", "Decision Tree", "Random Forest"]
    d_def = df_cv[df_cv["estagio"] == "Default"].set_index("modelo").loc[modelos_ordem]
    d_tun = df_cv[df_cv["estagio"] == "Tuned"].set_index("modelo").loc[modelos_ordem]
    dummy_rmse = float(df_cv.loc[df_cv["modelo"] == "Dummy (Mediana)", "cv_rmse_mean"].iloc[0])
    dummy_mae = float(df_cv.loc[df_cv["modelo"] == "Dummy (Mediana)", "cv_mae_mean"].iloc[0])
    olist_rmse = float(df_cv.loc[df_cv["modelo"] == "Prazo Estimado Olist", "cv_rmse_mean"].iloc[0])
    olist_mae = float(df_cv.loc[df_cv["modelo"] == "Prazo Estimado Olist", "cv_mae_mean"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(modelos_ordem))
    w = 0.36

    # --- Painel 1: RMSE ---
    ax = axes[0]
    b1 = ax.bar(
        x - w / 2,
        d_def["cv_rmse_mean"],
        w,
        yerr=d_def["cv_rmse_std"],
        capsize=3,
        color=COR_CLARA,
        edgecolor=FUNDO,
        label="Default (sem ajuste)",
    )
    b2 = ax.bar(
        x + w / 2,
        d_tun["cv_rmse_mean"],
        w,
        yerr=d_tun["cv_rmse_std"],
        capsize=3,
        color=COR,
        edgecolor=FUNDO,
        label="Otimizado (após tuning)",
    )
    ax.axhline(
        dummy_rmse,
        color=COR_DESTAQUE,
        ls="--",
        lw=1.5,
        label=f"Baseline Dummy (mediana): {fmt_dec(dummy_rmse)} d",
    )
    ax.axhline(
        olist_rmse,
        color="#8c2d19",
        ls=":",
        lw=1.5,
        label=f"Prazo Estimado Olist: {fmt_dec(olist_rmse)} d",
    )
    for bar in b1:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.14,
            fmt_dec(v),
            ha="center",
            fontsize=8,
            color=TEXTO_2,
        )
    for bar in b2:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.14,
            fmt_dec(v),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=TEXTO,
        )
    ax.set_xticks(x, modelos_ordem)
    ax.set_ylim(0, max(olist_rmse + 1.8, 16.5))
    ax.set_ylabel("RMSE na Validação Cruzada 5-Fold (dias)")
    ax.set_title("Erro Quadrático Médio (RMSE) — Default vs. Otimizado")
    ax.legend(loc="upper right", fontsize=8.5)

    # --- Painel 2: MAE ---
    ax = axes[1]
    b1 = ax.bar(
        x - w / 2,
        d_def["cv_mae_mean"],
        w,
        yerr=d_def["cv_mae_std"],
        capsize=3,
        color=COR_CLARA,
        edgecolor=FUNDO,
        label="Default (sem ajuste)",
    )
    b2 = ax.bar(
        x + w / 2,
        d_tun["cv_mae_mean"],
        w,
        yerr=d_tun["cv_mae_std"],
        capsize=3,
        color=COR,
        edgecolor=FUNDO,
        label="Otimizado (após tuning)",
    )
    ax.axhline(
        dummy_mae,
        color=COR_DESTAQUE,
        ls="--",
        lw=1.5,
        label=f"Baseline Dummy (mediana): {fmt_dec(dummy_mae)} d",
    )
    ax.axhline(
        olist_mae,
        color="#8c2d19",
        ls=":",
        lw=1.5,
        label=f"Prazo Estimado Olist: {fmt_dec(olist_mae)} d",
    )
    for bar in b1:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.14,
            fmt_dec(v),
            ha="center",
            fontsize=8,
            color=TEXTO_2,
        )
    for bar in b2:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.14,
            fmt_dec(v),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=TEXTO,
        )
    ax.set_xticks(x, modelos_ordem)
    ax.set_ylim(0, max(olist_mae + 1.8, 15.0))
    ax.set_ylabel("MAE na Validação Cruzada 5-Fold (dias)")
    ax.set_title("Erro Absoluto Médio (MAE) — Default vs. Otimizado")
    ax.legend(loc="upper right", fontsize=8.5)

    salvar(fig, "1_cv_default_vs_tuned")


def plot_2_overfitting_treino_vs_cv(df_cv: pd.DataFrame) -> None:
    """Mostra o gap entre erro de treino e erro de validação antes e depois do tuning."""
    modelos_ordem = ["Ridge", "KNN", "Decision Tree", "Random Forest"]
    d_def = df_cv[df_cv["estagio"] == "Default"].set_index("modelo").loc[modelos_ordem]
    d_tun = df_cv[df_cv["estagio"] == "Tuned"].set_index("modelo").loc[modelos_ordem]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    x = np.arange(len(modelos_ordem))
    w = 0.35

    for ax, d, titulo in [
        (axes[0], d_def, "Antes do Tuning (Hiperparâmetros Default)"),
        (axes[1], d_tun, "Depois do Tuning (Hiperparâmetros Otimizados)"),
    ]:
        b_tr = ax.bar(
            x - w / 2,
            d["train_rmse_mean"],
            w,
            color=COR_CLARA,
            edgecolor=FUNDO,
            label="RMSE no Treino",
        )
        b_cv = ax.bar(
            x + w / 2,
            d["cv_rmse_mean"],
            w,
            color=COR,
            edgecolor=FUNDO,
            label="RMSE na Validação (5-Fold CV)",
        )
        for bar in b_tr:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.12,
                fmt_dec(v),
                ha="center",
                fontsize=8,
                color=TEXTO_2,
            )
        for bar in b_cv:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.12,
                fmt_dec(v),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=TEXTO,
            )
        # Anota o gap de generalização (CV - Treino)
        for i, m in enumerate(modelos_ordem):
            gap = d.loc[m, "cv_rmse_mean"] - d.loc[m, "train_rmse_mean"]
            topo = max(d.loc[m, "cv_rmse_mean"], d.loc[m, "train_rmse_mean"])
            ax.text(
                i,
                topo + 0.55,
                f"Δ = +{fmt_dec(gap)} d",
                ha="center",
                fontsize=7.8,
                color=COR_DESTAQUE if gap > 1.5 else TEXTO_2,
                fontweight="bold" if gap > 1.5 else "normal",
            )
        ax.set_xticks(x, modelos_ordem)
        ax.set_ylim(0, 9.3)
        ax.set_title(titulo)
        ax.legend(loc="upper left", fontsize=8.5)

    axes[0].set_ylabel("RMSE (dias) — menor é melhor")
    fig.suptitle(
        "Controle de Overfitting: RMSE de Treino vs. Validação Cruzada (Δ = Gap de Generalização)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    salvar(fig, "2_overfitting_treino_vs_cv")


def plot_3_curvas_tuning(
    df_ridge_log: pd.DataFrame,
    dt_search: GridSearchCV,
    knn_curve_df: pd.DataFrame,
    rf_search: RandomizedSearchCV,
) -> None:
    """Painel 2x2 mostrando o comportamento dos hiperparâmetros em cada família."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))

    # --- (a) Ridge: alpha e escala do alvo (original vs log1p) ---
    ax = axes[0, 0]
    sub_orig = df_ridge_log[df_ridge_log["escala_alvo"].str.startswith("Original")]
    sub_log = df_ridge_log[df_ridge_log["escala_alvo"].str.startswith("log1p")]
    ax.plot(
        sub_orig["alpha"],
        sub_orig["cv_rmse"],
        marker="o",
        color=COR,
        lw=2,
        label="Alvo original — RMSE (CV)",
    )
    ax.plot(
        sub_log["alpha"],
        sub_log["cv_rmse"],
        marker="s",
        color=COR_DESTAQUE,
        lw=1.8,
        ls="--",
        label="Alvo em log1p (expm1) — RMSE (CV)",
    )
    best_r = sub_orig.loc[sub_orig["cv_rmse"].idxmin()]
    ax.scatter([best_r["alpha"]], [best_r["cv_rmse"]], color=COR_VERDE, zorder=5, s=60)
    ax.set_xscale("log")
    ax.set_xlabel("Regularização L2 (alpha, escala log)")
    ax.set_ylabel("RMSE 5-Fold CV (dias)")
    ax.set_title("(a) Ridge: efeito de alpha e da transformação log1p no alvo")
    ax.legend(loc="center right", fontsize=8.5)

    # --- (b) Decision Tree: max_depth x min_samples_leaf ---
    ax = axes[0, 1]
    cv_dt = pd.DataFrame(dt_search.cv_results_)
    cv_dt = cv_dt[
        (cv_dt["param_min_samples_split"] == 2) & (cv_dt["param_max_depth"].notna())
    ].copy()
    cv_dt["param_max_depth"] = cv_dt["param_max_depth"].astype(int)
    for leaf, cor_l, ls_l in [
        (1, "#d95f02", ":"),
        (20, COR_CLARA, "--"),
        (50, "#3987e5", "-."),
        (100, COR, "-"),
        (200, COR_VERDE, "--"),
    ]:
        sub = cv_dt[cv_dt["param_min_samples_leaf"] == leaf].sort_values("param_max_depth")
        if len(sub) > 0:
            ax.plot(
                sub["param_max_depth"],
                -sub["mean_test_rmse"],
                marker="o",
                lw=2.2 if leaf == 100 else 1.5,
                ls=ls_l,
                color=cor_l,
                label=f"min_samples_leaf = {leaf}",
            )
    best_dt_row = cv_dt.loc[cv_dt["mean_test_rmse"].idxmax()]
    ax.scatter(
        [best_dt_row["param_max_depth"]],
        [-best_dt_row["mean_test_rmse"]],
        color=COR_VERDE,
        s=65,
        zorder=5,
    )
    ax.set_xlabel("Profundidade máxima da árvore (max_depth)")
    ax.set_ylabel("RMSE 5-Fold CV (dias)")
    ax.set_title("(b) Decision Tree: poda de profundidade × tamanho mínimo de folha")
    ax.legend(loc="upper left", fontsize=8)

    # --- (c) KNN: n_neighbors x métrica de distância ---
    ax = axes[1, 0]
    for (p_val, w_val), grp in knn_curve_df.groupby(["p", "weights"]):
        nome_m = f"{'Manhattan (p=1)' if p_val == 1 else 'Euclidiana (p=2)'} · {w_val}"
        cor_k = COR if (p_val == 1 and w_val == "distance") else (COR_DESTAQUE if p_val == 1 else COR_CLARA)
        ls_k = "-" if w_val == "distance" else "--"
        ax.plot(
            grp["n_neighbors"],
            grp["cv_rmse"],
            marker="o",
            lw=2.2 if (p_val == 1 and w_val == "distance") else 1.5,
            ls=ls_k,
            color=cor_k,
            label=nome_m,
        )
    best_k = knn_curve_df.loc[knn_curve_df["cv_rmse"].idxmin()]
    ax.scatter([best_k["n_neighbors"]], [best_k["cv_rmse"]], color=COR_VERDE, s=65, zorder=5)
    ax.set_xlabel("Número de vizinhos (n_neighbors)")
    ax.set_ylabel("RMSE 5-Fold CV (dias)")
    ax.set_title("(c) KNN: número de vizinhos × métrica (Manhattan vs. Euclidiana)")
    ax.legend(loc="upper right", fontsize=8)

    # --- (d) Random Forest: candidatos do RandomizedSearchCV ---
    ax = axes[1, 1]
    cv_rf = pd.DataFrame(rf_search.cv_results_).sort_values("rank_test_rmse").reset_index(drop=True)
    labels_rf = [
        f"#{i+1}: mf={r['param_max_features']}, leaf={r['param_min_samples_leaf']}, "
        f"d={r['param_max_depth']}, n={r['param_n_estimators']}"
        for i, r in cv_rf.iterrows()
    ]
    vals = (-cv_rf["mean_test_rmse"]).to_numpy()
    cores_rf = [COR_VERDE if i == 0 else COR for i in range(len(cv_rf))]
    # Plota #1 na parte inferior (ou superior) de forma consistente entre barra, ytick e texto
    y_pos = np.arange(len(cv_rf))
    ax.barh(y_pos, vals, color=cores_rf, edgecolor=FUNDO, height=0.72)
    ax.set_yticks(y_pos, labels_rf, fontsize=7.8)
    ax.invert_yaxis()  # #1 no topo!
    ax.set_xlim(min(vals) - 0.06, max(vals) + 0.045)
    for i, v in enumerate(vals):
        ax.text(
            v + 0.003,
            i,
            fmt_dec(v, 3),
            va="center",
            fontsize=8,
            fontweight="bold" if i == 0 else "normal",
            color=TEXTO,
        )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE 5-Fold CV (dias)")
    ax.set_title("(d) Random Forest: ranking das combinações (RandomizedSearchCV)")

    fig.suptitle(
        "Processo de Otimização de Hiperparâmetros (Validação Cruzada 5-Fold no Treino)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    salvar(fig, "3_curvas_tuning_hiperparametros")


def plot_4_estabilidade_folds(registros_cv: list[dict]) -> None:
    """Mostra a distribuição do RMSE e R² nos 5 folds para os modelos otimizados."""
    tunados = [r for r in registros_cv if r["estagio"] == "Tuned"]
    nomes = [r["modelo"] for r in tunados]
    rmses_folds = [r["fold_rmses"] for r in tunados]
    r2s_folds = [r["fold_r2s"] for r in tunados]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    box_kw = dict(
        patch_artist=True,
        widths=0.45,
        boxprops={"facecolor": "#b7d3f6", "edgecolor": COR, "lw": 1.3},
        medianprops={"color": COR_DESTAQUE, "lw": 2},
        whiskerprops={"color": COR},
        capprops={"color": COR},
    )

    # RMSE por fold
    ax = axes[0]
    ax.boxplot(rmses_folds, tick_labels=nomes, **box_kw)
    for i, vals in enumerate(rmses_folds, start=1):
        ax.scatter([i] * len(vals), vals, color=COR, s=28, zorder=4, alpha=0.85)
        med = np.mean(vals)
        std = np.std(vals)
        ax.text(
            i,
            max(vals) + 0.018,
            f"{fmt_dec(med, 2)} ± {fmt_dec(std, 2)}",
            ha="center",
            fontsize=8,
            color=TEXTO_2,
        )
    ax.set_ylabel("RMSE em cada fold (dias)")
    ax.set_title("Estabilidade do RMSE nos 5 Folds (média ± desvio)")
    ax.set_ylim(5.45, 6.02)

    # R² por fold
    ax = axes[1]
    ax.boxplot(r2s_folds, tick_labels=nomes, **box_kw)
    for i, vals in enumerate(r2s_folds, start=1):
        ax.scatter([i] * len(vals), vals, color=COR, s=28, zorder=4, alpha=0.85)
        med = np.mean(vals)
        std = np.std(vals)
        ax.text(
            i,
            max(vals) + 0.004,
            f"{fmt_dec(med, 3)} ± {fmt_dec(std, 3)}",
            ha="center",
            fontsize=8,
            color=TEXTO_2,
        )
    ax.set_ylabel("Coeficiente de Determinação (R²) em cada fold")
    ax.set_title("Estabilidade do R² nos 5 Folds (média ± desvio)")
    ax.set_ylim(0.365, 0.468)

    salvar(fig, "4_estabilidade_folds_cv")


def plot_5_tradeoff_tempo_erro(df_cv: pd.DataFrame, df_tuning_summary: pd.DataFrame) -> None:
    """Trade-off entre tempo de treino/ajuste e erro de validação (RMSE)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))

    # Painel 1: Tempo de avaliação 5-fold CV do modelo ajustado vs RMSE
    ax = axes[0]
    sub = df_cv[df_cv["estagio"].isin(["Baseline", "Tuned"])].copy()
    sub = sub[sub["modelo"] != "Prazo Estimado Olist"]
    offsets = {
        "Dummy (Mediana)": (8, -6),
        "Ridge": (8, 8),
        "Decision Tree": (-12, 12),
        "KNN": (10, 6),
        "Random Forest": (10, -14),
    }
    for _, r in sub.iterrows():
        cor_p = COR_DESTAQUE if r["estagio"] == "Baseline" else (
            COR_VERDE if r["modelo"] == "Random Forest" else COR
        )
        t_val = max(float(r["tempo_s"]), 0.08)
        ax.scatter(t_val, r["cv_rmse_mean"], s=95, color=cor_p, zorder=4)
        dx, dy = offsets.get(r["modelo"], (8, 4))
        ax.annotate(
            f"{r['modelo']}\n({fmt_dec(r['cv_rmse_mean'])} d | {fmt_dec(r['tempo_s'], 1)}s)",
            (t_val, r["cv_rmse_mean"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.2,
            color=TEXTO,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.04, 180)
    ax.set_ylim(5.4, 8.1)
    ax.set_xlabel("Tempo do 5-Fold CV do modelo (segundos, escala log)")
    ax.set_ylabel("RMSE na Validação Cruzada (dias)")
    ax.set_title("Trade-off: Tempo de Execução (5-Fold CV) × RMSE")
    ax.grid(True, which="both", axis="both", alpha=0.5)

    # Painel 2: Tempo total da busca de hiperparâmetros
    ax = axes[1]
    y_pos = np.arange(len(df_tuning_summary))
    tempos = df_tuning_summary["tempo_busca_s"].to_numpy()
    modelos = df_tuning_summary["modelo"].to_list()
    metodos = df_tuning_summary["metodo_busca"].to_list()
    n_fits = df_tuning_summary["total_fits"].to_list()
    ax.barh(y_pos, tempos, color=COR, edgecolor=FUNDO, height=0.6)
    ax.set_yticks(y_pos, [f"{m}\n({met})" for m, met in zip(modelos, metodos)], fontsize=8.5)
    for i, (t_s, nf) in enumerate(zip(tempos, n_fits)):
        ax.text(
            t_s + max(tempos) * 0.02,
            i,
            f"{fmt_dec(t_s, 1)}s ({nf} fits)",
            va="center",
            fontsize=8.2,
            color=TEXTO,
        )
    ax.set_xlim(0, max(tempos) * 1.28)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tempo total de otimização de hiperparâmetros (s)")
    ax.set_title("Custo Computacional da Busca de Hiperparâmetros")

    salvar(fig, "5_tradeoff_tempo_vs_erro")


def plot_6_importancia_features(rf_model: RandomForestRegressor, cols: list[str]) -> pd.DataFrame:
    """Calcula e plota a importância das features do Random Forest otimizado."""
    imp = pd.DataFrame({"coluna": cols, "importancia": rf_model.feature_importances_})

    # Mapeia cada coluna pós-OneHot de volta para a feature original do Tópico 2
    prefixos_onehot = [
        "cliente_estado",
        "vendedor_estado",
        "tipo_pagamento",
        "compra_dia_semana",
        "compra_mes",
    ]

    def mapear_origem(c: str) -> str:
        for p in prefixos_onehot:
            if c.startswith(p + "_"):
                return p
        return c

    imp["feature_original"] = imp["coluna"].map(mapear_origem)
    por_feature = (
        imp.groupby("feature_original", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    por_feature["importancia_pct"] = (por_feature["importancia"] * 100).round(2)
    por_feature["importancia_acum_pct"] = por_feature["importancia_pct"].cumsum().round(2)

    rotulos_pt = {
        "distancia_media_km": "Distância média vend.→cliente (km)",
        "prazo_estimado_dias": "Prazo estimado pela Olist (dias)",
        "distancia_max_km": "Distância máxima vend.→cliente (km)",
        "cliente_lat": "Latitude do cliente",
        "cliente_lng": "Longitude do cliente",
        "frete_total": "Frete total (R$)",
        "prazo_postagem_dias": "Prazo limite de postagem (dias)",
        "peso_total_g": "Peso total do produto (g)",
        "volume_total_cm3": "Volume total do produto (cm³)",
        "preco_total": "Preço total do pedido (R$)",
        "razao_frete_preco": "Razão frete / preço",
        "vendedor_lat": "Latitude do vendedor",
        "vendedor_lng": "Longitude do vendedor",
        "categoria_produto": "Categoria do produto (TargetEncoder)",
        "compra_mes": "Mês da compra (One-Hot)",
        "cliente_estado": "UF do cliente (One-Hot)",
        "compra_hora": "Hora da compra",
        "compra_dia_semana": "Dia da semana da compra (One-Hot)",
        "mesmo_estado": "Vendedor e cliente no mesmo estado",
        "parcelas_max": "Parcelas (máx.)",
        "tipo_pagamento": "Tipo de pagamento (One-Hot)",
        "vendedor_estado": "UF do vendedor (One-Hot)",
    }
    por_feature["descricao"] = por_feature["feature_original"].map(
        lambda c: rotulos_pt.get(c, c)
    )

    top = por_feature.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.barh(top["descricao"], top["importancia_pct"], color=COR, edgecolor=FUNDO, height=0.72)
    for i, v in enumerate(top["importancia_pct"]):
        ax.text(v + 0.3, i, f"{fmt_dec(v, 1)}%", va="center", fontsize=8.5, color=TEXTO)
    ax.set_xlim(0, max(top["importancia_pct"]) * 1.14)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importância relativa na redução de impureza (MDI, %)")
    ax.set_title("Top 15 Features Mais Importantes — Random Forest Otimizado")
    salvar(fig, "6_importancia_features_rf")

    return por_feature


def plot_7_real_vs_previsto(predicoes_df: pd.DataFrame) -> None:
    """Compara Real vs. Previsto (hexbin) entre a promessa da Olist e os modelos."""
    from matplotlib.colors import LogNorm
    from eda import CMAP_SEQ

    pares = [
        ("pred_prazo_olist", "Baseline: Prazo Estimado Olist (Atual)"),
        ("pred_ridge", "Linear: Ridge (Otimizado)"),
        ("pred_decision_tree", "Árvore: Decision Tree (Otimizada)"),
        ("pred_random_forest", "Ensemble: Random Forest (Otimizado)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=True, sharey=True)
    y_true = predicoes_df["y_true"].to_numpy()
    lim = 43.0

    for ax, (col, titulo) in zip(axes.flat, pares):
        y_pred = predicoes_df[col].to_numpy()
        # filtra apenas para visualização dentro da janela [0, 43] dias
        m = (y_pred >= 0) & (y_pred <= lim)
        hb = ax.hexbin(
            y_true[m],
            y_pred[m],
            gridsize=45,
            extent=(0, lim, 0, lim),
            cmap=CMAP_SEQ,
            mincnt=1,
            norm=LogNorm(),
            linewidths=0,
        )
        ax.plot([0, lim], [0, lim], color=COR_DESTAQUE, ls="--", lw=1.8, label="Acerto perfeito (Previsto = Real)")
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_title(f"{titulo}\nMAE = {fmt_dec(mae)} d | RMSE = {fmt_dec(rmse)} d", fontsize=10)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(False)
        ax.legend(loc="upper left", fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel("Tempo de entrega REAL (dias)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Tempo de entrega PREVISTO (dias)")

    cb = fig.colorbar(hb, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("Quantidade de pedidos (escala log)")
    fig.suptitle(
        "Tempo Real vs. Previsto no Conjunto de Teste (19.020 pedidos)",
        x=0.04,
        ha="left",
        fontweight="bold",
    )
    fig.savefig(OUT_DIR / "7_real_vs_previsto_teste.png", bbox_inches="tight")
    plt.close(fig)
    print("  figura: 7_real_vs_previsto_teste.png")


def plot_8_erros_e_tolerancia(predicoes_df: pd.DataFrame) -> None:
    """Mostra a distribuição dos erros (Previsto - Real) e a taxa de acerto por margem."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    y_true = predicoes_df["y_true"].to_numpy()

    # --- Painel 1: Histograma do erro (Previsto - Real) ---
    ax = axes[0]
    erro_olist = predicoes_df["pred_prazo_olist"].to_numpy() - y_true
    erro_rf = predicoes_df["pred_random_forest"].to_numpy() - y_true
    bins = np.linspace(-25, 35, 65)
    ax.hist(
        erro_olist,
        bins=bins,
        color=COR_DESTAQUE,
        alpha=0.45,
        edgecolor=FUNDO,
        label=f"Prazo Atual Olist (viés médio: +{fmt_dec(erro_olist.mean(), 1)} d)",
    )
    ax.hist(
        erro_rf,
        bins=bins,
        color=COR,
        alpha=0.80,
        edgecolor=FUNDO,
        label=f"Random Forest Otimizado (viés médio: +{fmt_dec(erro_rf.mean(), 2)} d)",
    )
    ax.axvline(0, color=TEXTO, ls="--", lw=1.5, label="Erro zero (previsão exata)")
    ax.set_xlabel("Erro da previsão = Previsto − Real (dias)\n(< 0: atrasou | > 0: chegou antes do previsto)")
    ax.set_ylabel("Quantidade de pedidos no teste")
    ax.set_title("Distribuição dos Erros: Promessa da Olist vs. Random Forest")
    ax.legend(loc="upper right", fontsize=8.2)

    # --- Painel 2: Curva de tolerância (% de pedidos dentro de ±K dias) ---
    ax = axes[1]
    margens = np.arange(1, 11)
    series = [
        ("pred_random_forest", "Random Forest (Otimizado)", COR_VERDE, "-", 2.4),
        ("pred_knn", "KNN (Otimizado)", COR, "--", 1.7),
        ("pred_decision_tree", "Decision Tree (Otimizada)", "#3987e5", "-.", 1.6),
        ("pred_ridge", "Ridge (Otimizado)", COR_CLARA, ":", 1.8),
        ("pred_dummy_mediana", "Baseline Dummy (Mediana)", TEXTO_2, "--", 1.3),
        ("pred_prazo_olist", "Baseline Prazo Olist (Atual)", COR_DESTAQUE, "-", 2.0),
    ]
    for col, nome, cor_s, ls_s, lw_s in series:
        err_abs = np.abs(predicoes_df[col].to_numpy() - y_true)
        pcts = [(err_abs <= m).mean() * 100 for m in margens]
        ax.plot(margens, pcts, marker="o", ms=4.5, color=cor_s, ls=ls_s, lw=lw_s, label=nome)

    ax.axvline(3, color=GRADE, ls="--", lw=1)
    ax.axvline(5, color=GRADE, ls="--", lw=1)
    ax.set_xticks(margens, [f"±{m}d" for m in margens])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Margem de tolerância do erro absoluto (± dias)")
    ax.set_ylabel("% de pedidos acertados dentro da margem")
    ax.set_title("Taxa de Acerto por Margem de Tolerância (Conjunto de Teste)")
    ax.legend(loc="lower right", fontsize=8)

    salvar(fig, "8_erros_e_tolerancia_teste")


# --------------------------------------------------------------------------- #
# Execução principal
# --------------------------------------------------------------------------- #
def main(proc_dir: Path = PROCESSED_DIR, seed: int = 42) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("1. Carregando dados processados...")
    X_tr_df, X_te_df, y_tr_s, y_te_s, prazo_tr, prazo_te = carregar_dados(proc_dir, seed=seed)
    cols = list(X_tr_df.columns)
    X_tr = X_tr_df.to_numpy(dtype=np.float64)
    X_te = X_te_df.to_numpy(dtype=np.float64)
    y_tr = y_tr_s.to_numpy(dtype=np.float64)
    y_te = y_te_s.to_numpy(dtype=np.float64)
    print(f"   Treino: {X_tr.shape} | Teste: {X_te.shape}")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    registros_cv: list[dict] = []

    # ----------------------------------------------------------------------- #
    # 2. Baselines (Negócio e Estatístico)
    # ----------------------------------------------------------------------- #
    print("2. Avaliando Baselines em 5-Fold CV...")
    registros_cv.append(avaliar_baseline_olist(y_tr_s, prazo_tr, cv))
    registros_cv.append(
        avaliar_cv(
            DummyRegressor(strategy="median"),
            X_tr,
            y_tr,
            cv,
            modelo="Dummy (Mediana)",
            familia="Baseline (Estatístico)",
            estagio="Baseline",
        )
    )

    # ----------------------------------------------------------------------- #
    # 3. Modelos Default (antes do ajuste de hiperparâmetros)
    # ----------------------------------------------------------------------- #
    print("3. Avaliando os 4 modelos com hiperparâmetros default (5-Fold CV)...")
    defaults = [
        ("Ridge", "Linear (L2)", Ridge(random_state=seed), -1),
        ("Decision Tree", "Árvore de Decisão", DecisionTreeRegressor(random_state=seed), -1),
        ("KNN", "Baseado em Instâncias", KNeighborsRegressor(n_jobs=-1), 1),
        (
            "Random Forest",
            "Ensemble (Bagging)",
            RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            1,
        ),
    ]
    for nome, fam, est, nj in defaults:
        r = avaliar_cv(est, X_tr, y_tr, cv, modelo=nome, familia=fam, estagio="Default", n_jobs_cv=nj)
        registros_cv.append(r)
        print(
            f"   [Default] {nome:15s} -> CV RMSE: {r['cv_rmse_mean']:.4f} ± {r['cv_rmse_std']:.4f} | "
            f"MAE: {r['cv_mae_mean']:.4f} | R²: {r['cv_r2_mean']:.4f} ({r['tempo_s']:.1f}s)"
        )

    # ----------------------------------------------------------------------- #
    # 4. Ajuste de Hiperparâmetros (GridSearchCV e RandomizedSearchCV)
    # ----------------------------------------------------------------------- #
    print("4. Executando ajuste de hiperparâmetros (5-Fold CV)...")
    resumo_tuning: list[dict] = []

    # --- 4.1 Ridge (GridSearchCV + Experimento log1p da EDA) ---
    print("   4.1 Ridge (GridSearchCV + teste log1p)...")
    df_ridge_log = experimento_ridge_log(X_tr, y_tr, cv)
    salvar_tabela(df_ridge_log, "3_ridge_alvo_original_vs_log", index=False)

    ridge_grid_params = {"alpha": [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]}
    t0 = time.perf_counter()
    ridge_search = GridSearchCV(
        Ridge(random_state=seed),
        param_grid=ridge_grid_params,
        scoring=SCORING,
        refit="rmse",
        cv=cv,
        return_train_score=True,
        n_jobs=-1,
    )
    ridge_search.fit(X_tr, y_tr)
    t_ridge = time.perf_counter() - t0
    print(f"       Melhor Ridge: {ridge_search.best_params_} | RMSE={-ridge_search.best_score_:.4f} ({t_ridge:.1f}s)")

    # --- 4.2 Decision Tree (GridSearchCV) ---
    print("   4.2 Decision Tree (GridSearchCV)...")
    dt_grid_params = {
        "max_depth": [4, 6, 8, 10, 12, 14, 16, 20, None],
        "min_samples_leaf": [1, 20, 50, 100, 200],
        "min_samples_split": [2, 100],
    }
    t0 = time.perf_counter()
    dt_search = GridSearchCV(
        DecisionTreeRegressor(random_state=seed),
        param_grid=dt_grid_params,
        scoring=SCORING,
        refit="rmse",
        cv=cv,
        return_train_score=True,
        n_jobs=-1,
    )
    dt_search.fit(X_tr, y_tr)
    t_dt = time.perf_counter() - t0
    print(f"       Melhor Decision Tree: {dt_search.best_params_} | RMSE={-dt_search.best_score_:.4f} ({t_dt:.1f}s)")

    # --- 4.3 KNN (RandomizedSearchCV) ---
    print("   4.3 KNN (RandomizedSearchCV)...")
    X_tr_f32 = X_tr.astype(np.float32)
    knn_param_dist = {
        "n_neighbors": [5, 15, 25, 35, 51],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    t0 = time.perf_counter()
    knn_search = RandomizedSearchCV(
        KNeighborsRegressor(n_jobs=-1),
        param_distributions=knn_param_dist,
        n_iter=12,
        scoring=SCORING,
        refit="rmse",
        cv=cv,
        random_state=seed,
        return_train_score=False,
        n_jobs=1,
    )
    knn_search.fit(X_tr_f32, y_tr)
    t_knn = time.perf_counter() - t0
    print(f"       Melhor KNN: {knn_search.best_params_} | RMSE={-knn_search.best_score_:.4f} ({t_knn:.1f}s)")

    cv_knn_df = pd.DataFrame(knn_search.cv_results_)
    knn_curve_df = pd.DataFrame(
        {
            "n_neighbors": cv_knn_df["param_n_neighbors"].astype(int),
            "weights": cv_knn_df["param_weights"].astype(str),
            "p": cv_knn_df["param_p"].astype(int),
            "cv_rmse": -cv_knn_df["mean_test_rmse"].astype(float),
        }
    ).sort_values(["p", "weights", "n_neighbors"])

    # --- 4.4 Random Forest (RandomizedSearchCV) ---
    print("   4.4 Random Forest (RandomizedSearchCV)...")
    rf_param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [12, 16, 20, 24, None],
        "min_samples_leaf": [2, 5, 10, 20],
        "min_samples_split": [2, 10, 20],
        "max_features": ["sqrt", 0.3, 0.5, 0.7],
    }
    t0 = time.perf_counter()
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=10,
        scoring=SCORING,
        refit="rmse",
        cv=cv,
        random_state=seed,
        return_train_score=False,
        n_jobs=1,
    )
    rf_search.fit(X_tr_f32, y_tr)
    t_rf = time.perf_counter() - t0
    print(f"       Melhor Random Forest: {rf_search.best_params_} | RMSE={-rf_search.best_score_:.4f} ({t_rf:.1f}s)")

    # ----------------------------------------------------------------------- #
    # 5. Avaliação padronizada (com train score e folds) dos 4 modelos Tuned
    # ----------------------------------------------------------------------- #
    print("5. Consolidando métricas 5-Fold CV dos modelos otimizados...")
    tuned_models = [
        ("Ridge", "Linear (L2)", ridge_search.best_estimator_, ridge_search, "GridSearchCV", t_ridge),
        (
            "Decision Tree",
            "Árvore de Decisão",
            dt_search.best_estimator_,
            dt_search,
            "GridSearchCV",
            t_dt,
        ),
        (
            "KNN",
            "Baseado em Instâncias",
            knn_search.best_estimator_,
            knn_search,
            "RandomizedSearchCV",
            t_knn,
        ),
        (
            "Random Forest",
            "Ensemble (Bagging)",
            rf_search.best_estimator_,
            rf_search,
            "RandomizedSearchCV",
            t_rf,
        ),
    ]

    for nome, fam, best_est, search_obj, metodo, t_busca in tuned_models:
        idx = search_obj.best_index_
        cv_res = search_obj.cv_results_
        rmses = [-float(cv_res[f"split{k}_test_rmse"][idx]) for k in range(cv.get_n_splits())]
        maes = [-float(cv_res[f"split{k}_test_mae"][idx]) for k in range(cv.get_n_splits())]
        r2s = [float(cv_res[f"split{k}_test_r2"][idx]) for k in range(cv.get_n_splits())]
        t_cv_est = float(cv_res["mean_fit_time"][idx] + cv_res["mean_score_time"][idx]) * cv.get_n_splits()

        if "mean_train_rmse" in cv_res:
            tr_rmse = float(-cv_res["mean_train_rmse"][idx])
            tr_r2 = float(cv_res["mean_train_r2"][idx])
        else:
            X_use = X_tr_f32 if nome in ("KNN", "Random Forest") else X_tr
            pred_tr = best_est.predict(X_use)
            tr_rmse = float(root_mean_squared_error(y_tr, pred_tr))
            tr_r2 = float(r2_score(y_tr, pred_tr))

        r_tun = {
            "modelo": nome,
            "familia": fam,
            "estagio": "Tuned",
            "train_rmse_mean": round(tr_rmse, 4),
            "cv_rmse_mean": round(float(np.mean(rmses)), 4),
            "cv_rmse_std": round(float(np.std(rmses)), 4),
            "cv_mae_mean": round(float(np.mean(maes)), 4),
            "cv_mae_std": round(float(np.std(maes)), 4),
            "train_r2_mean": round(tr_r2, 4),
            "cv_r2_mean": round(float(np.mean(r2s)), 4),
            "cv_r2_std": round(float(np.std(r2s)), 4),
            "tempo_s": round(t_cv_est, 2),
            "fold_rmses": rmses,
            "fold_r2s": r2s,
        }
        registros_cv.append(r_tun)

        r_def = next(x for x in registros_cv if x["modelo"] == nome and x["estagio"] == "Default")
        n_cand = len(search_obj.cv_results_["params"])
        resumo_tuning.append(
            {
                "modelo": nome,
                "familia": fam,
                "metodo_busca": metodo,
                "candidatos_testados": n_cand,
                "total_fits": n_cand * cv.get_n_splits(),
                "melhores_hiperparametros": json.dumps(search_obj.best_params_, ensure_ascii=False),
                "rmse_cv_default": r_def["cv_rmse_mean"],
                "rmse_cv_tuned": r_tun["cv_rmse_mean"],
                "ganho_rmse_dias": round(r_def["cv_rmse_mean"] - r_tun["cv_rmse_mean"], 4),
                "mae_cv_default": r_def["cv_mae_mean"],
                "mae_cv_tuned": r_tun["cv_mae_mean"],
                "r2_cv_default": r_def["cv_r2_mean"],
                "r2_cv_tuned": r_tun["cv_r2_mean"],
                "tempo_busca_s": round(t_busca, 2),
            }
        )

    # Salva tabela 1 (todos os resultados de CV) e tabela 2 (resumo do tuning)
    df_cv = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("fold_")} for r in registros_cv]
    )
    salvar_tabela(df_cv, "1_baselines_e_modelos_cv", index=False)

    df_tuning = pd.DataFrame(resumo_tuning)
    salvar_tabela(df_tuning, "2_espaco_busca_e_melhores_parametros", index=False)

    # ----------------------------------------------------------------------- #
    # 6. Avaliação preliminar no conjunto de Teste (ponte p/ Acompanhamento e Tópico 5)
    # ----------------------------------------------------------------------- #
    print("6. Calculando resultados preliminares no conjunto de teste (19.020 pedidos)...")
    linhas_teste = []
    predicoes_df = pd.DataFrame({"y_true": y_te, "pred_prazo_olist": prazo_te.to_numpy()})

    # Baseline 1: Prazo Olist
    linhas_teste.append(
        metricas_teste(
            "Prazo Estimado Olist",
            "Baseline (Negócio)",
            "Baseline",
            y_te,
            prazo_te.to_numpy(),
        )
    )

    # Baseline 2: Dummy Mediana
    dummy_est = DummyRegressor(strategy="median").fit(X_tr, y_tr)
    pred_dummy = dummy_est.predict(X_te)
    predicoes_df["pred_dummy_mediana"] = pred_dummy
    linhas_teste.append(
        metricas_teste("Dummy (Mediana)", "Baseline (Estatístico)", "Baseline", y_te, pred_dummy)
    )

    # 4 Modelos Tuned no Teste + salvamento com joblib
    arquivos_modelos = {
        "Ridge": "ridge_tuned.joblib",
        "Decision Tree": "decision_tree_tuned.joblib",
        "KNN": "knn_tuned.joblib",
        "Random Forest": "random_forest_tuned.joblib",
    }
    for nome, fam, best_est, _, _, _ in tuned_models:
        X_te_use = X_te.astype(np.float32) if nome in ("KNN", "Random Forest") else X_te
        y_pred = best_est.predict(X_te_use)
        col_pred = "pred_" + nome.lower().replace(" ", "_")
        predicoes_df[col_pred] = np.round(y_pred, 4)
        linhas_teste.append(metricas_teste(nome, fam, "Tuned", y_te, y_pred))
        joblib.dump(best_est, MODELS_DIR / arquivos_modelos[nome])

    df_teste = pd.DataFrame(linhas_teste).sort_values("rmse_teste").reset_index(drop=True)
    salvar_tabela(df_teste, "4_resultados_parciais_teste", index=False)
    predicoes_df.to_csv(PROCESSED_DIR / "predicoes_teste.csv", index=False)
    print("   Predições no teste salvas em data/processed/predicoes_teste.csv")

    # ----------------------------------------------------------------------- #
    # 7. Geração das figuras para os slides
    # ----------------------------------------------------------------------- #
    print("7. Gerando figuras em docs/figures/04_treinamento/...")
    plot_1_default_vs_tuned(df_cv)
    plot_2_overfitting_treino_vs_cv(df_cv)
    plot_3_curvas_tuning(df_ridge_log, dt_search, knn_curve_df, rf_search)
    plot_4_estabilidade_folds(registros_cv)
    plot_5_tradeoff_tempo_erro(df_cv, df_tuning)
    df_imp = plot_6_importancia_features(rf_search.best_estimator_, cols)
    salvar_tabela(df_imp, "5_importancia_features", index=False)
    plot_7_real_vs_previsto(predicoes_df)
    plot_8_erros_e_tolerancia(predicoes_df)

    print("\nResumo final (Validação Cruzada 5-Fold):")
    print(df_cv[["modelo", "familia", "estagio", "train_rmse_mean", "cv_rmse_mean", "cv_mae_mean", "cv_r2_mean"]].to_string(index=False))
    print("\nResumo preliminar (Conjunto de Teste):")
    print(df_teste.to_string(index=False))
    print(f"\nConcluído! Figuras e tabelas em {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proc-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.proc_dir, a.seed)
