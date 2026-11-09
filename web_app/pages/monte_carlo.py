import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    METALS_CONFIG,
    METHODS,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ACCENT,
    generate_synthetic_data,
    apply_censoring,
    compute_statistics,
    compute_bias_rmse,
    get_method_name,
    metric_card,
    get_plotly_layout,
)


@st.cache_data(show_spinner=False)
def _run_monte_carlo(
    metal_key: str,
    n: int,
    censor_pct: int,
    n_iter: int,
    base_seed: int,
    method_keys: tuple,
):
    """
    Монте-Карло: n_iter итераций с разными seed.
    Возвращает DataFrame со строками (iter, method, bias_mean, bias_median,
    bias_std, rmse).
    """
    rows = []

    for i in range(n_iter):
        seed = base_seed + i
        raw = generate_synthetic_data(metal_key, n=n, seed=seed)
        true_stats = compute_statistics(raw)
        censored, lod, actual_pct = apply_censoring(
            raw, target_pct=censor_pct / 100.0
        )

        for mkey in method_keys:
            func = METHODS[mkey]["func"]
            try:
                processed = func(censored, lod)
                stats = compute_statistics(processed)
                br = compute_bias_rmse(true_stats, stats)
                rows.append(
                    {
                        "iter": i,
                        "method": mkey,
                        "actual_pct": actual_pct,
                        "bias_mean": br["bias_mean"],
                        "bias_median": br["bias_median"],
                        "bias_std": br["bias_std"],
                        "rmse": br["rmse"],
                    }
                )
            except Exception:
                rows.append(
                    {
                        "iter": i,
                        "method": mkey,
                        "actual_pct": actual_pct,
                        "bias_mean": np.nan,
                        "bias_median": np.nan,
                        "bias_std": np.nan,
                        "rmse": np.nan,
                    }
                )

    return pd.DataFrame(rows)


def render_monte_carlo(t: dict):
    """Монте-Карло симуляция для статистической валидации."""

    st.markdown(
        f"## 🎲 {t.get('mc_title', 'Монте-Карло симуляция')}"
    )
    st.markdown(
        t.get(
            "mc_desc",
            "Многократный прогон с разными случайными seed позволяет оценить "
            "**стабильность** и **статистическую значимость** результатов. "
            "Каждая итерация генерирует новые данные, применяет цензурирование "
            "и сравнивает методы.",
        )
    )

    st.markdown("---")

    # ── Управление ──
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        metal_keys = list(METALS_CONFIG.keys())
        metal_labels = [
            f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})" for k in metal_keys
        ]
        sel = st.selectbox(
            t.get("select_metal", "Металл"),
            metal_labels,
            index=0,
            key="mc_page_metal",
        )
        metal_key = metal_keys[metal_labels.index(sel)]
        meta = METALS_CONFIG[metal_key]

    with c2:
        censor_pct = st.slider(
            t.get("censor_pct", "Доля цензурирования (%)"),
            5, 70, 30, 5,
            key="mc_page_pct",
        )

    with c3:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            [100, 250, 500, 1000],
            index=2,
            key="mc_page_n",
        )

    with c4:
        n_iter = st.selectbox(
            t.get("n_iterations", "Итераций"),
            [10, 25, 50, 100, 200],
            index=2,
            key="mc_page_iter",
        )

    base_seed = st.number_input(
        t.get("base_seed", "Базовый seed"),
        0, 99999, 42, 1,
        key="mc_page_seed",
    )

    # ── Прогон ──
    method_keys = tuple(METHODS.keys())

    with st.spinner(
        t.get("mc_running", "Монте-Карло выполняется... Это может занять время.")
    ):
        df_mc = _run_monte_carlo(
            metal_key, n_samples, censor_pct, n_iter, base_seed, method_keys
        )

    if df_mc.empty:
        st.error(t.get("no_data", "Нет данных."))
        return

    # ── Агрегация ──
    metric_cols = ["bias_mean", "bias_median", "bias_std", "rmse"]
    agg = (
        df_mc.groupby("method")[metric_cols]
        .agg(["mean", "std", "median", lambda x: x.quantile(0.025),
              lambda x: x.quantile(0.975)])
        .reset_index()
    )
    # Упрощаем multi-index колонки
    agg.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in agg.columns
    ]

    # Более читаемая агрегация
    summary_rows = []
    for mkey in method_keys:
        sub = df_mc[df_mc["method"] == mkey].dropna(subset=["rmse"])
        if sub.empty:
            continue
        row = {"method": mkey}
        for mc in metric_cols:
            vals = sub[mc].values
            row[f"{mc}_mean"] = np.mean(vals)
            row[f"{mc}_std"] = np.std(vals)
            row[f"{mc}_ci_low"] = np.percentile(vals, 2.5)
            row[f"{mc}_ci_high"] = np.percentile(vals, 97.5)
            row[f"{mc}_median"] = np.median(vals)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # ── Ключевые метрики ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(
            t.get("iterations_done", "Итераций"),
            str(n_iter),
            "🔄",
            COLOR_PRIMARY,
        )
    with m2:
        best_mc = df_summary.loc[df_summary["rmse_mean"].idxmin()]
        metric_card(
            t.get("best_method", "Лучший метод"),
            f"{METHODS[best_mc['method']]['icon']} "
            f"{get_method_name(best_mc['method'], t)}",
            "🏆",
            COLOR_SUCCESS,
        )
    with m3:
        metric_card(
            t.get("mean_rmse_best", "Ср. RMSE лидера"),
            f"{best_mc['rmse_mean']:.4g}",
            "📐",
            COLOR_ACCENT,
        )
    with m4:
        metric_card(
            t.get("stability_best", "σ RMSE лидера"),
            f"{best_mc['rmse_std']:.4g}",
            "📊",
            COLOR_WARNING,
        )

    st.markdown("---")

    # ── 1. Box plot: распределение bias_mean по методам ──
    st.markdown(
        f"### 📦 {t.get('mc_box_bias', 'Распределение смещения среднего')}"
    )

    fig_box_bias = go.Figure()
    for mkey in method_keys:
        sub = df_mc[df_mc["method"] == mkey].dropna(subset=["bias_mean"])
        if sub.empty:
            continue
        fig_box_bias.add_trace(
            go.Box(
                y=sub["bias_mean"],
                name=f"{METHODS[mkey]['icon']} {get_method_name(mkey, t)}",
                marker_color=METHODS[mkey]["color"],
                boxmean=True,
                hoverinfo="y+name",
            )
        )

    fig_box_bias.add_hline(
        y=0, line_width=2, line_color="#333333", line_dash="solid"
    )

    layout_bb = get_plotly_layout(
        title=f"{meta['name_ru']} — "
        + t.get("bias_distribution", "распределение смещения среднего"),
        yaxis_title=t.get("bias_mean", "Смещение среднего"),
    )
    fig_box_bias.update_layout(**layout_bb, height=450)
    fig_box_bias.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
        )
    )
    st.plotly_chart(fig_box_bias, use_container_width=True)

    st.markdown("---")

    # ── 2. Box plot: распределение RMSE ──
    st.markdown(
        f"### 📦 {t.get('mc_box_rmse', 'Распределение RMSE')}"
    )

    fig_box_rmse = go.Figure()
    for mkey in method_keys:
        sub = df_mc[df_mc["method"] == mkey].dropna(subset=["rmse"])
        if sub.empty:
            continue
        fig_box_rmse.add_trace(
            go.Box(
                y=sub["rmse"],
                name=f"{METHODS[mkey]['icon']} {get_method_name(mkey, t)}",
                marker_color=METHODS[mkey]["color"],
                boxmean=True,
                hoverinfo="y+name",
            )
        )

    layout_br = get_plotly_layout(
        title=f"{meta['name_ru']} — " + t.get("rmse_distribution", "распределение RMSE"),
        yaxis_title="RMSE",
    )
    fig_box_rmse.update_layout(**layout_br, height=450)
    fig_box_rmse.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
        )
    )
    st.plotly_chart(fig_box_rmse, use_container_width=True)

    st.markdown("---")

    # ── 3. Violin plot: bias_median ──
    st.markdown(
        f"### 🎻 {t.get('mc_violin', 'Violin plot: смещение медианы')}"
    )

    fig_violin = go.Figure()
    for mkey in method_keys:
        sub = df_mc[df_mc["method"] == mkey].dropna(subset=["bias_median"])
        if sub.empty:
            continue
        fig_violin.add_trace(
            go.Violin(
                y=sub["bias_median"],
                name=f"{METHODS[mkey]['icon']} {get_method_name(mkey, t)}",
                line_color=METHODS[mkey]["color"],
                fillcolor=METHODS[mkey]["color"],
                opacity=0.5,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig_violin.add_hline(
        y=0, line_width=2, line_color="#333333", line_dash="solid"
    )

    layout_v = get_plotly_layout(
        title=f"{meta['name_ru']} — "
        + t.get("bias_median_dist", "распределение смещения медианы"),
        yaxis_title=t.get("bias_median", "Смещение медианы"),
    )
    fig_violin.update_layout(**layout_v, height=450)
    fig_violin.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
        )
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")

    # ── 4. Error bar chart: mean ± 95% CI для RMSE ──
    st.markdown(
        f"### 📊 {t.get('mc_ci_chart', 'RMSE: среднее ± 95% доверительный интервал')}"
    )

    if not df_summary.empty:
        df_ci = df_summary.sort_values("rmse_mean").copy()
        df_ci["label"] = df_ci["method"].apply(
            lambda mk: f"{METHODS[mk]['icon']} {get_method_name(mk, t)}"
        )
        df_ci["color"] = df_ci["method"].apply(lambda mk: METHODS[mk]["color"])

        fig_ci = go.Figure()
        fig_ci.add_trace(
            go.Bar(
                x=df_ci["label"],
                y=df_ci["rmse_mean"],
                marker_color=df_ci["color"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=df_ci["rmse_ci_high"] - df_ci["rmse_mean"],
                    arrayminus=df_ci["rmse_mean"] - df_ci["rmse_ci_low"],
                    color="#333333",
                    thickness=1.5,
                    width=6,
                ),
                text=[f"{v:.4g}" for v in df_ci["rmse_mean"]],
                textposition="outside",
            )
        )

        layout_ci = get_plotly_layout(
            title=t.get("rmse_ci_title", "RMSE с 95% доверительным интервалом"),
            xaxis_title=t.get("method", "Метод"),
            yaxis_title="RMSE",
        )
        fig_ci.update_layout(**layout_ci, height=420)
        st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown("---")

    # ── 5. Сводная таблица ──
    st.markdown(
        f"### 📋 {t.get('mc_summary_table', 'Сводная таблица Монте-Карло')}"
    )

    if not df_summary.empty:
        table_rows = []
        for _, row in df_summary.sort_values("rmse_mean").iterrows():
            mk = row["method"]
            table_rows.append(
                {
                    t.get("method", "Метод"): f"{METHODS[mk]['icon']} {get_method_name(mk, t)}",
                    t.get("bias_mean", "Смещ. средн.") + " (μ±σ)": (
                        f"{row['bias_mean_mean']:+.3f} ± {row['bias_mean_std']:.3f}"
                    ),
                    t.get("bias_median", "Смещ. медиан.") + " (μ±σ)": (
                        f"{row['bias_median_mean']:+.3f} ± {row['bias_median_std']:.3f}"
                    ),
                    "RMSE (μ±σ)": (
                        f"{row['rmse_mean']:.4g} ± {row['rmse_std']:.4g}"
                    ),
                    "RMSE 95% CI": (
                        f"[{row['rmse_ci_low']:.4g}, {row['rmse_ci_high']:.4g}]"
                    ),
                }
            )

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Пояснение ──
    st.info(
        t.get(
            "mc_note",
            "💡 **Интерпретация:** узкий box/violin означает **стабильный** метод "
            "(мало зависит от конкретной выборки). Метод с низкой медианой bias "
            "и узким разбросом — наиболее надёжный для данного металла и "
            "уровня цензурирования.",
        )
    )