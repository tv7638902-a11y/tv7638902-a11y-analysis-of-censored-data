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
    generate_synthetic_data,
    apply_censoring,
    compute_statistics,
    compute_bias_rmse,
    get_method_name,
    metric_card,
    get_plotly_layout,
)


@st.cache_data(show_spinner=False)
def _run_sweep(metal_key: str, n: int, seed: int, method_keys: tuple, pcts: tuple):
    """Кэшированный sweep по процентам цензурирования."""
    raw = generate_synthetic_data(metal_key, n=n, seed=seed)
    true_stats = compute_statistics(raw)

    rows = []
    for pct in pcts:
        censored, lod, actual_pct = apply_censoring(raw, target_pct=pct / 100.0)
        for mkey in method_keys:
            func = METHODS[mkey]["func"]
            try:
                processed = func(censored, lod)
                stats = compute_statistics(processed)
                br = compute_bias_rmse(true_stats, stats)
                rows.append(
                    {
                        "censor_pct": actual_pct,
                        "target_pct": pct,
                        "method": mkey,
                        "bias_mean": br["bias_mean"],
                        "bias_median": br["bias_median"],
                        "bias_std": br["bias_std"],
                        "rmse": br["rmse"],
                    }
                )
            except Exception:
                rows.append(
                    {
                        "censor_pct": actual_pct,
                        "target_pct": pct,
                        "method": mkey,
                        "bias_mean": np.nan,
                        "bias_median": np.nan,
                        "bias_std": np.nan,
                        "rmse": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def render_censoring_dependency(t: dict):
    """Зависимость точности методов от доли цензурирования."""

    st.markdown(
        f"## 📈 {t.get('cens_dep_title', 'Зависимость от % цензурирования')}"
    )
    st.markdown(
        t.get(
            "cens_dep_desc",
            "Как меняется точность каждого метода при увеличении доли "
            "цензурированных значений? Графики ниже показывают смещение "
            "и RMSE в зависимости от процента данных ниже LOD.",
        )
    )

    st.markdown("---")

    # ── Управление ──
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        metal_keys = list(METALS_CONFIG.keys())
        metal_labels = [
            f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})" for k in metal_keys
        ]
        sel = st.selectbox(
            t.get("select_metal", "Металл"),
            metal_labels,
            index=0,
            key="cd_metal",
        )
        metal_key = metal_keys[metal_labels.index(sel)]
        meta = METALS_CONFIG[metal_key]

    with c2:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            [100, 250, 500, 1000],
            index=2,
            key="cd_n",
        )

    with c3:
        seed = st.number_input(
            "Seed", 0, 9999, 42, 1, key="cd_seed"
        )

    # Multiselect методов
    all_mkeys = list(METHODS.keys())
    default_methods = ["zero", "lod_half", "kaplan_meier", "mle_lognormal", "ros"]
    selected_methods = st.multiselect(
        t.get("select_methods", "Методы для сравнения"),
        options=all_mkeys,
        default=[m for m in default_methods if m in all_mkeys],
        format_func=lambda mk: f"{METHODS[mk]['icon']} {get_method_name(mk, t)}",
        key="cd_methods",
    )

    if not selected_methods:
        st.warning(t.get("select_at_least_one", "Выберите хотя бы один метод."))
        return

    # ── Sweep ──
    pcts = tuple(range(5, 85, 5))

    with st.spinner(t.get("computing", "Вычисление...")):
        df_sweep = _run_sweep(
            metal_key, n_samples, seed, tuple(selected_methods), pcts
        )

    # ── Метрики ──
    worst_bias = df_sweep["bias_mean"].abs().max()
    best_method_row = df_sweep.loc[df_sweep["rmse"].idxmin()] if not df_sweep["rmse"].isna().all() else None

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card(
            t.get("pcts_tested", "Точек теста"),
            str(len(pcts)),
            "📊",
            COLOR_PRIMARY,
        )
    with m2:
        metric_card(
            t.get("max_bias", "Макс. |смещение|"),
            f"{worst_bias:.2%}" if not np.isnan(worst_bias) else "—",
            "⚠️",
            COLOR_DANGER,
        )
    with m3:
        if best_method_row is not None:
            bm = get_method_name(best_method_row["method"], t)
            metric_card(
                t.get("best_method", "Лучший метод"),
                bm,
                "🏆",
                COLOR_SUCCESS,
            )
        else:
            metric_card(t.get("best_method", "Лучший метод"), "—", "🏆", COLOR_SUCCESS)

    st.markdown("---")

    # ── Вспомогательная функция для линейных графиков ──
    def _line_chart(y_col, y_label, title, y_format=".2%"):
        fig = go.Figure()
        for mkey in selected_methods:
            sub = df_sweep[df_sweep["method"] == mkey].sort_values("censor_pct")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["censor_pct"],
                    y=sub[y_col],
                    mode="lines+markers",
                    name=f"{METHODS[mkey]['icon']} {get_method_name(mkey, t)}",
                    line=dict(color=METHODS[mkey]["color"], width=2.5),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"% ценз.: %{{x:.0f}}%<br>"
                        f"{y_label}: %{{y:{y_format}}}<extra></extra>"
                    ),
                )
            )

        # Нулевая линия для bias
        if "bias" in y_col:
            fig.add_hline(
                y=0, line_width=2, line_color="#333333", line_dash="solid"
            )

        layout = get_plotly_layout(
            title=title,
            xaxis_title=t.get("censor_pct_axis", "Доля цензурирования (%)"),
            yaxis_title=y_label,
        )
        fig.update_layout(**layout, height=420)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 1. Bias среднего ──
    st.markdown(
        f"### 📉 {t.get('bias_mean_vs_pct', 'Смещение среднего vs % цензурирования')}"
    )
    _line_chart(
        "bias_mean",
        t.get("bias_mean", "Смещение среднего"),
        f"{meta['name_ru']} — " + t.get("bias_mean_title", "смещение среднего"),
    )

    st.markdown("---")

    # ── 2. Bias медианы ──
    st.markdown(
        f"### 📉 {t.get('bias_median_vs_pct', 'Смещение медианы vs % цензурирования')}"
    )
    _line_chart(
        "bias_median",
        t.get("bias_median", "Смещение медианы"),
        f"{meta['name_ru']} — " + t.get("bias_median_title", "смещение медианы"),
    )

    st.markdown("---")

    # ── 3. RMSE ──
    st.markdown(
        f"### 📉 {t.get('rmse_vs_pct', 'RMSE vs % цензурирования')}"
    )
    _line_chart(
        "rmse",
        "RMSE",
        f"{meta['name_ru']} — RMSE",
        y_format=".4g",
    )

    st.markdown("---")

    # ── 4. Bias ст. отклонения ──
    st.markdown(
        f"### 📉 {t.get('bias_std_vs_pct', 'Смещение ст. отклонения vs % цензурирования')}"
    )
    _line_chart(
        "bias_std",
        t.get("bias_std", "Смещение ст. откл."),
        f"{meta['name_ru']} — " + t.get("bias_std_title", "смещение ст. отклонения"),
    )

    st.markdown("---")

    # ── Таблица ──
    st.markdown(
        f"### 📋 {t.get('sweep_table', 'Таблица результатов sweep')}"
    )

    df_show = df_sweep.copy()
    df_show["method_name"] = df_show["method"].apply(
        lambda mk: f"{METHODS[mk]['icon']} {get_method_name(mk, t)}"
    )
    df_show = df_show.rename(
        columns={
            "censor_pct": t.get("actual_pct", "Факт. %"),
            "bias_mean": t.get("bias_mean", "Смещ. средн."),
            "bias_median": t.get("bias_median", "Смещ. медиан."),
            "bias_std": t.get("bias_std", "Смещ. ст.откл."),
            "rmse": "RMSE",
        }
    )
    cols_to_show = [
        t.get("actual_pct", "Факт. %"),
        "method_name",
        t.get("bias_mean", "Смещ. средн."),
        t.get("bias_median", "Смещ. медиан."),
        "RMSE",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_show.columns]
    st.dataframe(
        df_show[cols_to_show].sort_values(
            [t.get("actual_pct", "Факт. %"), "method_name"]
        ),
        use_container_width=True,
        hide_index=True,
    )