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


def render_censoring_experiment(t: dict):
    """Эксперимент с цензурированием — интерактивное сравнение 8 методов."""

    st.markdown(
        f"## 🧪 {t.get('censoring_exp_title', 'Эксперимент с цензурированием')}"
    )
    st.markdown(
        t.get(
            "censoring_exp_desc",
            "Выберите металл и долю цензурирования. Все 8 методов будут применены "
            "к одним и тем же данным, и вы увидите, насколько каждый из них "
            "восстанавливает истинные статистики.",
        )
    )

    st.markdown("---")

    # ── Панель управления ──
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])

    with c1:
        metal_keys = list(METALS_CONFIG.keys())
        metal_labels = [
            f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})" for k in metal_keys
        ]
        sel_label = st.selectbox(
            t.get("select_metal", "Металл"),
            options=metal_labels,
            index=0,
            key="cens_exp_metal",
        )
        metal_key = metal_keys[metal_labels.index(sel_label)]
        meta = METALS_CONFIG[metal_key]

    with c2:
        censor_pct = st.slider(
            t.get("censor_pct", "Доля цензурирования (%)"),
            min_value=5,
            max_value=80,
            value=30,
            step=5,
            key="cens_exp_pct",
        )

    with c3:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            options=[100, 250, 500, 1000],
            index=2,
            key="cens_exp_n",
        )

    with c4:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="cens_exp_seed",
        )

    # ── Генерация и цензурирование ──
    raw_data = generate_synthetic_data(metal_key, n=n_samples, seed=seed)
    censored_data, lod_value, actual_pct = apply_censoring(
        raw_data, target_pct=censor_pct / 100.0
    )
    true_stats = compute_statistics(raw_data)

    # ── Применение всех методов ──
    results = {}
    for mkey, minfo in METHODS.items():
        func = minfo["func"]
        try:
            processed = func(censored_data, lod_value)
            stats = compute_statistics(processed)
            bias_rmse = compute_bias_rmse(true_stats, stats)
            results[mkey] = {
                "stats": stats,
                "bias": bias_rmse["bias_mean"],
                "rmse": bias_rmse["rmse"],
                "color": minfo["color"],
                "icon": minfo["icon"],
                "name": get_method_name(mkey, t),
            }
        except Exception as e:
            results[mkey] = {
                "stats": None,
                "bias": None,
                "rmse": None,
                "color": minfo["color"],
                "icon": minfo["icon"],
                "name": get_method_name(mkey, t),
                "error": str(e),
            }

    # ── Ключевые метрики ──
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card(
            t.get("true_mean", "Истинное среднее"),
            f"{true_stats['mean']:.4g}",
            "🎯",
            COLOR_SUCCESS,
        )
    with m2:
        metric_card(
            t.get("lod_value", "Значение LOD"),
            f"{lod_value:.4g}",
            "🔍",
            COLOR_WARNING,
        )
    with m3:
        metric_card(
            t.get("actual_censor", "Факт. % ценз."),
            f"{actual_pct:.1f}%",
            "🚫",
            COLOR_DANGER,
        )

    st.markdown("---")

    # ── Bar chart: смещение среднего по методам ──
    st.markdown(
        f"### 📊 {t.get('bias_chart_title', 'Смещение среднего по методам')}"
    )

    valid_methods = {k: v for k, v in results.items() if v["bias"] is not None}

    if valid_methods:
        method_names = [v["name"] for v in valid_methods.values()]
        biases = [v["bias"] for v in valid_methods.values()]
        colors = [v["color"] for v in valid_methods.values()]

        fig_bias = go.Figure()
        fig_bias.add_trace(
            go.Bar(
                x=method_names,
                y=biases,
                marker_color=colors,
                text=[f"{b:+.2%}" for b in biases],
                textposition="outside",
                hovertemplate=(
                    t.get("method", "Метод") + ": %{x}<br>"
                    + t.get("bias", "Смещение") + ": %{y:+.4f}<extra></extra>"
                ),
            )
        )

        # Нулевая линия
        fig_bias.add_hline(
            y=0, line_width=2, line_color="#333333", line_dash="solid"
        )

        layout_bias = get_plotly_layout(
            title=t.get("bias_mean_title", "Относительное смещение среднего"),
            xaxis_title=t.get("method", "Метод"),
            yaxis_title=t.get("bias", "Смещение"),
        )
        fig_bias.update_layout(**layout_bias)
        st.plotly_chart(fig_bias, use_container_width=True)

    st.markdown("---")

    # ── Box plot: распределения после обработки ──
    st.markdown(
        f"### 📦 {t.get('boxplot_title', 'Распределения после обработки')}"
    )

    fig_box = go.Figure()

    # Исходные данные (эталон)
    fig_box.add_trace(
        go.Box(
            y=raw_data,
            name=t.get("original", "Исходные"),
            marker_color="#2c3e50",
            boxmean=True,
        )
    )

    for mkey, res in valid_methods.items():
        func = METHODS[mkey]["func"]
        try:
            processed = func(censored_data, lod_value)
            fig_box.add_trace(
                go.Box(
                    y=processed,
                    name=res["name"],
                    marker_color=res["color"],
                    boxmean=True,
                )
            )
        except Exception:
            pass

    # Линия ПДК
    pdk = meta["pdk"]
    fig_box.add_hline(
        y=pdk,
        line_width=3,
        line_color="#e74c3c",
        line_dash="dash",
    )
    fig_box.add_annotation(
        x=1.02,
        y=pdk,
        xref="paper",
        yref="y",
        text=f"ПДК = {pdk}",
        showarrow=False,
        font=dict(color="#e74c3c", size=13, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e74c3c",
        borderwidth=1,
        borderpad=4,
    )

    layout_box = get_plotly_layout(
        title=f"{meta['name_ru']} — {t.get('distributions', 'распределения')}",
        yaxis_title=t.get("concentration", "Концентрация") + f", {meta['unit']}",
    )
    fig_box.update_layout(**layout_box)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ── Сводная таблица ──
    st.markdown(
        f"### 📋 {t.get('summary_table', 'Сводная таблица результатов')}"
    )

    rows = []
    for mkey, res in results.items():
        if res["stats"] is not None:
            s = res["stats"]
            rows.append(
                {
                    t.get("method", "Метод"): f"{res['icon']} {res['name']}",
                    t.get("mean", "Среднее"): f"{s['mean']:.4g}",
                    t.get("median", "Медиана"): f"{s['median']:.4g}",
                    t.get("std", "Ст. откл."): f"{s['std']:.4g}",
                    t.get("bias", "Смещение"): (
                        f"{res['bias']:+.2%}" if res["bias"] is not None else "—"
                    ),
                    t.get("rmse", "RMSE"): (
                        f"{res['rmse']:.4g}" if res["rmse"] is not None else "—"
                    ),
                }
            )
        else:
            rows.append(
                {
                    t.get("method", "Метод"): f"{res['icon']} {res['name']}",
                    t.get("mean", "Среднее"): "⚠️ Ошибка",
                    t.get("median", "Медиана"): "—",
                    t.get("std", "Ст. откл."): "—",
                    t.get("bias", "Смещение"): "—",
                    t.get("rmse", "RMSE"): "—",
                }
            )

    # Эталонная строка
    rows.insert(
        0,
        {
            t.get("method", "Метод"): "🎯 " + t.get("true_values", "Истинные"),
            t.get("mean", "Среднее"): f"{true_stats['mean']:.4g}",
            t.get("median", "Медиана"): f"{true_stats['median']:.4g}",
            t.get("std", "Ст. откл."): f"{true_stats['std']:.4g}",
            t.get("bias", "Смещение"): "0.00%",
            t.get("rmse", "RMSE"): "0",
        },
    )

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)