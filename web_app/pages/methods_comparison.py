import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    get_stat_name,
    metric_card,
    get_plotly_layout,
)


def render_methods_comparison(t: dict):
    """Сравнение методов — heatmap, radar, grouped bars."""

    st.markdown(
        f"## ⚖️ {t.get('methods_comp_title', 'Сравнение методов')}"
    )
    st.markdown(
        t.get(
            "methods_comp_desc",
            "Детальное сравнение всех 8 методов по нескольким статистикам. "
            "Тепловая карта показывает относительное смещение, "
            "радарная диаграмма — общий профиль точности.",
        )
    )

    st.markdown("---")

    # ── Управление ──
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        metal_keys = list(METALS_CONFIG.keys())
        metal_labels = [
            f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})" for k in metal_keys
        ]
        sel = st.selectbox(
            t.get("select_metal", "Металл"),
            metal_labels,
            index=0,
            key="mc_metal",
        )
        metal_key = metal_keys[metal_labels.index(sel)]
        meta = METALS_CONFIG[metal_key]

    with c2:
        censor_pct = st.slider(
            t.get("censor_pct", "Доля цензурирования (%)"),
            5, 80, 30, 5,
            key="mc_pct",
        )

    with c3:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            [100, 250, 500, 1000],
            index=2,
            key="mc_n",
        )

    # ── Данные ──
    raw = generate_synthetic_data(metal_key, n=n_samples, seed=42)
    censored, lod, actual_pct = apply_censoring(raw, target_pct=censor_pct / 100.0)
    true_stats = compute_statistics(raw)

    # ── Обработка каждым методом ──
    stat_keys = ["mean", "median", "std", "q25", "q75"]
    stat_labels = [get_stat_name(sk, t) for sk in stat_keys]

    method_names = []
    method_colors = []
    method_icons = []
    bias_matrix = []  # строки = методы, столбцы = статистики
    rmse_list = []

    for mkey, minfo in METHODS.items():
        try:
            processed = minfo["func"](censored, lod)
            stats = compute_statistics(processed)
            row = []
            for sk in stat_keys:
                true_val = true_stats[sk]
                proc_val = stats[sk]
                rel_bias = (proc_val - true_val) / abs(true_val) if true_val != 0 else 0
                row.append(rel_bias)
            bias_matrix.append(row)

            br = compute_bias_rmse(true_stats, stats)
            rmse_list.append(br["rmse"])

            method_names.append(get_method_name(mkey, t))
            method_colors.append(minfo["color"])
            method_icons.append(minfo["icon"])
        except Exception:
            bias_matrix.append([np.nan] * len(stat_keys))
            rmse_list.append(np.nan)
            method_names.append(get_method_name(mkey, t))
            method_colors.append(minfo["color"])
            method_icons.append(minfo["icon"])

    st.markdown("---")

    # ── 1. Тепловая карта смещений ──
    st.markdown(
        f"### 🗺️ {t.get('heatmap_title', 'Тепловая карта смещений (%)')}"
    )

    bias_pct = np.array(bias_matrix) * 100
    hover_text = [
        [
            f"{method_names[i]}<br>{stat_labels[j]}<br>{bias_pct[i, j]:+.1f}%"
            for j in range(len(stat_keys))
        ]
        for i in range(len(method_names))
    ]

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=bias_pct,
            x=stat_labels,
            y=[f"{ic} {nm}" for ic, nm in zip(method_icons, method_names)],
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:+.1f}%" for v in row] for row in bias_pct],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title=t.get("bias_pct", "Смещение (%)"),
                ticksuffix="%",
            ),
        )
    )

    layout_heat = get_plotly_layout(
        title=t.get("heatmap_subtitle", "Относительное смещение по статистикам"),
    )
    fig_heat.update_layout(**layout_heat, height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── 2. Radar chart (RMSE + |bias_mean| + |bias_median| + |bias_std|) ──
    st.markdown(
        f"### 🕸️ {t.get('radar_title', 'Радарная диаграмма точности')}"
    )

    radar_dims = [
        t.get("rmse", "RMSE"),
        t.get("bias_mean", "|Смещ. средн.|"),
        t.get("bias_median", "|Смещ. медиан.|"),
        t.get("bias_std", "|Смещ. ст.откл.|"),
    ]

    fig_radar = go.Figure()

    for i, mkey in enumerate(METHODS.keys()):
        row = bias_matrix[i]
        if np.isnan(row[0]):
            continue
        values = [
            rmse_list[i],
            abs(row[0]),  # mean bias
            abs(row[1]),  # median bias
            abs(row[2]),  # std bias
        ]
        # Замыкаем полигон
        values += values[:1]
        dims_closed = radar_dims + radar_dims[:1]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=dims_closed,
                fill="toself",
                name=f"{method_icons[i]} {method_names[i]}",
                line_color=method_colors[i],
                opacity=0.6,
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        font=dict(size=13, family="Arial"),
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── 3. Grouped bar: RMSE по методам ──
    st.markdown(
        f"### 📊 {t.get('rmse_bar_title', 'RMSE по методам')}"
    )

    valid_idx = [i for i in range(len(rmse_list)) if not np.isnan(rmse_list[i])]
    if valid_idx:
        fig_rmse = go.Figure()
        fig_rmse.add_trace(
            go.Bar(
                x=[method_names[i] for i in valid_idx],
                y=[rmse_list[i] for i in valid_idx],
                marker_color=[method_colors[i] for i in valid_idx],
                text=[f"{rmse_list[i]:.4g}" for i in valid_idx],
                textposition="outside",
            )
        )
        layout_rmse = get_plotly_layout(
            title=t.get("rmse_subtitle", "Среднеквадратичная ошибка"),
            xaxis_title=t.get("method", "Метод"),
            yaxis_title="RMSE",
        )
        fig_rmse.update_layout(**layout_rmse)
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

    # ── 4. Полная таблица ──
    st.markdown(
        f"### 📋 {t.get('full_table', 'Полная таблица результатов')}"
    )

    table_rows = []
    for i, mkey in enumerate(METHODS.keys()):
        row = bias_matrix[i]
        entry = {
            t.get("method", "Метод"): f"{method_icons[i]} {method_names[i]}",
        }
        for j, sk in enumerate(stat_keys):
            entry[f"{stat_labels[j]} (смещ.%)"] = (
                f"{row[j]:+.2%}" if not np.isnan(row[j]) else "—"
            )
        entry["RMSE"] = f"{rmse_list[i]:.4g}" if not np.isnan(rmse_list[i]) else "—"
        table_rows.append(entry)

    df_full = pd.DataFrame(table_rows)
    st.dataframe(df_full, use_container_width=True, hide_index=True)