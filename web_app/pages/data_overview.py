import streamlit as st
import os
import sys
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    METALS_CONFIG,
    COLOR_PRIMARY,
    COLOR_DANGER,
    generate_synthetic_data,
    apply_censoring,
    compute_statistics,
    metric_card,
    get_plotly_layout,
)


def render_data_overview(t: dict):
    """Обзор данных — гистограммы, статистика, таблица."""

    st.markdown(
        f"## 📊 {t.get('data_overview_title', 'Обзор данных')}"
    )
    st.markdown(
        t.get(
            "data_overview_desc",
            "Синтетические данные генерируются из логнормального распределения, "
            "параметры которого подобраны под реальные диапазоны концентраций.",
        )
    )

    st.markdown("---")

    # ── Настройки ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])

    with col_ctrl1:
        metal_keys = list(METALS_CONFIG.keys())
        metal_labels = [
            f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})" for k in metal_keys
        ]
        sel_metal_label = st.selectbox(
            t.get("select_metal", "Металл"),
            options=metal_labels,
            index=0,
        )
        metal_key = metal_keys[metal_labels.index(sel_metal_label)]
        meta = METALS_CONFIG[metal_key]

    with col_ctrl2:
        censor_pct = st.slider(
            t.get("censor_pct", "Доля цензурирования (%)"),
            min_value=0,
            max_value=80,
            value=30,
            step=5,
        )

    with col_ctrl3:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            options=[100, 250, 500, 1000],
            index=2,
        )

    # ── Генерация данных ──
    raw_data = generate_synthetic_data(metal_key, n=n_samples, seed=42)
    censored_data, lod_value, actual_pct = apply_censoring(
        raw_data, target_pct=censor_pct / 100.0
    )

    stats_raw = compute_statistics(raw_data)
    stats_cens = compute_statistics(censored_data)

    # ── Метрики ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(
            t.get("mean_label", "Среднее (исх.)"),
            f"{stats_raw['mean']:.4g}",
            "📐",
            COLOR_PRIMARY,
        )
    with m2:
        metric_card(
            t.get("mean_cens_label", "Среднее (ценз.)"),
            f"{stats_cens['mean']:.4g}",
            "📐",
            COLOR_DANGER,
        )
    with m3:
        metric_card(
            t.get("median_label", "Медиана (исх.)"),
            f"{stats_raw['median']:.4g}",
            "📏",
            COLOR_PRIMARY,
        )
    with m4:
        metric_card(
            t.get("actual_censor", "Факт. % ценз."),
            f"{actual_pct:.1f}%",
            "🚫",
            COLOR_DANGER,
        )

    st.markdown("---")

    # ── Гистограмма ──
    st.markdown(f"### 📊 {t.get('histogram_title', 'Распределение концентраций')}")

    fig = go.Figure()

    # Исходные данные
    fig.add_trace(
        go.Histogram(
            x=raw_data,
            nbinsx=40,
            name=t.get("original", "Исходные"),
            marker_color=COLOR_PRIMARY,
            opacity=0.7,
        )
    )

    # Цензурированные (только те, что были заменены)
    censored_mask = censored_data != raw_data
    if censored_mask.any():
        fig.add_trace(
            go.Histogram(
                x=censored_data[censored_mask],
                nbinsx=10,
                name=t.get("censored", "Цензурированные"),
                marker_color=COLOR_DANGER,
                opacity=0.5,
            )
        )

    # Линия ПДК
    pdk = meta["pdk"]
    fig.add_vline(
        x=pdk,
        line_width=3,
        line_color="#e74c3c",
        line_dash="dash",
    )
    fig.add_annotation(
        x=pdk,
        y=1.02,
        xref="x",
        yref="paper",
        text=f"ПДК = {pdk} {meta['unit']}",
        showarrow=False,
        font=dict(color="#e74c3c", size=13, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e74c3c",
        borderwidth=1,
        borderpad=4,
    )

    # Линия LOD
    fig.add_vline(
        x=lod_value,
        line_width=2,
        line_color="#f39c12",
        line_dash="dot",
    )
    fig.add_annotation(
        x=lod_value,
        y=0.92,
        xref="x",
        yref="paper",
        text=f"LOD = {lod_value:.4g}",
        showarrow=False,
        font=dict(color="#f39c12", size=12),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#f39c12",
        borderwidth=1,
        borderpad=3,
    )

    layout = get_plotly_layout(
        title=f"{meta['name_ru']} ({metal_key.upper()}) — N={n_samples}",
        xaxis_title=t.get("concentration", "Концентрация") + f", {meta['unit']}",
        yaxis_title=t.get("count", "Частота"),
    )
    fig.update_layout(**layout, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Статистика ──
    st.markdown(f"### 📋 {t.get('stats_table_title', 'Сравнительная статистика')}")

    import pandas as pd

    stat_names = {
        "mean": t.get("mean", "Среднее"),
        "median": t.get("median", "Медиана"),
        "std": t.get("std", "Ст. откл."),
        "q25": t.get("q25", "Q25"),
        "q75": t.get("q75", "Q75"),
        "min": t.get("min", "Мин."),
        "max": t.get("max", "Макс."),
    }

    rows = []
    for key, label in stat_names.items():
        rows.append(
            {
                t.get("statistic", "Статистика"): label,
                t.get("original_col", "Исходные"): f"{stats_raw[key]:.4g}",
                t.get("censored_col", "Цензурированные"): f"{stats_cens[key]:.4g}",
                t.get("diff_col", "Разница (%)"): (
                    f"{(stats_cens[key] - stats_raw[key]) / abs(stats_raw[key]) * 100:+.1f}%"
                    if stats_raw[key] != 0
                    else "—"
                ),
            }
        )

    df_stats = pd.DataFrame(rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Таблица данных (первые 50 строк) ──
    st.markdown(f"### 📄 {t.get('data_table_title', 'Таблица данных (первые 50 строк)')}")

    df_show = pd.DataFrame(
        {
            t.get("idx", "№"): range(1, min(51, n_samples + 1)),
            t.get("raw_val", "Исходное"): raw_data[:50],
            t.get("cens_val", "Цензурированное"): censored_data[:50],
            t.get("is_cens", "Ценз.?"): [
                "🔴" if c else "🟢" for c in censored_mask[:50]
            ],
        }
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)