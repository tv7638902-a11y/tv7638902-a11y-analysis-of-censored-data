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
def _compute_rankings(
    metal_keys: tuple,
    pcts: tuple,
    n: int,
    seed: int,
    method_keys: tuple,
):
    """
    Прогоняет все методы по всем металлам × процентам цензурирования,
    считает ранги и возвращает сводный DataFrame.
    """
    all_rows = []

    for metal_key in metal_keys:
        raw = generate_synthetic_data(metal_key, n=n, seed=seed)
        true_stats = compute_statistics(raw)

        for pct in pcts:
            censored, lod, actual_pct = apply_censoring(
                raw, target_pct=pct / 100.0
            )
            scenario_rows = []

            for mkey in method_keys:
                func = METHODS[mkey]["func"]
                try:
                    processed = func(censored, lod)
                    stats = compute_statistics(processed)
                    br = compute_bias_rmse(true_stats, stats)
                    scenario_rows.append(
                        {
                            "metal": metal_key,
                            "censor_pct": pct,
                            "actual_pct": actual_pct,
                            "method": mkey,
                            "abs_bias_mean": abs(br["bias_mean"]),
                            "abs_bias_median": abs(br["bias_median"]),
                            "abs_bias_std": abs(br["bias_std"]),
                            "rmse": br["rmse"],
                        }
                    )
                except Exception:
                    scenario_rows.append(
                        {
                            "metal": metal_key,
                            "censor_pct": pct,
                            "actual_pct": actual_pct,
                            "method": mkey,
                            "abs_bias_mean": np.nan,
                            "abs_bias_median": np.nan,
                            "abs_bias_std": np.nan,
                            "rmse": np.nan,
                        }
                    )

            # Ранжируем внутри каждого сценария (меньше = лучше → ранг 1)
            df_scen = pd.DataFrame(scenario_rows)
            for col in ["abs_bias_mean", "abs_bias_median", "abs_bias_std", "rmse"]:
                rank_col = f"rank_{col}"
                df_scen[rank_col] = df_scen[col].rank(method="min")
            all_rows.append(df_scen)

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


def render_ranking(t: dict):
    """Итоговый рейтинг методов."""

    st.markdown(
        f"## 🏆 {t.get('ranking_title', 'Итоговый рейтинг методов')}"
    )
    st.markdown(
        t.get(
            "ranking_desc",
            "Автоматическое ранжирование всех методов по совокупности "
            "металлов и уровней цензурирования. Чем ниже средний ранг — "
            "тем лучше метод.",
        )
    )

    st.markdown("---")

    # ── Управление ──
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        metal_keys_all = list(METALS_CONFIG.keys())
        selected_metals = st.multiselect(
            t.get("select_metals", "Металлы"),
            options=metal_keys_all,
            default=metal_keys_all,
            format_func=lambda k: f"{METALS_CONFIG[k]['name_ru']} ({k.upper()})",
            key="rank_metals",
        )

    with c2:
        n_samples = st.selectbox(
            t.get("n_samples", "N"),
            [100, 250, 500, 1000],
            index=2,
            key="rank_n",
        )

    with c3:
        seed = st.number_input(
            "Seed", 0, 9999, 42, 1, key="rank_seed"
        )

    if not selected_metals:
        st.warning(t.get("select_at_least_one", "Выберите хотя бы один металл."))
        return

    # ── Параметры sweep ──
    pcts = (10, 20, 30, 40, 50, 60, 70)
    method_keys = tuple(METHODS.keys())

    with st.spinner(t.get("computing", "Вычисление...")):
        df_all = _compute_rankings(
            tuple(selected_metals), pcts, n_samples, seed, method_keys
        )

    if df_all.empty:
        st.error(t.get("no_data", "Нет данных для отображения."))
        return

    # ── Средние ранги по методам ──
    rank_cols = [
        "rank_abs_bias_mean",
        "rank_abs_bias_median",
        "rank_abs_bias_std",
        "rank_rmse",
    ]
    rank_labels = [
        t.get("rank_bias_mean", "Ранг |смещ. средн.|"),
        t.get("rank_bias_median", "Ранг |смещ. медиан.|"),
        t.get("rank_bias_std", "Ранг |смещ. ст.откл.|"),
        t.get("rank_rmse", "Ранг RMSE"),
    ]

    summary = df_all.groupby("method")[rank_cols].mean().reset_index()
    summary["avg_rank"] = summary[rank_cols].mean(axis=1)
    summary = summary.sort_values("avg_rank").reset_index(drop=True)
    summary["overall_rank"] = range(1, len(summary) + 1)

    # ── Ключевые метрики ──
    best_mkey = summary.iloc[0]["method"]
    worst_mkey = summary.iloc[-1]["method"]
    total_scenarios = len(selected_metals) * len(pcts)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(
            t.get("scenarios", "Сценариев"),
            str(total_scenarios),
            "🔬",
            COLOR_PRIMARY,
        )
    with m2:
        metric_card(
            t.get("best_overall", "🥇 Лучший"),
            f"{METHODS[best_mkey]['icon']} {get_method_name(best_mkey, t)}",
            "🏆",
            COLOR_SUCCESS,
        )
    with m3:
        metric_card(
            t.get("worst_overall", "Последний"),
            f"{METHODS[worst_mkey]['icon']} {get_method_name(worst_mkey, t)}",
            "⚠️",
            COLOR_DANGER,
        )
    with m4:
        metric_card(
            t.get("avg_best_rank", "Ср. ранг лидера"),
            f"{summary.iloc[0]['avg_rank']:.2f}",
            "📊",
            COLOR_ACCENT,
        )

    st.markdown("---")

    # ── 1. Горизонтальный bar chart — общий рейтинг ──
    st.markdown(
        f"### 📊 {t.get('overall_ranking_chart', 'Общий рейтинг (средний ранг)')}"
    )

    # Переворачиваем для горизонтального бара (лучший сверху)
    df_plot = summary.iloc[::-1].copy()
    df_plot["label"] = df_plot["method"].apply(
        lambda mk: f"{METHODS[mk]['icon']} {get_method_name(mk, t)}"
    )
    df_plot["bar_color"] = df_plot["method"].apply(
        lambda mk: METHODS[mk]["color"]
    )

    fig_rank = go.Figure()
    fig_rank.add_trace(
        go.Bar(
            y=df_plot["label"],
            x=df_plot["avg_rank"],
            orientation="h",
            marker_color=df_plot["bar_color"],
            text=[
                f"#{int(r)}  (ср. ранг {v:.2f})"
                for r, v in zip(df_plot["overall_rank"][::-1], df_plot["avg_rank"])
            ],
            textposition="outside",
            hovertemplate=(
                t.get("method", "Метод") + ": %{y}<br>"
                + t.get("avg_rank", "Ср. ранг") + ": %{x:.2f}<extra></extra>"
            ),
        )
    )

    # Заменяем блок для fig_rank
    layout_rank = get_plotly_layout(
        title=t.get("lower_is_better", "← Ниже = лучше"),
        xaxis_title=t.get("avg_rank", "Средний ранг"),
        height=max(350, len(summary) * 55),
    )
    fig_rank.update_layout(**layout_rank)
    fig_rank.update_xaxes(range=[0, df_plot["avg_rank"].max() * 1.25])
    st.plotly_chart(fig_rank, use_container_width=True)
    st.markdown("---")

    # ── 2. Grouped bar: ранги по отдельным метрикам ──
    st.markdown(
        f"### 📊 {t.get('rank_by_metric', 'Ранги по отдельным метрикам')}"
    )

    fig_grouped = go.Figure()
    for i, (rc, rl) in enumerate(zip(rank_cols, rank_labels)):
        df_sorted = summary.sort_values("avg_rank")
        fig_grouped.add_trace(
            go.Bar(
                x=[
                    f"{METHODS[mk]['icon']} {get_method_name(mk, t)}"
                    for mk in df_sorted["method"]
                ],
                y=df_sorted[rc],
                name=rl,
                marker_color=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER][i],
                opacity=0.8,
            )
        )

    # Заменяем старый блок вызова layout для fig_grouped
    layout_grp = get_plotly_layout(
        title=t.get("rank_comparison", "Сравнение рангов по метрикам"),
        xaxis_title=t.get("method", "Метод"),
        yaxis_title=t.get("avg_rank", "Средний ранг"),
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    fig_grouped.update_layout(**layout_grp)
    st.plotly_chart(fig_grouped, use_container_width=True)
    st.markdown("---")

    # ── 3. Heatmap: ранг по металл × метод ──
    st.markdown(
        f"### 🗺️ {t.get('rank_heatmap', 'Тепловая карта: средний ранг по металлам')}"
    )

    metal_method = (
        df_all.groupby(["metal", "method"])["avg_rank"]
        .mean()
        .reset_index()
    ) if "avg_rank" in df_all.columns else pd.DataFrame()

    if metal_method.empty:
        # Считаем avg_rank на лету
        df_all["avg_rank"] = df_all[rank_cols].mean(axis=1)
        metal_method = (
            df_all.groupby(["metal", "method"])["avg_rank"]
            .mean()
            .reset_index()
        )

    pivot = metal_method.pivot(index="method", columns="metal", values="avg_rank")
    pivot = pivot.reindex(
        index=summary.sort_values("avg_rank")["method"].tolist()
    )

    y_labels = [
        f"{METHODS[mk]['icon']} {get_method_name(mk, t)}"
        for mk in pivot.index
    ]
    x_labels = [
        f"{METALS_CONFIG[mk]['name_ru']}" for mk in pivot.columns
    ]

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn_r",
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorbar=dict(title=t.get("avg_rank", "Ср. ранг")),
            hovertemplate=(
                t.get("method", "Метод") + ": %{y}<br>"
                + t.get("metal", "Металл") + ": %{x}<br>"
                + t.get("avg_rank", "Ср. ранг") + ": %{z:.2f}<extra></extra>"
            ),
        )
    )

    layout_hm = get_plotly_layout(
        title=t.get("heatmap_rank_title", "Средний ранг (зелёный = лучше)"),
    )
    fig_hm.update_layout(**layout_hm, height=max(350, len(pivot) * 50))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ── 4. Итоговая таблица ──
    st.markdown(
        f"### 📋 {t.get('ranking_table', 'Итоговая таблица рейтинга')}"
    )

    table_rows = []
    for _, row in summary.iterrows():
        mk = row["method"]
        entry = {
            "🏅": f"#{int(row['overall_rank'])}",
            t.get("method", "Метод"): f"{METHODS[mk]['icon']} {get_method_name(mk, t)}",
        }
        for rc, rl in zip(rank_cols, rank_labels):
            entry[rl] = f"{row[rc]:.2f}"
        entry[t.get("avg_rank", "Ср. ранг")] = f"{row['avg_rank']:.2f}"
        table_rows.append(entry)

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Пояснение ──
    st.markdown("---")
    st.info(
        t.get(
            "ranking_note",
            "💡 **Как читать рейтинг:** для каждого сценария (металл × % цензурирования) "
            "методы ранжируются от 1 (лучший) до 8 (худший) по каждой метрике. "
            "Итоговый ранг — среднее по всем сценариям и метрикам. "
            "Чем ближе к 1, тем стабильнее и точнее метод.",
        )
    )