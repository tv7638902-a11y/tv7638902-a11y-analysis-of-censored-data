import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    METHODS,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ACCENT,
    compute_statistics,
    compute_bias_rmse,
    get_method_name,
    metric_card,
    get_plotly_layout,
)


def _detect_lod(series: pd.Series) -> float:
    """
    Простая эвристика: LOD ≈ минимальное положительное значение,
    которое встречается чаще 1 раза (округлённое до значащей цифры).
    """
    positive = series[series > 0].dropna()
    if positive.empty:
        return 1.0
    # Частоты значений
    counts = positive.round(6).value_counts()
    # Значения, встречающиеся >1 раза — кандидаты на LOD
    candidates = counts[counts > 1].index
    if len(candidates) > 0:
        return float(candidates.min())
    return float(positive.min())


def render_custom_data(t: dict):
    """Загрузка и анализ собственных данных."""

    st.markdown(
        f"## 📂 {t.get('custom_title', 'Загрузить свои данные')}"
    )
    st.markdown(
        t.get(
            "custom_desc",
            "Загрузите CSV или Excel файл с данными концентраций. "
            "Приложение автоматически определит цензурированные значения "
            "и применит все 8 методов обработки.",
        )
    )

    st.markdown("---")

    # ── 1. Загрузка файла ──
    st.markdown(f"### 📤 {t.get('upload_title', 'Загрузка файла')}")

    uploaded_file = st.file_uploader(
        t.get("upload_label", "Выберите CSV или Excel файл"),
        type=["csv", "xlsx", "xls"],
        help=t.get(
            "upload_help",
            "Файл должен содержать хотя бы один столбец с числовыми "
            "значениями концентраций. Цензурированные значения можно "
            "отметить как '<LOD', 'nd', 'н/о', NaN или отрицательные числа.",
        ),
    )

    if uploaded_file is None:
        st.info(
            t.get(
                "upload_hint",
                "💡 **Формат файла:** таблица, где строки — пробы, "
                "столбцы — металлы или параметры. Значения ниже LOD "
                "можно записать как `0`, `-1`, `NaN`, `<0.01` и т.д.",
            )
        )

        # ── Демо-данные ──
        st.markdown("---")
        st.markdown(f"### 🧪 {t.get('demo_title', 'Или попробуйте демо-данные')}")

        if st.button(
            t.get("load_demo", "📥 Загрузить демо-данные"),
            type="primary",
        ):
            np.random.seed(123)
            n = 200
            raw = np.random.lognormal(mean=-2.0, sigma=1.2, size=n)
            lod = np.percentile(raw, 25)
            data = raw.copy()
            data[data < lod] = 0  # имитация цензурирования

            df_demo = pd.DataFrame(
                {
                    "sample_id": [f"S{i+1:03d}" for i in range(n)],
                    "concentration": np.round(data, 6),
                    "is_censored": (data == 0).astype(int),
                }
            )
            st.session_state["custom_df"] = df_demo
            st.session_state["custom_col"] = "concentration"
            st.session_state["custom_lod"] = float(lod)
            st.rerun()

        return

    # ── 2. Чтение файла ──
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(
            t.get("read_error", "Ошибка чтения файла") + f": {e}"
        )
        return

    st.success(
        f"✅ {t.get('file_loaded', 'Файл загружен')}: "
        f"{df_raw.shape[0]} {t.get('rows', 'строк')}, "
        f"{df_raw.shape[1]} {t.get('columns', 'столбцов')}"
    )

    with st.expander(
        t.get("preview_raw", "👁️ Предпросмотр исходных данных"),
        expanded=False,
    ):
        st.dataframe(df_raw.head(20), use_container_width=True)

    # ── 3. Выбор столбца ──
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error(
            t.get(
                "no_numeric",
                "В файле не найдено числовых столбцов. "
                "Проверьте формат данных.",
            )
        )
        return

    st.markdown("---")
    st.markdown(f"### ⚙️ {t.get('settings_title', 'Настройки анализа')}")

    c1, c2 = st.columns(2)

    with c1:
        selected_col = st.selectbox(
            t.get("select_column", "Столбец с концентрациями"),
            options=numeric_cols,
            index=0,
            key="custom_col_select",
        )

    with c2:
        lod_mode = st.radio(
            t.get("lod_mode", "Определение LOD"),
            options=[
                t.get("lod_auto", "Автоопределение"),
                t.get("lod_manual", "Вручную"),
            ],
            index=0,
            horizontal=True,
            key="custom_lod_mode",
        )

    # ── 4. Обработка значений ──
    series = df_raw[selected_col].copy()

    # Замена строковых маркеров цензурирования
    if series.dtype == object:
        series = series.replace(
            {r"^\s*<\s*": "", r"^\s*nd\s*$": "0", r"^\s*н/о\s*$": "0"},
            regex=True,
        )
        series = pd.to_numeric(series, errors="coerce")

    # Отрицательные → 0 (цензурированные)
    series = series.fillna(0)
    censored_mask = series <= 0
    series[censored_mask] = 0

    # ── 5. LOD ──
    if lod_mode == t.get("lod_auto", "Автоопределение"):
        positive_vals = series[series > 0]
        if len(positive_vals) > 0:
            lod_value = _detect_lod(series)
        else:
            lod_value = 1.0
        st.info(
            f"🔍 {t.get('lod_detected', 'Обнаруженный LOD')}: "
            f"**{lod_value:.4g}**"
        )
    else:
        lod_value = st.number_input(
            t.get("enter_lod", "Введите значение LOD"),
            min_value=0.0,
            value=0.01,
            step=0.001,
            format="%.4g",
            key="custom_lod_input",
        )

    # Заменяем цензурированные на LOD для базовой обработки
    processed_series = series.copy()
    processed_series[censored_mask] = lod_value

    n_total = len(series)
    n_censored = censored_mask.sum()
    pct_censored = n_censored / n_total * 100 if n_total > 0 else 0

    st.markdown("---")

    # ── Метрики ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(
            t.get("total_samples", "Всего проб"),
            str(n_total),
            "🧪",
            COLOR_PRIMARY,
        )
    with m2:
        metric_card(
            t.get("censored_count", "Цензурированных"),
            str(n_censored),
            "🚫",
            COLOR_DANGER,
        )
    with m3:
        metric_card(
            t.get("censor_pct", "Доля ценз."),
            f"{pct_censored:.1f}%",
            "📊",
            COLOR_WARNING,
        )
    with m4:
        metric_card(
            "LOD",
            f"{lod_value:.4g}",
            "🔍",
            COLOR_ACCENT,
        )

    if n_censored == 0:
        st.warning(
            t.get(
                "no_censored",
                "⚠️ Цензурированные значения не обнаружены. "
                "Все методы дадут одинаковый результат.",
            )
        )

    st.markdown("---")

    # ── 6. Применение методов ──
    st.markdown(
        f"### 🔬 {t.get('results_title', 'Результаты обработки')}"
    )

    censored_array = processed_series.values.astype(float)
    true_stats = compute_statistics(censored_array)

    results = {}
    for mkey, minfo in METHODS.items():
        func = minfo["func"]
        try:
            processed = func(censored_array, lod_value)
            stats = compute_statistics(processed)
            results[mkey] = {
                "processed": processed,
                "stats": stats,
                "name": get_method_name(mkey, t),
                "color": minfo["color"],
                "icon": minfo["icon"],
                "error": None,
            }
        except Exception as e:
            results[mkey] = {
                "processed": None,
                "stats": None,
                "name": get_method_name(mkey, t),
                "color": minfo["color"],
                "icon": minfo["icon"],
                "error": str(e),
            }

    # ── 7. Гистограмма ──
    st.markdown(
        f"### 📊 {t.get('hist_title', 'Гистограмма после обработки')}"
    )

    fig_hist = go.Figure()

    # Исходные (нецензурированные)
    non_cens = censored_array[~censored_mask.values]
    if len(non_cens) > 0:
        fig_hist.add_trace(
            go.Histogram(
                x=non_cens,
                nbinsx=30,
                name=t.get("detected", "Обнаруженные"),
                marker_color=COLOR_PRIMARY,
                opacity=0.7,
            )
        )

    # Цензурированные (заменённые на LOD)
    cens_vals = censored_array[censored_mask.values]
    if len(cens_vals) > 0:
        fig_hist.add_trace(
            go.Histogram(
                x=cens_vals,
                nbinsx=5,
                name=t.get("censored", "Цензурированные"),
                marker_color=COLOR_DANGER,
                opacity=0.5,
            )
        )

    # Линия LOD
    fig_hist.add_vline(
        x=lod_value,
        line_width=3,
        line_color="#e74c3c",
        line_dash="dash",
    )
    fig_hist.add_annotation(
        x=lod_value,
        y=1.02,
        xref="x",
        yref="paper",
        text=f"LOD = {lod_value:.4g}",
        showarrow=False,
        font=dict(color="#e74c3c", size=13, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e74c3c",
        borderwidth=1,
        borderpad=4,
    )

    layout_hist = get_plotly_layout(
        title=t.get("custom_hist_title", "Распределение концентраций"),
        xaxis_title=t.get("concentration", "Концентрация"),
        yaxis_title=t.get("count", "Частота"),
    )
    fig_hist.update_layout(**layout_hist, barmode="overlay")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── 8. Box plot всех методов ──
    st.markdown(
        f"### 📦 {t.get('box_methods', 'Сравнение распределений по методам')}"
    )

    fig_box = go.Figure()

    fig_box.add_trace(
        go.Box(
            y=censored_array,
            name=t.get("input_data", "Входные данные"),
            marker_color="#2c3e50",
            boxmean=True,
        )
    )

    for mkey, res in results.items():
        if res["processed"] is not None:
            fig_box.add_trace(
                go.Box(
                    y=res["processed"],
                    name=f"{res['icon']} {res['name']}",
                    marker_color=res["color"],
                    boxmean=True,
                )
            )

    layout_box = get_plotly_layout(
        title=t.get("methods_distributions", "Распределения после обработки"),
        yaxis_title=t.get("concentration", "Концентрация"),
    )
    fig_box.update_layout(**layout_box, height=450)
    fig_box.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
        )
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ── 9. Bar chart: средние значения ──
    st.markdown(
        f"### 📊 {t.get('means_bar', 'Средние значения по методам')}"
    )

    valid = {k: v for k, v in results.items() if v["stats"] is not None}
    if valid:
        names = [f"{v['icon']} {v['name']}" for v in valid.values()]
        means = [v["stats"]["mean"] for v in valid.values()]
        colors = [v["color"] for v in valid.values()]

        fig_means = go.Figure()
        fig_means.add_trace(
            go.Bar(
                x=names,
                y=means,
                marker_color=colors,
                text=[f"{m:.4g}" for m in means],
                textposition="outside",
            )
        )

        # Линия "истинного" среднего (входные данные)
        fig_means.add_hline(
            y=true_stats["mean"],
            line_width=2,
            line_color="#333333",
            line_dash="dash",
            annotation_text=t.get("input_mean", "Среднее входных"),
            annotation_position="top right",
        )

        layout_means = get_plotly_layout(
            title=t.get("mean_comparison", "Сравнение средних"),
            xaxis_title=t.get("method", "Метод"),
            yaxis_title=t.get("mean", "Среднее"),
        )
        fig_means.update_layout(**layout_means)
        st.plotly_chart(fig_means, use_container_width=True)

    st.markdown("---")

    # ── 10. Сводная таблица ──
    st.markdown(
        f"### 📋 {t.get('summary_table', 'Сводная таблица')}"
    )

    table_rows = []
    for mkey, res in results.items():
        if res["stats"] is not None:
            s = res["stats"]
            table_rows.append(
                {
                    t.get("method", "Метод"): f"{res['icon']} {res['name']}",
                    t.get("mean", "Среднее"): f"{s['mean']:.4g}",
                    t.get("median", "Медиана"): f"{s['median']:.4g}",
                    t.get("std", "Ст. откл."): f"{s['std']:.4g}",
                    t.get("q25", "Q25"): f"{s['q25']:.4g}",
                    t.get("q75", "Q75"): f"{s['q75']:.4g}",
                    t.get("min", "Мин."): f"{s['min']:.4g}",
                    t.get("max", "Макс."): f"{s['max']:.4g}",
                }
            )
        else:
            table_rows.append(
                {
                    t.get("method", "Метод"): f"{res['icon']} {res['name']}",
                    t.get("mean", "Среднее"): f"⚠️ {res['error']}",
                    t.get("median", "Медиана"): "—",
                    t.get("std", "Ст. откл."): "—",
                    t.get("q25", "Q25"): "—",
                    t.get("q75", "Q75"): "—",
                    t.get("min", "Мин."): "—",
                    t.get("max", "Макс."): "—",
                }
            )

    # Эталон
    table_rows.insert(
        0,
        {
            t.get("method", "Метод"): "📥 " + t.get("input_data", "Входные данные"),
            t.get("mean", "Среднее"): f"{true_stats['mean']:.4g}",
            t.get("median", "Медиана"): f"{true_stats['median']:.4g}",
            t.get("std", "Ст. откл."): f"{true_stats['std']:.4g}",
            t.get("q25", "Q25"): f"{true_stats['q25']:.4g}",
            t.get("q75", "Q75"): f"{true_stats['q75']:.4g}",
            t.get("min", "Мин."): f"{true_stats['min']:.4g}",
            t.get("max", "Макс."): f"{true_stats['max']:.4g}",
        },
    )

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 11. Экспорт ──
    st.markdown(f"### 💾 {t.get('export_title', 'Экспорт результатов')}")

    if valid:
        export_data = {}
        export_data[t.get("input_data", "Входные")] = censored_array
        for mkey, res in valid.items():
            export_data[res["name"]] = res["processed"]

        df_export = pd.DataFrame(export_data)

        csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=t.get("download_csv", "📥 Скачать CSV"),
            data=csv_bytes,
            file_name="censoring_results.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── Пояснение ──
    st.info(
        t.get(
            "custom_note",
            "💡 **Примечание:** при работе с реальными данными «истинные» "
            "значения цензурированных проб неизвестны, поэтому смещение и "
            "RMSE рассчитать нельзя. Таблица показывает **результаты "
            "обработки** каждым методом. Для оценки точности используйте "
            "раздел «🎲 Монте-Карло» с синтетическими данными.",
        )
    )