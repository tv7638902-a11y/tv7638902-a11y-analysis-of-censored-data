import streamlit as st
import os
import sys

# ── Добавляем корень проекта в path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Конфигурация страницы (ОБЯЗАТЕЛЬНО первым вызовом!) ──
st.set_page_config(
    page_title="Тяжёлые металлы — Анализ LOD",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import (
    init_session_state,
    inject_custom_css,
    get_translations,
)

# ── Инициализация ──
init_session_state()
inject_custom_css()

# ── Импорт страниц ──
from pages.home import render_home
from pages.data_overview import render_data_overview
from pages.censoring_experiment import render_censoring_experiment
from pages.methods_comparison import render_methods_comparison
from pages.censoring_dependency import render_censoring_dependency
from pages.ranking import render_ranking
from pages.monte_carlo import render_monte_carlo
from pages.custom_data import render_custom_data

# ── Конфигурация навигации ──
PAGES = [
    ("🏠", "home",               "Главная"),
    ("📊", "data_overview",      "Обзор данных"),
    ("🧪", "censoring_exp",      "Эксперимент с цензурированием"),
    ("⚖️", "methods_compare",    "Сравнение методов"),
    ("📈", "censoring_dep",      "Зависимость от % цензурирования"),
    ("🏆", "ranking",            "Итоговый рейтинг"),
    ("🎲", "monte_carlo",        "Монте-Карло"),
    ("📂", "custom_data",        "Загрузить свои данные"),
]

PAGE_RENDERERS = {
    "home":            render_home,
    "data_overview":   render_data_overview,
    "censoring_exp":   render_censoring_experiment,
    "methods_compare": render_methods_comparison,
    "censoring_dep":   render_censoring_dependency,
    "ranking":         render_ranking,
    "monte_carlo":     render_monte_carlo,
    "custom_data":     render_custom_data,
}

# ══════════════════════════════════════════
#  САЙДБАР
# ══════════════════════════════════════════
with st.sidebar:
    # ── Заголовок ──
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 5px 0;">
            <h2 style="margin:0; color:#ffffff;">🔬 Тяжёлые металлы</h2>
            <p style="margin:4px 0 0 0; color:#8899aa; font-size:0.95em;">
                Анализ данных ниже LOD
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Навигация (ВВЕРХУ) ──
    page_labels = [f"{icon}  {label}" for icon, key, label in PAGES]
    page_keys   = [key for _, key, _ in PAGES]

    current_idx = page_keys.index(st.session_state.get("current_page", "home"))
    selected = st.radio(
        "Навигация",
        options=page_labels,
        index=current_idx,
        label_visibility="collapsed",
    )
    selected_key = page_keys[page_labels.index(selected)]
    st.session_state["current_page"] = selected_key

    st.markdown("---")

    # ── Язык (ВНИЗУ) ──
    lang = st.selectbox(
        "🌐 Язык / Language",
        options=["Русский", "English"],
        index=0 if st.session_state.get("lang", "ru") == "ru" else 1,
    )
    st.session_state["lang"] = "ru" if lang == "Русский" else "en"

    # ── Подвал ──
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#556677; font-size:0.8em;">'
        "v1.0.0 • Streamlit • 2026</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
#  ОСНОВНОЙ КОНТЕНТ
# ══════════════════════════════════════════
t = get_translations(st.session_state["lang"])
renderer = PAGE_RENDERERS.get(selected_key, render_home)
renderer(t)
