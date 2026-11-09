import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    METALS_CONFIG,
    METHODS,
    metric_card,
    get_translations,
)


def render_home(t: dict):
    """Главная страница — обзор проекта."""

    # ── Заголовок ──
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 30px 35px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid #334155;
            color: white;
        ">
            <h1 style="margin:0 0 8px 0; font-size:2.2em; color:#ffffff;">
                🔬 {t.get("home_title", "Анализ цензурированных данных")}
            </h1>
            <p style="margin:0; color:#94a3b8; font-size:1.1em;">
                {t.get("home_subtitle",
                "Сравнение методов обработки концентраций тяжёлых металлов ниже предела обнаружения (LOD)")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Ключевые цифры ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            t.get("metals_count", "Металлов"),
            str(len(METALS_CONFIG)),
            "🧪",
            "#3498db",
        )
    with col2:
        metric_card(
            t.get("methods_count", "Методов"),
            str(len(METHODS)),
            "⚙️",
            "#2ecc71",
        )
    with col3:
        metric_card(
            t.get("pdk_label", "Нормативов ПДК"),
            str(len(METALS_CONFIG)),
            "📋",
            "#e74c3c",
        )
    with col4:
        metric_card(
            t.get("pages_count", "Разделов"),
            "8",
            "📑",
            "#f39c12",
        )

    st.markdown("")

    # ── О проекте ──
    st.markdown(f"### 📖 {t.get('about_title', 'О проекте')}")
    st.markdown(
        t.get(
            "about_text",
            "При концентрации тяжёлого металла в пробе **ниже предела обнаружения** "
            "(LOD — *Limit of Detection*) прибор не может выдать точное числовое значение. "
            "Такие данные называются **цензурированными слева** (*left-censored*).\n\n"
            "Простая замена на ноль, на LOD или на LOD/2 может **сильно исказить** "
            "среднее, дисперсию и квантили. Это приложение позволяет:\n\n"
            "1. **Сгенерировать** реалистичные синтетические данные по 6 металлам\n"
            "2. **Применить** 8 различных методов обработки цензурирования\n"
            "3. **Сравнить** точность каждого метода по смещению и RMSE\n"
            "4. **Исследовать**, как результаты зависят от доли цензурированных значений\n"
            "5. **Загрузить** собственные данные и проанализировать их",
        )
    )

    st.markdown("---")

    # ── Металлы ──
    st.markdown(f"### 🧪 {t.get('metals_title', 'Рассматриваемые металлы')}")

    metals_cols = st.columns(3)
    for i, (key, meta) in enumerate(METALS_CONFIG.items()):
        with metals_cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background: #1e293b;
                    border-left: 5px solid #3498db;
                    padding: 14px 18px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    border-top: 1px solid #334155;
                    border-right: 1px solid #334155;
                    border-bottom: 1px solid #334155;
                ">
                    <div style="font-size:1.1em; font-weight:bold; color:#ffffff; margin-bottom:4px;">
                        {meta['name_ru']} <span style="color:#94a3b8; font-size:0.9em;">({key.upper()})</span>
                    </div>
                    <div style="color:#ff6b6b; font-weight:600; font-size:0.95em;">
                        ПДК: {meta['pdk']} {meta['unit']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Методы ──
    st.markdown(f"### ⚙️ {t.get('methods_title', 'Доступные методы')}")

    methods_cols = st.columns(2)
    for i, (mkey, minfo) in enumerate(METHODS.items()):
        with methods_cols[i % 2]:
            st.markdown(
                f"""
                <div style="
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    background: #1e293b;
                    border: 1px solid #334155;
                ">
                    <span style="color:{minfo['color']}; font-weight:bold; font-size:1.05em;">
                        {minfo['icon']} {minfo['name_ru']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Быстрый старт ──
    st.markdown(f"### 🚀 {t.get('quickstart_title', 'Быстрый старт')}")
    st.markdown(
        t.get(
            "quickstart_text",
            "1. Перейдите в раздел **«📊 Обзор данных»** чтобы увидеть синтетические данные\n"
            "2. Откройте **«🧪 Эксперимент с цензурированием»** для интерактивного сравнения\n"
            "3. Используйте **«📈 Зависимость от % цензурирования»** для анализа устойчивости\n"
            "4. Проверьте **«🏆 Итоговый рейтинг»** для сводных результатов\n"
            "5. Запустите **«🎲 Монте-Карло»** для статистической валидации",
        )
    )