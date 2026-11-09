import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ══════════════════════════════════════════
#  ЦВЕТА
# ══════════════════════════════════════════
COLOR_PRIMARY = "#3498db"
COLOR_DANGER = "#e74c3c"
COLOR_SUCCESS = "#2ecc71"
COLOR_WARNING = "#f39c12"
COLOR_DARK = "#1a2332"
COLOR_ACCENT = "#2196F3"

# ══════════════════════════════════════════
#  КОНФИГУРАЦИЯ МЕТАЛЛОВ
# ══════════════════════════════════════════
METALS_CONFIG: Dict[str, dict] = {
    "cd": {
        "name_ru": "Кадмий",
        "name_en": "Cadmium",
        "pdk": 0.001,
        "unit": "мг/л",
        "lognorm_mu": -7.0,
        "lognorm_sigma": 1.5,
    },
    "pb": {
        "name_ru": "Свинец",
        "name_en": "Lead",
        "pdk": 0.01,
        "unit": "мг/л",
        "lognorm_mu": -5.0,
        "lognorm_sigma": 1.3,
    },
    "hg": {
        "name_ru": "Ртуть",
        "name_en": "Mercury",
        "pdk": 0.0005,
        "unit": "мг/л",
        "lognorm_mu": -8.0,
        "lognorm_sigma": 1.6,
    },
    "as": {
        "name_ru": "Мышьяк",
        "name_en": "Arsenic",
        "pdk": 0.01,
        "unit": "мг/л",
        "lognorm_mu": -5.5,
        "lognorm_sigma": 1.2,
    },
    "cu": {
        "name_ru": "Медь",
        "name_en": "Copper",
        "pdk": 1.0,
        "unit": "мг/л",
        "lognorm_mu": -1.0,
        "lognorm_sigma": 1.0,
    },
    "zn": {
        "name_ru": "Цинк",
        "name_en": "Zinc",
        "pdk": 1.0,
        "unit": "мг/л",
        "lognorm_mu": -0.5,
        "lognorm_sigma": 0.9,
    },
}

# ══════════════════════════════════════════
#  ФУНКЦИИ ОБРАБОТКИ ЦЕНЗУРИРОВАНИЯ
# ══════════════════════════════════════════

def _method_zero(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    result[result <= 0] = 0.0
    return result

def _method_lod(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    result[result <= 0] = lod
    return result

def _method_lod_half(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    result[result <= 0] = lod / 2.0
    return result

def _method_lod_sqrt2(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    result[result <= 0] = lod / np.sqrt(2)
    return result

def _method_kaplan_meier(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    mask = result <= 0
    n_cens = mask.sum()
    n_total = len(result)
    if n_cens == 0 or n_cens == n_total:
        result[mask] = lod / 2.0
        return result
    observed = result[~mask]
    observed_sorted = np.sort(observed)
    km_values = []
    n_remaining = n_total
    for val in observed_sorted:
        prob = (n_remaining - 1) / n_remaining if n_remaining > 0 else 0
        km_values.append(val * (1 - prob))
        n_remaining -= 1
    km_mean = np.mean(km_values) if km_values else lod / 2.0
    result[mask] = max(km_mean, lod * 0.1)
    return result

def _method_ros(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    mask = result <= 0
    n_cens = mask.sum()
    if n_cens == 0:
        return result
    observed = result[~mask]
    if len(observed) < 3:
        result[mask] = lod / 2.0
        return result
    log_obs = np.log(observed)
    mu, sigma = np.mean(log_obs), np.std(log_obs)
    if sigma < 1e-10:
        sigma = 0.1
    n_total = len(result)
    for i in range(n_cens):
        p = (i + 0.5) / n_total
        from scipy.stats import norm
        z = norm.ppf(p)
        imputed = np.exp(mu + sigma * z)
        imputed = min(imputed, lod * 0.99)
        imputed = max(imputed, 1e-15)
        censored_indices = np.where(mask)[0]
        if i < len(censored_indices):
            result[censored_indices[i]] = imputed
    return result

def _method_mle_lognormal(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    mask = result <= 0
    n_cens = mask.sum()
    if n_cens == 0:
        return result
    observed = result[~mask]
    if len(observed) < 2:
        result[mask] = lod / 2.0
        return result
    log_obs = np.log(observed)
    mu = np.mean(log_obs)
    sigma = np.std(log_obs)
    if sigma < 1e-10:
        sigma = 0.1
    from scipy.stats import lognorm
    censored_mean = lognorm.mean(s=sigma, scale=np.exp(mu))
    if np.isfinite(censored_mean) and censored_mean > 0:
        result[mask] = min(censored_mean * 0.5, lod * 0.5)
    else:
        result[mask] = lod / 2.0
    result[mask] = np.maximum(result[mask], 1e-15)
    return result

def _method_imputation_uniform(data: np.ndarray, lod: float) -> np.ndarray:
    result = data.copy()
    mask = result <= 0
    n_cens = mask.sum()
    if n_cens == 0:
        return result
    rng = np.random.default_rng(42)
    result[mask] = rng.uniform(1e-10, lod * 0.99, size=n_cens)
    return result

# ══════════════════════════════════════════
#  РЕЕСТР МЕТОДОВ
# ══════════════════════════════════════════
METHODS: Dict[str, dict] = {
    "zero": {
        "func": _method_zero,
        "name_ru": "Замена на 0",
        "name_en": "Replace with 0",
        "color": "#95a5a6",
        "icon": "⬛",
    },
    "lod": {
        "func": _method_lod,
        "name_ru": "Замена на LOD",
        "name_en": "Replace with LOD",
        "color": "#e67e22",
        "icon": "🟧",
    },
    "lod_half": {
        "func": _method_lod_half,
        "name_ru": "LOD / 2",
        "name_en": "LOD / 2",
        "color": "#3498db",
        "icon": "🔵",
    },
    "lod_sqrt2": {
        "func": _method_lod_sqrt2,
        "name_ru": "LOD / √2",
        "name_en": "LOD / √2",
        "color": "#9b59b6",
        "icon": "🟣",
    },
    "kaplan_meier": {
        "func": _method_kaplan_meier,
        "name_ru": "Каплана-Мейера",
        "name_en": "Kaplan-Meier",
        "color": "#2ecc71",
        "icon": "🟢",
    },
    "ros": {
        "func": _method_ros,
        "name_ru": "ROS (регрессия на порядковые статистики)",
        "name_en": "ROS",
        "color": "#1abc9c",
        "icon": "🔷",
    },
    "mle_lognormal": {
        "func": _method_mle_lognormal,
        "name_ru": "MLE (логнормальное)",
        "name_en": "MLE Lognormal",
        "color": "#e74c3c",
        "icon": "🔴",
    },
    "imputation_uniform": {
        "func": _method_imputation_uniform,
        "name_ru": "Случайное (равномерное)",
        "name_en": "Uniform Imputation",
        "color": "#f39c12",
        "icon": "🟡",
    },
}

# ══════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Тёмный сайдбар */
        section[data-testid="stSidebar"] {
            background-color: #1a2332 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #e0e0e0 !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #ffffff !important;
            font-size: 1.05em;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background-color: #2c3e50 !important;
            border-radius: 6px;
        }
        /* Основной контент */
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }
        /* DataFrame */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        /* Кнопки */
        .stButton > button {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════
def init_session_state():
    defaults = {
        "current_page": "home",
        "lang": "ru",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ══════════════════════════════════════════
#  ГЕНЕРАЦИЯ ДАННЫХ
# ══════════════════════════════════════════
def generate_synthetic_data(metal_key: str, n: int = 500, seed: int = 42) -> np.ndarray:
    if metal_key not in METALS_CONFIG:
        raise ValueError(f"Неизвестный металл: {metal_key}")
    meta = METALS_CONFIG[metal_key]
    rng = np.random.default_rng(seed)
    data = rng.lognormal(
        mean=meta["lognorm_mu"],
        sigma=meta["lognorm_sigma"],
        size=n,
    )
    return data

# ══════════════════════════════════════════
#  ЦЕНЗУРИРОВАНИЕ
# ══════════════════════════════════════════
def apply_censoring(
    data: np.ndarray, target_pct: float = 0.3
) -> Tuple[np.ndarray, float, float]:
    """
    Цензурирует данные слева.
    Возвращает: (censored_data, lod_value, actual_pct).
    """
    data = np.array(data, dtype=float)
    sorted_data = np.sort(data)
    n = len(data)
    idx = max(0, min(int(n * target_pct), n - 1))
    lod_value = sorted_data[idx]

    censored = data.copy()
    mask = censored < lod_value
    censored[mask] = 0.0
    actual_pct = mask.sum() / n * 100.0

    return censored, lod_value, actual_pct

# ══════════════════════════════════════════
#  СТАТИСТИКИ
# ══════════════════════════════════════════
def compute_statistics(data: np.ndarray) -> dict:
    data = np.array(data, dtype=float)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }

# ══════════════════════════════════════════
#  BIAS & RMSE
# ══════════════════════════════════════════
def compute_bias_rmse(true_stats: dict, proc_stats: dict) -> dict:
    def _rel_bias(key):
        t = true_stats[key]
        p = proc_stats[key]
        if abs(t) < 1e-15:
            return 0.0
        return (p - t) / abs(t)

    bias_mean = _rel_bias("mean")
    bias_median = _rel_bias("median")
    bias_std = _rel_bias("std")

    rmse = float(np.sqrt(np.mean([
        (proc_stats[k] - true_stats[k]) ** 2
        for k in ["mean", "median", "std"]
    ])))

    return {
        "bias_mean": bias_mean,
        "bias_median": bias_median,
        "bias_std": bias_std,
        "rmse": rmse,
    }

# ══════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ
# ══════════════════════════════════════════
def get_method_name(mkey: str, t: dict) -> str:
    lang = "ru" if t.get("_lang", "ru") == "ru" else "en"
    info = METHODS.get(mkey, {})
    if lang == "en":
        return info.get("name_en", mkey)
    return info.get("name_ru", mkey)

def get_metal_name(mkey: str, t: dict) -> str:
    lang = "ru" if t.get("_lang", "ru") == "ru" else "en"
    info = METALS_CONFIG.get(mkey, {})
    if lang == "en":
        return info.get("name_en", mkey)
    return info.get("name_ru", mkey)

def get_stat_name(stat_key: str, t: dict) -> str:
    mapping = {
        "mean": t.get("mean", "Среднее"),
        "median": t.get("median", "Медиана"),
        "std": t.get("std", "Ст. откл."),
        "q25": t.get("q25", "Q25"),
        "q75": t.get("q75", "Q75"),
        "min": t.get("min", "Мин."),
        "max": t.get("max", "Макс."),
    }
    return mapping.get(stat_key, stat_key)

def metric_card(label: str, value: str, icon: str = "", color: str = "#3498db"):
    st.markdown(
        f"""
        <div style="
            background: #1e293b;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        ">
            <div style="color:#94a3b8; font-size:0.85em; margin-bottom:4px;">
                {icon} {label}
            </div>
            <div style="color:#ffffff; font-size:1.5em; font-weight:700;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    **kwargs,
) -> dict:
    layout = {
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "font": dict(size=13, family="Arial", color="#1a2332"),  # Принудительно тёмный шрифт
        "title": dict(text=title, font=dict(size=16, color="#1a2332")),
        "margin": dict(l=60, r=30, t=60, b=60),
        "hoverlabel": dict(bgcolor="#1a2332", font_size=12, font_color="#ffffff"),
        "legend": dict(
            font=dict(color="#1a2332", size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e0e0e0",
            borderwidth=1,
        ),
        "xaxis": dict(
            title=dict(text=xaxis_title, font=dict(color="#1a2332", size=13)),
            tickfont=dict(color="#1a2332", size=12),
            gridcolor="#e2e8f0",
            zerolinecolor="#94a3b8",
            showgrid=True,
        ),
        "yaxis": dict(
            title=dict(text=yaxis_title, font=dict(color="#1a2332", size=13)),
            tickfont=dict(color="#1a2332", size=12),
            gridcolor="#e2e8f0",
            zerolinecolor="#94a3b8",
            showgrid=True,
        ),
    }
    for key in ("xaxis", "yaxis"):
        if key in kwargs:
            layout[key].update(kwargs.pop(key))
    layout.update(kwargs)
    return layout

# ══════════════════════════════════════════
#  ЛОКАЛИЗАЦИЯ
# ══════════════════════════════════════════
def get_translations(lang: str) -> dict:
    t = {"_lang": lang}
    if lang == "en":
        t.update({
            "home_title": "Censored Data Analysis",
            "home_subtitle": "Comparison of methods for handling heavy metal concentrations below LOD",
            "metals_count": "Metals",
            "methods_count": "Methods",
            "pdk_label": "MPC Standards",
            "pages_count": "Sections",
            "about_title": "About",
            "metals_title": "Metals",
            "methods_title": "Methods",
            "quickstart_title": "Quick Start",
            "select_metal": "Metal",
            "censor_pct": "Censoring (%)",
            "n_samples": "N",
            "mean_label": "Mean (orig.)",
            "mean_cens_label": "Mean (cens.)",
            "median_label": "Median (orig.)",
            "actual_censor": "Actual % cens.",
            "original": "Original",
            "censored": "Censored",
            "concentration": "Concentration",
            "count": "Count",
            "mean": "Mean",
            "median": "Median",
            "std": "Std Dev",
            "q25": "Q25",
            "q75": "Q75",
            "min": "Min",
            "max": "Max",
            "method": "Method",
            "bias": "Bias",
            "rmse": "RMSE",
            "computing": "Computing...",
            "best_method": "Best method",
            "select_methods": "Methods",
            "select_metals": "Metals",
            "scenarios": "Scenarios",
            "avg_rank": "Avg Rank",
            "n_iterations": "Iterations",
            "base_seed": "Base seed",
            "upload_label": "Upload CSV or Excel",
            "load_demo": "Load demo data",
        })
    else:
        t.update({
            "home_title": "Анализ цензурированных данных",
            "home_subtitle": "Сравнение методов обработки концентраций тяжёлых металлов ниже LOD",
            "metals_count": "Металлов",
            "methods_count": "Методов",
            "pdk_label": "Нормативов ПДК",
            "pages_count": "Разделов",
            "about_title": "О проекте",
            "metals_title": "Рассматриваемые металлы",
            "methods_title": "Доступные методы",
            "quickstart_title": "Быстрый старт",
            "select_metal": "Металл",
            "censor_pct": "Доля цензурирования (%)",
            "n_samples": "N",
            "mean_label": "Среднее (исх.)",
            "mean_cens_label": "Среднее (ценз.)",
            "median_label": "Медиана (исх.)",
            "actual_censor": "Факт. % ценз.",
            "original": "Исходные",
            "censored": "Цензурированные",
            "concentration": "Концентрация",
            "count": "Частота",
            "mean": "Среднее",
            "median": "Медиана",
            "std": "Ст. откл.",
            "q25": "Q25",
            "q75": "Q75",
            "min": "Мин.",
            "max": "Макс.",
            "method": "Метод",
            "bias": "Смещение",
            "rmse": "RMSE",
            "computing": "Вычисление...",
            "best_method": "Лучший метод",
            "select_methods": "Методы для сравнения",
            "select_metals": "Металлы",
            "scenarios": "Сценариев",
            "avg_rank": "Ср. ранг",
            "n_iterations": "Итераций",
            "base_seed": "Базовый seed",
            "upload_label": "Выберите CSV или Excel",
            "load_demo": "Загрузить демо-данные",
        })
    return t