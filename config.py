"""
config.py — Конфигурационный файл для проекта анализа цензурированных данных
тяжёлых металлов в объектах окружающей среды.

Содержит:
    - ПДК тяжёлых металлов (СанПиН 1.2.3685-21)
    - Параметры цензурирования
    - Список методов обработки цензурированных данных
    - Метрики оценки качества
    - Параметры Монте-Карло симуляций
    - Параметры генерации синтетических данных
    - Настройки визуализации
    - Настройки веб-приложения

Python >= 3.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final


# =============================================================================
# 1. ТЯЖЁЛЫЕ МЕТАЛЛЫ И ИХ ПДК
# =============================================================================
"""
Словарь тяжёлых металлов с их символами, предельно допустимыми концентрациями
(ПДК) для питьевой воды согласно СанПиН 1.2.3685-21 и единицами измерения.
"""

HEAVY_METALS: Final[dict[str, dict[str, Any]]] = {
    "Свинец":   {"symbol": "Pb", "mpc": 0.01,   "unit": "мг/л"},
    "Кадмий":   {"symbol": "Cd", "mpc": 0.001,  "unit": "мг/л"},
    "Ртуть":    {"symbol": "Hg", "mpc": 0.0005, "unit": "мг/л"},
    "Мышьяк":   {"symbol": "As", "mpc": 0.01,   "unit": "мг/л"},
    "Хром":     {"symbol": "Cr", "mpc": 0.05,   "unit": "мг/л"},
    "Никель":   {"symbol": "Ni", "mpc": 0.02,   "unit": "мг/л"},
    "Медь":     {"symbol": "Cu", "mpc": 1.0,    "unit": "мг/л"},
    "Цинк":     {"symbol": "Zn", "mpc": 5.0,    "unit": "мг/л"},
    "Марганец": {"symbol": "Mn", "mpc": 0.1,    "unit": "мг/л"},
    "Железо":   {"symbol": "Fe", "mpc": 0.3,    "unit": "мг/л"},
}


# =============================================================================
# 2. ПАРАМЕТРЫ ЦЕНЗУРИРОВАНИЯ
# =============================================================================
"""
Параметры для моделирования различных схем цензурирования данных:
доли, типы и механизмы цензурирования.
"""

CENSORING_FRACTIONS: Final[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

CENSORING_TYPES: Final[dict[str, list[str]]] = {
    "primary":   ["left"],              # основной: левое цензурирование (< LOD)
    "secondary": ["right", "interval"], # дополнительные типы
}

CENSORING_MECHANISMS: Final[dict[str, str]] = {
    "fixed_lod":    "Единый порог обнаружения (LOD) для всей выборки",
    "variable_lod": "Несколько порогов LOD (имитация разных лабораторий/приборов)",
    "random_lod":   "Случайный порог LOD для каждого наблюдения",
}


# =============================================================================
# 3. МЕТОДЫ ОБРАБОТКИ ЦЕНЗУРИРОВАННЫХ ДАННЫХ
# =============================================================================
"""
Список реализуемых методов обработки цензурированных наблюдений с краткими
описаниями. Включает простые подстановки, параметрические (MLE),
полупараметрические (ROS), непараметрические (K-M) и байесовские подходы.
"""

METHODS: Final[dict[str, str]] = {
    "substitution_zero":      "Подстановка нуля вместо значений ниже LOD",
    "substitution_lod":       "Подстановка значения LOD вместо цензурированных",
    "substitution_half_lod":  "Подстановка LOD/2 (рекомендация ВОЗ)",
    "substitution_lod_sqrt2": "Подстановка LOD/√2 (рекомендация EPA)",
    "mle_normal":             "Maximum Likelihood Estimation (нормальное распределение)",
    "mle_lognormal":          "Maximum Likelihood Estimation (логнормальное распределение)",
    "mle_gamma":              "Maximum Likelihood Estimation (гамма-распределение)",
    "ros_lognormal":          "Regression on Order Statistics (логнормальное)",
    "ros_normal":             "Regression on Order Statistics (нормальное)",
    "kaplan_meier":           "Непараметрическая оценка Каплана-Мейера",
    "bayesian_lognormal":     "Байесовский подход с логнормальным априорным",
    "multiple_imputation":    "Множественная импутация (MICE)",
    "bootstrap_mle":          "Бутстрап-оценка в сочетании с MLE",
}


# =============================================================================
# 4. МЕТРИКИ ОЦЕНКИ КАЧЕСТВА
# =============================================================================
"""
Метрики для сравнения методов обработки цензурированных данных:
смещения оценок статистик, RMSE, вероятности покрытия ДИ и др.
"""

METRICS: Final[dict[str, str]] = {
    "relative_bias_mean":          "Относительное смещение среднего (%)",
    "relative_bias_variance":      "Относительное смещение дисперсии (%)",
    "relative_bias_p90":           "Относительное смещение 90-го перцентиля (%)",
    "relative_bias_p95":           "Относительное смещение 95-го перцентиля (%)",
    "rmse_mean":                   "RMSE оценки среднего",
    "rmse_variance":               "RMSE оценки дисперсии",
    "coverage_probability":        "Вероятность покрытия 95% доверительного интервала",
    "exceedance_probability_error":"Ошибка оценки вероятности превышения ПДК",
    "ks_statistic":                "Статистика Колмогорова-Смирнова",
}


# =============================================================================
# 5. ПАРАМЕТРЫ МОНТЕ-КАРЛО СИМУЛЯЦИИ
# =============================================================================
"""
Параметры воспроизводимых симуляций Монте-Карло для оценки методов.
"""

@dataclass(frozen=True)
class MonteCarloConfig:
    """Конфигурация Монте-Карло симуляций."""
    n_simulations: int = 1000
    sample_sizes: list[int] = field(default_factory=lambda: [30, 50, 100, 200, 500])
    random_seed: int = 42


MONTE_CARLO: Final[MonteCarloConfig] = MonteCarloConfig()


# =============================================================================
# 6. ПАРАМЕТРЫ ГЕНЕРАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ
# =============================================================================
"""
Параметры логнормального распределения для каждого металла, подобранные так,
чтобы медиана концентрации составляла ~0.3-0.7 от ПДК. Также параметры
генерации выбросов (доля и диапазон множителя).

Медиана логнормального распределения: median = exp(mu)
Поэтому mu = ln(target_median), target_median ≈ 0.5 * MPC
"""

import math

def _lognormal_params(mpc: float, median_fraction: float = 0.5,
                      sigma: float = 0.8) -> dict[str, float]:
    """Вспомогательная функция подбора параметров логнормального распределения."""
    target_median = median_fraction * mpc
    mu = math.log(target_median)
    return {"mu": round(mu, 4), "sigma": sigma, "target_median": target_median}


SYNTHETIC_DATA_PARAMS: Final[dict[str, dict[str, float]]] = {
    "Свинец":   _lognormal_params(0.01,   median_fraction=0.5, sigma=0.8),
    "Кадмий":   _lognormal_params(0.001,  median_fraction=0.4, sigma=0.9),
    "Ртуть":    _lognormal_params(0.0005, median_fraction=0.3, sigma=1.0),
    "Мышьяк":   _lognormal_params(0.01,   median_fraction=0.5, sigma=0.8),
    "Хром":     _lognormal_params(0.05,   median_fraction=0.5, sigma=0.7),
    "Никель":   _lognormal_params(0.02,   median_fraction=0.6, sigma=0.7),
    "Медь":     _lognormal_params(1.0,    median_fraction=0.6, sigma=0.6),
    "Цинк":     _lognormal_params(5.0,    median_fraction=0.7, sigma=0.6),
    "Марганец": _lognormal_params(0.1,    median_fraction=0.5, sigma=0.8),
    "Железо":   _lognormal_params(0.3,    median_fraction=0.6, sigma=0.7),
}

OUTLIERS_CONFIG: Final[dict[str, Any]] = {
    "enabled":          True,
    "fraction":         0.02,  # доля выбросов
    "multiplier_range": (3.0, 10.0),  # диапазон множителя относительно верхнего квантиля
}


# =============================================================================
# 7. НАСТРОЙКИ ВИЗУАЛИЗАЦИИ
# =============================================================================
"""
Настройки для построения графиков: цветовая палитра методов, размеры фигур,
стили matplotlib, параметры шрифтов.
"""

METHOD_COLORS: Final[dict[str, str]] = {
    "substitution_zero":      "#808080",  # серый
    "substitution_lod":       "#A0522D",  # коричневый
    "substitution_half_lod":  "#FF8C00",  # оранжевый
    "substitution_lod_sqrt2": "#FFD700",  # золотой
    "mle_normal":             "#1f77b4",  # синий
    "mle_lognormal":          "#2ca02c",  # зелёный
    "mle_gamma":              "#17becf",  # бирюзовый
    "ros_lognormal":          "#9467bd",  # фиолетовый
    "ros_normal":             "#8c564b",  # тёмно-коричневый
    "kaplan_meier":           "#e377c2",  # розовый
    "bayesian_lognormal":     "#d62728",  # красный
    "multiple_imputation":    "#bcbd22",  # оливковый
    "bootstrap_mle":          "#7f7f7f",  # тёмно-серый
}

PLOT_CONFIG: Final[dict[str, Any]] = {
    "figure_size":       (10, 6),
    "figure_size_large": (14, 8),
    "figure_size_small": (6, 4),
    "style":             "seaborn-v0_8-whitegrid",
    "dpi":               150,
    "font_family":       "DejaVu Sans",
    "font_size":         12,
    "title_size":        14,
    "label_size":        12,
    "legend_size":       10,
}


# =============================================================================
# 8. НАСТРОЙКИ ВЕБ-ПРИЛОЖЕНИЯ
# =============================================================================
"""
Настройки для развёртывания Streamlit/Flask веб-приложения.
"""

@dataclass(frozen=True)
class WebAppConfig:
    """Конфигурация веб-приложения."""
    HOST: str = "0.0.0.0"
    PORT: int = 8501
    DEBUG: bool = True
    MAX_UPLOAD_SIZE_MB: int = 50


WEB_APP: Final[WebAppConfig] = WebAppConfig()


# =============================================================================
# API: получение и валидация конфигурации
# =============================================================================

def get_config() -> dict[str, Any]:
    """
    Возвращает полный словарь конфигурации проекта.

    Returns:
        dict: словарь со всеми параметрами конфигурации.
    """
    return {
        "heavy_metals":          HEAVY_METALS,
        "censoring_fractions":   CENSORING_FRACTIONS,
        "censoring_types":       CENSORING_TYPES,
        "censoring_mechanisms":  CENSORING_MECHANISMS,
        "methods":               METHODS,
        "metrics":               METRICS,
        "monte_carlo": {
            "n_simulations": MONTE_CARLO.n_simulations,
            "sample_sizes":  MONTE_CARLO.sample_sizes,
            "random_seed":   MONTE_CARLO.random_seed,
        },
        "synthetic_data_params": SYNTHETIC_DATA_PARAMS,
        "outliers_config":       OUTLIERS_CONFIG,
        "method_colors":         METHOD_COLORS,
        "plot_config":           PLOT_CONFIG,
        "web_app": {
            "HOST":               WEB_APP.HOST,
            "PORT":               WEB_APP.PORT,
            "DEBUG":              WEB_APP.DEBUG,
            "MAX_UPLOAD_SIZE_MB": WEB_APP.MAX_UPLOAD_SIZE_MB,
        },
    }


def validate_config() -> bool:
    """
    Проверяет корректность параметров конфигурации.

    Проверки:
        - ПДК всех металлов положительны
        - Доли цензурирования в (0, 1)
        - Количество симуляций > 0
        - Размеры выборок положительны
        - Все методы имеют назначенные цвета
        - Параметры логнормального распределения заданы для всех металлов
        - Доля выбросов в [0, 1)
        - PORT в допустимом диапазоне

    Returns:
        bool: True, если конфигурация валидна.

    Raises:
        ValueError: при обнаружении некорректных параметров.
    """
    # 1. ПДК
    for name, info in HEAVY_METALS.items():
        if info["mpc"] <= 0:
            raise ValueError(f"ПДК для '{name}' должна быть > 0, получено {info['mpc']}")
        if "symbol" not in info or "unit" not in info:
            raise ValueError(f"Для металла '{name}' отсутствуют поля symbol/unit")

    # 2. Доли цензурирования
    for frac in CENSORING_FRACTIONS:
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Доля цензурирования должна быть в (0, 1), получено {frac}")

    # 3. Монте-Карло
    if MONTE_CARLO.n_simulations <= 0:
        raise ValueError("n_simulations должно быть > 0")
    if not all(n > 0 for n in MONTE_CARLO.sample_sizes):
        raise ValueError("Все размеры выборок должны быть > 0")

    # 4. Соответствие методов и цветов
    methods_set = set(METHODS.keys())
    colors_set  = set(METHOD_COLORS.keys())
    missing = methods_set - colors_set
    if missing:
        raise ValueError(f"Не заданы цвета для методов: {missing}")

    # 5. Параметры синтетических данных
    for metal in HEAVY_METALS:
        if metal not in SYNTHETIC_DATA_PARAMS:
            raise ValueError(f"Не заданы параметры генерации для '{metal}'")
        params = SYNTHETIC_DATA_PARAMS[metal]
        if params["sigma"] <= 0:
            raise ValueError(f"sigma для '{metal}' должна быть > 0")

    # 6. Выбросы
    if not (0.0 <= OUTLIERS_CONFIG["fraction"] < 1.0):
        raise ValueError("Доля выбросов должна быть в [0, 1)")
    low, high = OUTLIERS_CONFIG["multiplier_range"]
    if low <= 0 or high <= low:
        raise ValueError("Некорректный диапазон множителя выбросов")

    # 7. Веб-приложение
    if not (1 <= WEB_APP.PORT <= 65535):
        raise ValueError(f"PORT должен быть в диапазоне 1-65535, получено {WEB_APP.PORT}")
    if WEB_APP.MAX_UPLOAD_SIZE_MB <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_MB должно быть > 0")

    return True


if __name__ == "__main__":
    # Самотестирование при запуске файла напрямую
    try:
        validate_config()
        print("✓ Конфигурация валидна")
        cfg = get_config()
        print(f"✓ Загружено {len(cfg['heavy_metals'])} металлов")
        print(f"✓ Загружено {len(cfg['methods'])} методов обработки")
        print(f"✓ Загружено {len(cfg['metrics'])} метрик оценки")
        print(f"✓ Параметры Монте-Карло: {cfg['monte_carlo']}")
    except ValueError as e:
        print(f"✗ Ошибка в конфигурации: {e}")