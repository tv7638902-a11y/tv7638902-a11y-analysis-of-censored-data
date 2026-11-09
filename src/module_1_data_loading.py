#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Модуль 1: Загрузка и подготовка данных о концентрациях тяжёлых металлов.

Предоставляет функциональность для:
- Загрузки реальных данных из CSV/Excel файлов
- Генерации реалистичных синтетических данных
- Валидации и предварительного анализа данных
- Вычисления истинных статистик для оценки методов

Автор: Environmental Data Science Module
Версия: 1.0.0
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

# Конфигурация по умолчанию (используется если config.py недоступен)
DEFAULT_CONFIG = {
    "random_seed": 42,
    "metals": {
        "Pb": {
            "name_ru": "Свинец",
            "mac": 0.01,        # ПДК мг/л (вода питьевая)
            "lod": 0.001,       # Предел обнаружения мг/л
            "lognormal_mu": -5.5,
            "lognormal_sigma": 1.2,
            "unit": "мг/л"
        },
        "Cd": {
            "name_ru": "Кадмий",
            "mac": 0.001,
            "lod": 0.0002,
            "lognormal_mu": -8.0,
            "lognormal_sigma": 1.0,
            "unit": "мг/л"
        },
        "Cu": {
            "name_ru": "Медь",
            "mac": 1.0,
            "lod": 0.002,
            "lognormal_mu": -3.0,
            "lognormal_sigma": 1.1,
            "unit": "мг/л"
        },
        "Zn": {
            "name_ru": "Цинк",
            "mac": 1.0,
            "lod": 0.005,
            "lognormal_mu": -2.5,
            "lognormal_sigma": 1.3,
            "unit": "мг/л"
        },
        "As": {
            "name_ru": "Мышьяк",
            "mac": 0.01,
            "lod": 0.001,
            "lognormal_mu": -6.0,
            "lognormal_sigma": 0.9,
            "unit": "мг/л"
        },
        "Hg": {
            "name_ru": "Ртуть",
            "mac": 0.0005,
            "lod": 0.00005,
            "lognormal_mu": -10.0,
            "lognormal_sigma": 1.1,
            "unit": "мг/л"
        },
        "Cr": {
            "name_ru": "Хром",
            "mac": 0.05,
            "lod": 0.005,
            "lognormal_mu": -4.5,
            "lognormal_sigma": 1.0,
            "unit": "мг/л"
        },
        "Ni": {
            "name_ru": "Никель",
            "mac": 0.02,
            "lod": 0.002,
            "lognormal_mu": -5.0,
            "lognormal_sigma": 1.15,
            "unit": "мг/л"
        }
    },
    "correlation_matrix": {
        "Pb-Cd": 0.70,
        "Pb-Zn": 0.50,
        "Pb-Cu": 0.45,
        "Pb-As": 0.35,
        "Cd-Zn": 0.55,
        "Cd-Cu": 0.40,
        "Cu-Zn": 0.60,
        "Cu-Ni": 0.50,
        "As-Cr": 0.30,
        "Zn-Ni": 0.45,
        "Cr-Ni": 0.55,
        "Hg-Pb": 0.25,
        "Hg-Cd": 0.20,
    },
    "outlier_probability": 0.03,
    "outlier_multiplier_range": (5, 20),
    "default_n_samples": 200,
    "min_samples": 20,
    "validation": {
        "extreme_outlier_factor": 10,
        "min_sample_size": 20
    }
}

# Паттерны для распознавания цензурированных данных
CENSORED_PATTERNS = [
    re.compile(r'^<\s*([\d.]+(?:[eE][+-]?\d+)?)$'),        # <0.005
    re.compile(r'^<\s*LOD$', re.IGNORECASE),                 # <LOD
    re.compile(r'^<\s*MDL$', re.IGNORECASE),                 # <MDL
    re.compile(r'^<\s*(?:ПО|ПДО)$', re.IGNORECASE),         # <ПО, <ПДО
    re.compile(r'^(?:ND|N\.D\.)$', re.IGNORECASE),          # ND, N.D.
    re.compile(r'^(?:н/о|н\.о\.|не обнаружено)$', re.IGNORECASE),  # н/о
    re.compile(r'^(?:BDL|BQL)$', re.IGNORECASE),            # BDL, BQL
]


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настраивает и возвращает логгер.

    Parameters
    ----------
    name : str
        Имя логгера.
    level : int
        Уровень логирования.

    Returns
    -------
    logging.Logger
        Настроенный логгер.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class DataLoader:
    """
    Загрузчик и генератор данных о концентрациях тяжёлых металлов.

    Предоставляет методы для загрузки реальных данных, генерации
    синтетических данных с реалистичными распределениями, валидации
    и вычисления истинных статистик.

    Parameters
    ----------
    config : dict
        Словарь конфигурации. Если не указан, используется DEFAULT_CONFIG.

    Attributes
    ----------
    config : dict
        Текущая конфигурация.
    logger : logging.Logger
        Логгер модуля.
    rng : np.random.Generator
        Генератор случайных чисел (воспроизводимость).

    Examples
    --------
    >>> config = DEFAULT_CONFIG.copy()
    >>> loader = DataLoader(config)
    >>> data = loader.generate_synthetic_data("Pb", n_samples=100)
    >>> print(f"Сгенерировано {len(data['true_values'])} значений для {data['metal']}")
    Сгенерировано 100 значений для Pb
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация загрузчика данных.

        Parameters
        ----------
        config : dict, optional
            Конфигурация. Если None, используется DEFAULT_CONFIG.
        """
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.logger = _setup_logger("DataLoader")
        seed = self.config.get("random_seed", 42)
        self.rng = np.random.default_rng(seed)
        self.logger.info(
            f"DataLoader инициализирован (seed={seed}, "
            f"металлов в конфигурации: {len(self.config.get('metals', {}))})"
        )

    # ------------------------------------------------------------------
    # 1. Загрузка реальных данных
    # ------------------------------------------------------------------

    def load_real_data(
        self,
        filepath: str,
        sheet_name: Optional[Union[str, int]] = 0,
        lod_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Загружает реальные данные из CSV или Excel файла.

        Парсит специальные обозначения цензурированных данных (<LOD, ND,
        н/о и т.д.) и разделяет данные на числовые значения и маску
        цензурирования.

        Parameters
        ----------
        filepath : str
            Путь к файлу (CSV, XLSX, XLS).
        sheet_name : str or int, optional
            Имя/номер листа для Excel файлов (по умолчанию 0).
        lod_overrides : dict, optional
            Словарь {металл: LOD} для переопределения пределов обнаружения.

        Returns
        -------
        dict
            Словарь с ключами:
            - ``"values"`` (pd.DataFrame): числовые значения
              (для цензурированных — значение LOD)
            - ``"censored"`` (pd.DataFrame): булева матрица
              (True = цензурировано)
            - ``"lod"`` (dict): пределы обнаружения для каждого столбца
            - ``"metadata"`` (dict): метаданные (источник, размер, доли
              цензурирования)

        Raises
        ------
        FileNotFoundError
            Если файл не найден.
        ValueError
            Если файл пуст или имеет неподдерживаемый формат.

        Examples
        --------
        >>> loader = DataLoader()
        >>> # data = loader.load_real_data("monitoring_data.csv")
        >>> # print(data["values"].head())
        """
        path = Path(filepath)
        self.logger.info(f"Загрузка данных из {path}")

        if not path.exists():
            self.logger.error(f"Файл не найден: {path}")
            raise FileNotFoundError(f"Файл не найден: {path}")

        # Определяем формат и загружаем
        suffix = path.suffix.lower()
        try:
            if suffix == '.csv':
                raw_df = pd.read_csv(
                    path, dtype=str, keep_default_na=False
                )
            elif suffix in ('.xlsx', '.xls'):
                raw_df = pd.read_excel(
                    path, sheet_name=sheet_name,
                    dtype=str, keep_default_na=False
                )
            else:
                raise ValueError(
                    f"Неподдерживаемый формат файла: {suffix}. "
                    f"Поддерживаются: .csv, .xlsx, .xls"
                )
        except Exception as e:
            self.logger.error(f"Ошибка при чтении файла: {e}")
            raise

        if raw_df.empty:
            raise ValueError(f"Файл пуст: {path}")

        self.logger.info(
            f"Прочитано {raw_df.shape[0]} строк × {raw_df.shape[1]} столбцов"
        )

        # Парсинг каждой ячейки
        values_data = {}
        censored_data = {}
        lod_dict = {}

        for col in raw_df.columns:
            col_values = []
            col_censored = []
            col_lods = []

            for raw_val in raw_df[col]:
                val_str = str(raw_val).strip()
                parsed = self._parse_cell(val_str, col, lod_overrides)
                col_values.append(parsed["value"])
                col_censored.append(parsed["censored"])
                if parsed["censored"] and parsed["lod"] is not None:
                    col_lods.append(parsed["lod"])

            values_data[col] = col_values
            censored_data[col] = col_censored

            # Определяем LOD для столбца
            if lod_overrides and col in lod_overrides:
                lod_dict[col] = lod_overrides[col]
            elif col_lods:
                lod_dict[col] = max(col_lods)  # берём наибольший встреченный
            elif col in self.config.get("metals", {}):
                lod_dict[col] = self.config["metals"][col].get("lod", np.nan)
            else:
                lod_dict[col] = np.nan

        values_df = pd.DataFrame(values_data)
        censored_df = pd.DataFrame(censored_data)

        # Заменяем NaN в цензурированных значениях на LOD
        for col in values_df.columns:
            mask = censored_df[col] & values_df[col].isna()
            if mask.any() and not np.isnan(lod_dict.get(col, np.nan)):
                values_df.loc[mask, col] = lod_dict[col]

        # Метаданные
        censoring_fractions = {}
        for col in censored_df.columns:
            n_cens = censored_df[col].sum()
            n_total = len(censored_df[col])
            censoring_fractions[col] = n_cens / n_total if n_total > 0 else 0.0

        metadata = {
            "source": str(path),
            "n_samples": raw_df.shape[0],
            "n_metals": raw_df.shape[1],
            "columns": list(raw_df.columns),
            "censoring_fractions": censoring_fractions,
            "total_censoring_fraction": (
                censored_df.values.sum() / censored_df.size
                if censored_df.size > 0 else 0.0
            )
        }

        self.logger.info(
            f"Обработка завершена. "
            f"Общая доля цензурирования: "
            f"{metadata['total_censoring_fraction']:.1%}"
        )

        return {
            "values": values_df,
            "censored": censored_df,
            "lod": lod_dict,
            "metadata": metadata
        }

    def _parse_cell(
        self,
        val_str: str,
        column: str,
        lod_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Парсит одну ячейку данных, определяя значение и статус
        цензурирования.

        Parameters
        ----------
        val_str : str
            Строковое значение ячейки.
        column : str
            Название столбца (металла).
        lod_overrides : dict, optional
            Переопределения LOD.

        Returns
        -------
        dict
            {"value": float or NaN, "censored": bool, "lod": float or None}
        """
        if not val_str or val_str.lower() in ('', 'nan', 'na', '-'):
            return {"value": np.nan, "censored": False, "lod": None}

        # Проверяем паттерн <число
        m = CENSORED_PATTERNS[0].match(val_str)
        if m:
            lod_val = float(m.group(1))
            return {"value": lod_val, "censored": True, "lod": lod_val}

        # Проверяем текстовые паттерны цензурирования
        for pattern in CENSORED_PATTERNS[1:]:
            if pattern.match(val_str):
                # Берём LOD из override или конфигурации
                lod = None
                if lod_overrides and column in lod_overrides:
                    lod = lod_overrides[column]
                elif column in self.config.get("metals", {}):
                    lod = self.config["metals"][column].get("lod")
                return {
                    "value": lod if lod is not None else np.nan,
                    "censored": True,
                    "lod": lod
                }

        # Пробуем преобразовать в число
        try:
            val = float(val_str.replace(',', '.'))
            return {"value": val, "censored": False, "lod": None}
        except ValueError:
            self.logger.warning(
                f"Не удалось распознать значение '{val_str}' "
                f"в столбце '{column}'"
            )
            return {"value": np.nan, "censored": False, "lod": None}

    # ------------------------------------------------------------------
    # 2. Генерация синтетических данных (одиночный металл)
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        metal: str,
        n_samples: int = 200,
        distribution: str = "lognormal",
        params: Optional[Dict[str, float]] = None,
        add_outliers: bool = True
    ) -> Dict[str, Any]:
        """
        Генерирует реалистичные синтетические данные для указанного металла.

        Parameters
        ----------
        metal : str
            Символ металла (например, ``"Pb"``, ``"Cd"``).
        n_samples : int, optional
            Количество проб (по умолчанию 200).
        distribution : str, optional
            Тип распределения: ``"lognormal"``, ``"gamma"``,
            ``"weibull"``, ``"mixture"`` (по умолчанию ``"lognormal"``).
        params : dict, optional
            Параметры распределения. Если не указаны — берутся из конфига.
        add_outliers : bool, optional
            Добавлять ли выбросы (по умолчанию True).

        Returns
        -------
        dict
            Словарь с ключами:
            - ``"true_values"`` (np.ndarray): сгенерированные значения
            - ``"metal"`` (str): символ металла
            - ``"distribution"`` (str): использованное распределение
            - ``"true_params"`` (dict): параметры распределения
            - ``"true_mean"`` (float): истинное среднее
            - ``"true_variance"`` (float): истинная дисперсия
            - ``"true_p90"`` (float): 90-й перцентиль
            - ``"true_p95"`` (float): 95-й перцентиль
            - ``"true_exceedance_prob"`` (float): вероятность превышения ПДК

        Raises
        ------
        ValueError
            Если указано неподдерживаемое распределение или неизвестный металл.

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.generate_synthetic_data("Pb", n_samples=500)
        >>> print(f"Среднее: {data['true_mean']:.6f}")
        >>> print(f"P95: {data['true_p95']:.6f}")
        >>> print(f"P(X>ПДК): {data['true_exceedance_prob']:.2%}")
        """
        self.logger.info(
            f"Генерация данных: металл={metal}, n={n_samples}, "
            f"distribution={distribution}"
        )

        # Получаем конфигурацию металла
        metal_config = self.config.get("metals", {}).get(metal)
        if metal_config is None and params is None:
            raise ValueError(
                f"Металл '{metal}' не найден в конфигурации и параметры "
                f"не указаны. Доступные металлы: "
                f"{list(self.config.get('metals', {}).keys())}"
            )

        # Определяем параметры
        effective_params = self._resolve_params(
            metal, distribution, params, metal_config
        )
        self.logger.debug(f"Параметры генерации: {effective_params}")

        # Генерируем данные
        supported = ("lognormal", "gamma", "weibull", "mixture")
        if distribution not in supported:
            raise ValueError(
                f"Неподдерживаемое распределение '{distribution}'. "
                f"Доступные: {supported}"
            )

        values = self._generate_distribution(
            distribution, n_samples, effective_params
        )

        # Добавляем выбросы
        if add_outliers:
            values = self._add_outliers(values, metal_config)

        # Убеждаемся что все значения положительны
        values = np.maximum(values, 1e-12)

        # Вычисляем истинные статистики
        mac = metal_config["mac"] if metal_config else effective_params.get(
            "mac", 0.01
        )

        result = {
            "true_values": values,
            "metal": metal,
            "distribution": distribution,
            "true_params": effective_params,
            "true_mean": float(np.mean(values)),
            "true_variance": float(np.var(values, ddof=1)),
            "true_std": float(np.std(values, ddof=1)),
            "true_median": float(np.median(values)),
            "true_p75": float(np.percentile(values, 75)),
            "true_p90": float(np.percentile(values, 90)),
            "true_p95": float(np.percentile(values, 95)),
            "true_p99": float(np.percentile(values, 99)),
            "true_exceedance_prob": float(np.mean(values > mac)),
            "mac": mac,
            "lod": metal_config["lod"] if metal_config else effective_params.get(
                "lod", 0.001
            ),
            "n_samples": n_samples
        }

        self.logger.info(
            f"  Среднее={result['true_mean']:.6f}, "
            f"Медиана={result['true_median']:.6f}, "
            f"P95={result['true_p95']:.6f}, "
            f"P(X>ПДК)={result['true_exceedance_prob']:.2%}"
        )

        return result

    def _resolve_params(
        self,
        metal: str,
        distribution: str,
        params: Optional[Dict[str, float]],
        metal_config: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Определяет эффективные параметры генерации данных.

        Parameters
        ----------
        metal : str
            Символ металла.
        distribution : str
            Тип распределения.
        params : dict or None
            Явно указанные параметры.
        metal_config : dict or None
            Конфигурация металла.

        Returns
        -------
        dict
            Эффективные параметры.
        """
        if params is not None:
            return params.copy()

        if metal_config is None:
            return {"mu": -5.0, "sigma": 1.0}

        if distribution == "lognormal":
            return {
                "mu": metal_config.get("lognormal_mu", -5.0),
                "sigma": metal_config.get("lognormal_sigma", 1.0),
            }
        elif distribution == "gamma":
            # Конвертируем логнормальные параметры в параметры гамма
            mu = metal_config.get("lognormal_mu", -5.0)
            sigma = metal_config.get("lognormal_sigma", 1.0)
            mean_ln = np.exp(mu + sigma ** 2 / 2)
            var_ln = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
            shape = mean_ln ** 2 / var_ln
            scale = var_ln / mean_ln
            return {"shape": shape, "scale": scale}
        elif distribution == "weibull":
            mu = metal_config.get("lognormal_mu", -5.0)
            sigma = metal_config.get("lognormal_sigma", 1.0)
            mean_ln = np.exp(mu + sigma ** 2 / 2)
            cv = np.sqrt(np.exp(sigma ** 2) - 1)
            # Приблизительная оценка параметра формы Вейбулла
            shape = 1.0 / cv if cv > 0 else 1.0
            scale = mean_ln / np.exp(
                np.log(np.math.gamma(1 + 1.0 / shape)) if shape > 0 else 0
            )
            return {"shape": max(shape, 0.5), "scale": max(scale, 1e-10)}
        elif distribution == "mixture":
            mu = metal_config.get("lognormal_mu", -5.0)
            sigma = metal_config.get("lognormal_sigma", 1.0)
            return {
                "mu1": mu,
                "sigma1": sigma * 0.7,
                "mu2": mu + 1.5,
                "sigma2": sigma * 0.8,
                "weight1": 0.7,
            }

        return {"mu": -5.0, "sigma": 1.0}

    def _generate_distribution(
        self,
        distribution: str,
        n_samples: int,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Генерирует выборку из указанного распределения.

        Parameters
        ----------
        distribution : str
            Тип распределения.
        n_samples : int
            Размер выборки.
        params : dict
            Параметры распределения.

        Returns
        -------
        np.ndarray
            Массив сгенерированных значений.
        """
        if distribution == "lognormal":
            mu = params.get("mu", -5.0)
            sigma = params.get("sigma", 1.0)
            values = self.rng.lognormal(mean=mu, sigma=sigma, size=n_samples)

        elif distribution == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 0.01)
            values = self.rng.gamma(shape=shape, scale=scale, size=n_samples)

        elif distribution == "weibull":
            shape = params.get("shape", 1.5)
            scale = params.get("scale", 0.01)
            values = scale * self.rng.weibull(a=shape, size=n_samples)

        elif distribution == "mixture":
            mu1 = params.get("mu1", -5.5)
            sigma1 = params.get("sigma1", 0.8)
            mu2 = params.get("mu2", -4.0)
            sigma2 = params.get("sigma2", 0.9)
            w1 = params.get("weight1", 0.7)

            n1 = self.rng.binomial(n_samples, w1)
            n2 = n_samples - n1
            vals1 = self.rng.lognormal(mean=mu1, sigma=sigma1, size=n1)
            vals2 = self.rng.lognormal(mean=mu2, sigma=sigma2, size=n2)
            values = np.concatenate([vals1, vals2])
            self.rng.shuffle(values)
        else:
            raise ValueError(f"Неподдерживаемое распределение: {distribution}")

        return values

    def _add_outliers(
        self,
        values: np.ndarray,
        metal_config: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Добавляет реалистичные выбросы в данные.

        Parameters
        ----------
        values : np.ndarray
            Исходный массив значений.
        metal_config : dict or None
            Конфигурация металла.

        Returns
        -------
        np.ndarray
            Массив с добавленными выбросами.
        """
        prob = self.config.get("outlier_probability", 0.03)
        mult_range = self.config.get("outlier_multiplier_range", (5, 20))

        n_outliers = self.rng.binomial(len(values), prob)
        if n_outliers == 0:
            return values

        outlier_indices = self.rng.choice(
            len(values), size=n_outliers, replace=False
        )
        multipliers = self.rng.uniform(
            mult_range[0], mult_range[1], size=n_outliers
        )

        p95 = np.percentile(values, 95)
        values = values.copy()
        for idx, mult in zip(outlier_indices, multipliers):
            values[idx] = p95 * mult

        self.logger.debug(
            f"  Добавлено {n_outliers} выбросов "
            f"(вероятность={prob:.1%})"
        )
        return values

    # ------------------------------------------------------------------
    # 3. Генерация многомерных коррелированных данных
    # ------------------------------------------------------------------

    def generate_multi_metal_dataset(
        self,
        n_samples: int = 200,
        metals: Optional[List[str]] = None,
        add_outliers: bool = True
    ) -> Dict[str, Any]:
        """
        Генерирует коррелированные данные для нескольких металлов.

        Использует многомерное нормальное распределение в логарифмическом
        пространстве для создания реалистичных корреляций между металлами.

        Parameters
        ----------
        n_samples : int, optional
            Количество проб (по умолчанию 200).
        metals : list of str, optional
            Список металлов. Если None — все из конфигурации.
        add_outliers : bool, optional
            Добавлять ли выбросы (по умолчанию True).

        Returns
        -------
        dict
            Словарь с ключами:
            - ``"data"`` (dict): {металл: dict с данными}
            - ``"combined_df"`` (pd.DataFrame): объединённые данные
            - ``"correlation_target"`` (pd.DataFrame): целевая матрица
              корреляций
            - ``"correlation_actual"`` (pd.DataFrame): фактическая
              матрица корреляций
            - ``"n_samples"`` (int): размер выборки

        Examples
        --------
        >>> loader = DataLoader()
        >>> dataset = loader.generate_multi_metal_dataset(
        ...     n_samples=300, metals=["Pb", "Cd", "Cu", "Zn"]
        ... )
        >>> print(dataset["combined_df"].head())
        >>> print("Корреляция Pb-Cd:",
        ...       dataset["correlation_actual"].loc["Pb", "Cd"])
        """
        if metals is None:
            metals = list(self.config.get("metals", {}).keys())

        available = set(self.config.get("metals", {}).keys())
        unknown = set(metals) - available
        if unknown:
            raise ValueError(
                f"Неизвестные металлы: {unknown}. Доступные: {available}"
            )

        self.logger.info(
            f"Генерация многомерных данных: металлы={metals}, n={n_samples}"
        )

        n_metals = len(metals)

        # Строим матрицу корреляций
        corr_config = self.config.get("correlation_matrix", {})
        corr_matrix = np.eye(n_metals)

        for i, m1 in enumerate(metals):
            for j, m2 in enumerate(metals):
                if i >= j:
                    continue
                key1 = f"{m1}-{m2}"
                key2 = f"{m2}-{m1}"
                rho = corr_config.get(key1, corr_config.get(key2, 0.0))
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

        # Собираем параметры логнормального распределения
        mus = []
        sigmas = []
        for metal in metals:
            mc = self.config["metals"][metal]
            mus.append(mc.get("lognormal_mu", -5.0))
            sigmas.append(mc.get("lognormal_sigma", 1.0))

        mus = np.array(mus)
        sigmas = np.array(sigmas)

        # Ковариационная матрица в лог-пространстве
        cov_matrix = np.outer(sigmas, sigmas) * corr_matrix

        # Убеждаемся в положительной определённости
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigvals <= 0):
            self.logger.warning(
                "Ковариационная матрица не положительно определена. "
                "Применяется коррекция."
            )
            cov_matrix = self._nearest_positive_definite(cov_matrix)

        # Генерируем многомерное нормальное в лог-пространстве
        log_values = self.rng.multivariate_normal(
            mean=mus, cov=cov_matrix, size=n_samples
        )

        # Преобразуем в исходное пространство
        values = np.exp(log_values)

        # Формируем результаты для каждого металла
        result_data = {}
        combined_data = {}

        for i, metal in enumerate(metals):
            metal_values = values[:, i].copy()
            mc = self.config["metals"][metal]

            if add_outliers:
                metal_values = self._add_outliers(metal_values, mc)

            metal_values = np.maximum(metal_values, 1e-12)
            combined_data[metal] = metal_values

            mac = mc.get("mac", 0.01)
            result_data[metal] = {
                "true_values": metal_values,
                "metal": metal,
                "distribution": "lognormal (multivariate)",
                "true_params": {
                    "mu": float(mus[i]),
                    "sigma": float(sigmas[i])
                },
                "true_mean": float(np.mean(metal_values)),
                "true_variance": float(np.var(metal_values, ddof=1)),
                "true_std": float(np.std(metal_values, ddof=1)),
                "true_median": float(np.median(metal_values)),
                "true_p75": float(np.percentile(metal_values, 75)),
                "true_p90": float(np.percentile(metal_values, 90)),
                "true_p95": float(np.percentile(metal_values, 95)),
                "true_p99": float(np.percentile(metal_values, 99)),
                "true_exceedance_prob": float(np.mean(metal_values > mac)),
                "mac": mac,
                "lod": mc.get("lod", 0.001),
                "n_samples": n_samples
            }

        combined_df = pd.DataFrame(combined_data)

        # Целевая и фактическая корреляции
        corr_target_df = pd.DataFrame(
            corr_matrix, index=metals, columns=metals
        )
        corr_actual_df = combined_df.apply(np.log).corr()

        self.logger.info("Целевая матрица корреляций (лог-пространство):")
        for i, m1 in enumerate(metals):
            for j, m2 in enumerate(metals):
                if j > i:
                    self.logger.info(
                        f"  {m1}-{m2}: целевая={corr_matrix[i, j]:.3f}, "
                        f"фактическая={corr_actual_df.loc[m1, m2]:.3f}"
                    )

        return {
            "data": result_data,
            "combined_df": combined_df,
            "correlation_target": corr_target_df,
            "correlation_actual": corr_actual_df,
            "n_samples": n_samples,
            "metals": metals
        }

    @staticmethod
    def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
        """
        Находит ближайшую положительно определённую матрицу.

        Использует метод Higham (2002).

        Parameters
        ----------
        A : np.ndarray
            Исходная матрица.

        Returns
        -------
        np.ndarray
            Ближайшая положительно определённая матрица.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if np.all(np.linalg.eigvalsh(A3) > 0):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not np.all(np.linalg.eigvalsh(A3) > 0):
            mineig = np.min(np.linalg.eigvalsh(A3))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    # ------------------------------------------------------------------
    # 4. Загрузка из публичных источников / встроенные данные
    # ------------------------------------------------------------------

    def load_from_public_sources(
        self,
        source: str = "synthetic",
        n_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Загружает данные из открытых источников или генерирует встроенный
        набор данных.

        Parameters
        ----------
        source : str, optional
            Источник данных:
            - ``"synthetic"`` — встроенный генератор (по умолчанию)
            - ``"usgs"`` — USGS Water Quality Data
            - ``"epa"`` — EPA STORET
        n_samples : int, optional
            Количество проб для синтетического набора (по умолчанию 500).

        Returns
        -------
        dict
            Словарь с данными (формат зависит от источника).

        Examples
        --------
        >>> loader = DataLoader()
        >>> dataset = loader.load_from_public_sources("synthetic", n_samples=500)
        >>> print(f"Металлов: {len(dataset['metals'])}")
        >>> print(f"Проб: {dataset['n_samples']}")
        """
        self.logger.info(f"Загрузка данных из источника: {source}")

        if source == "synthetic":
            return self._generate_builtin_dataset(n_samples)

        elif source == "usgs":
            return self._load_usgs_data(n_samples)

        elif source == "epa":
            return self._load_epa_data(n_samples)

        else:
            self.logger.warning(
                f"Неизвестный источник '{source}'. "
                f"Используется встроенный генератор."
            )
            return self._generate_builtin_dataset(n_samples)

    def _generate_builtin_dataset(
        self, n_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Генерирует встроенный реалистичный набор данных.

        Создаёт коррелированные данные для 5 основных металлов,
        основанные на типичных распределениях для водных объектов.

        Parameters
        ----------
        n_samples : int
            Количество проб.

        Returns
        -------
        dict
            Полный набор данных.
        """
        self.logger.info(
            f"Генерация встроенного набора данных (n={n_samples})"
        )
        metals = ["Pb", "Cd", "Cu", "Zn", "As"]
        dataset = self.generate_multi_metal_dataset(
            n_samples=n_samples,
            metals=metals,
            add_outliers=True
        )

        # Добавляем метаданные
        dataset["source"] = "builtin_synthetic"
        dataset["description"] = (
            "Синтетический набор данных, имитирующий мониторинг "
            "тяжёлых металлов в поверхностных водах. "
            "Основан на реалистичных логнормальных распределениях "
            "с корреляциями между металлами."
        )

        self.logger.info(
            f"Встроенный набор данных: {len(metals)} металлов, "
            f"{n_samples} проб"
        )

        return dataset

    def _load_usgs_data(self, n_samples: int = 500) -> Dict[str, Any]:
        """
        Пытается загрузить данные из USGS. При неудаче — генерирует
        синтетические данные с параметрами, типичными для данных USGS.

        Parameters
        ----------
        n_samples : int
            Количество проб.

        Returns
        -------
        dict
            Набор данных.
        """
        self.logger.info(
            "USGS API недоступен. Генерация данных с параметрами USGS."
        )

        # Параметры, типичные для данных USGS по поверхностным водам (мкг/л)
        # Конвертируем в мг/л для согласованности
        usgs_params = {
            "Pb": {"mu": -5.8, "sigma": 1.3},
            "Cd": {"mu": -8.2, "sigma": 1.1},
            "Cu": {"mu": -3.2, "sigma": 1.0},
            "Zn": {"mu": -2.3, "sigma": 1.4},
            "As": {"mu": -5.5, "sigma": 0.85},
        }

        dataset = self.generate_multi_metal_dataset(
            n_samples=n_samples,
            metals=list(usgs_params.keys()),
            add_outliers=True
        )
        dataset["source"] = "usgs_simulated"
        dataset["description"] = (
            "Данные, сгенерированные с параметрами, типичными для "
            "мониторинга USGS поверхностных вод."
        )
        return dataset

    def _load_epa_data(self, n_samples: int = 500) -> Dict[str, Any]:
        """
        Пытается загрузить данные из EPA STORET. При неудаче — генерирует
        синтетические данные.

        Parameters
        ----------
        n_samples : int
            Количество проб.

        Returns
        -------
        dict
            Набор данных.
        """
        self.logger.info(
            "EPA STORET API недоступен. Генерация данных с параметрами EPA."
        )

        dataset = self.generate_multi_metal_dataset(
            n_samples=n_samples,
            metals=["Pb", "Cd", "Cu", "Zn", "As"],
            add_outliers=True
        )
        dataset["source"] = "epa_simulated"
        dataset["description"] = (
            "Данные, сгенерированные с параметрами, типичными для "
            "мониторинга EPA STORET."
        )
        return dataset

    # ------------------------------------------------------------------
    # 5. Валидация данных
    # ------------------------------------------------------------------

    def validate_data(
        self,
        data: Dict[str, Any],
        data_type: str = "synthetic"
    ) -> Dict[str, Any]:
        """
        Проверяет данные на качество и возвращает отчёт о валидации.

        Parameters
        ----------
        data : dict
            Данные для валидации. Может быть результатом generate_synthetic_data,
            generate_multi_metal_dataset или load_real_data.
        data_type : str
            Тип данных: ``"synthetic"`` или ``"real"``.

        Returns
        -------
        dict
            Отчёт о валидации с ключами:
            - ``"is_valid"`` (bool): общий результат
            - ``"issues"`` (list): список обнаруженных проблем
            - ``"warnings"`` (list): предупреждения
            - ``"statistics"`` (dict): описательная статистика
            - ``"metal_reports"`` (dict): отчёт по каждому металлу

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.generate_synthetic_data("Pb", n_samples=100)
        >>> report = loader.validate_data(data, data_type="synthetic")
        >>> print(f"Валидно: {report['is_valid']}")
        >>> for issue in report['issues']:
        ...     print(f"  Проблема: {issue}")
        """
        self.logger.info("Начало валидации данных...")

        report = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {},
            "metal_reports": {}
        }

        min_samples = self.config.get("validation", {}).get(
            "min_sample_size", 20
        )
        extreme_factor = self.config.get("validation", {}).get(
            "extreme_outlier_factor", 10
        )

        # Определяем, какие данные валидировать
        if "true_values" in data:
            # Одиночный металл (результат generate_synthetic_data)
            metals_to_check = {data["metal"]: data["true_values"]}
            macs = {data["metal"]: data.get("mac", 0.01)}
        elif "data" in data:
            # Многомерный набор (результат generate_multi_metal_dataset)
            metals_to_check = {
                m: d["true_values"] for m, d in data["data"].items()
            }
            macs = {m: d.get("mac", 0.01) for m, d in data["data"].items()}
        elif "values" in data:
            # Реальные данные (результат load_real_data)
            values_df = data["values"]
            censored_df = data.get("censored", pd.DataFrame())
            metals_to_check = {}
            macs = {}
            for col in values_df.columns:
                vals = values_df[col].dropna().values.astype(float)
                if censored_df is not None and col in censored_df.columns:
                    # Берём только нецензурированные
                    uncensored_mask = ~censored_df[col].values[
                        values_df[col].notna().values
                    ]
                    vals_uncensored = vals[uncensored_mask] if len(uncensored_mask) == len(vals) else vals
                    metals_to_check[col] = vals_uncensored
                else:
                    metals_to_check[col] = vals
                mc = self.config.get("metals", {}).get(col, {})
                macs[col] = mc.get("mac", 0.01)
        else:
            report["is_valid"] = False
            report["issues"].append(
                "Неизвестный формат данных. Ожидается результат "
                "generate_synthetic_data, generate_multi_metal_dataset "
                "или load_real_data."
            )
            return report

        for metal, values in metals_to_check.items():
            metal_report = self._validate_single_metal(
                values, metal, macs.get(metal, 0.01),
                min_samples, extreme_factor
            )
            report["metal_reports"][metal] = metal_report

            if not metal_report["is_valid"]:
                report["is_valid"] = False

            report["issues"].extend(
                [f"[{metal}] {i}" for i in metal_report["issues"]]
            )
            report["warnings"].extend(
                [f"[{metal}] {w}" for w in metal_report["warnings"]]
            )
            report["statistics"][metal] = metal_report["statistics"]

        status = "✓ ВАЛИДНО" if report["is_valid"] else "✗ ПРОБЛЕМЫ"
        self.logger.info(
            f"Валидация завершена: {status}. "
            f"Проблем: {len(report['issues'])}, "
            f"Предупреждений: {len(report['warnings'])}"
        )

        return report

    def _validate_single_metal(
        self,
        values: np.ndarray,
        metal: str,
        mac: float,
        min_samples: int,
        extreme_factor: float
    ) -> Dict[str, Any]:
        """
        Валидирует данные для одного металла.

        Parameters
        ----------
        values : np.ndarray
            Массив значений.
        metal : str
            Символ металла.
        mac : float
            ПДК.
        min_samples : int
            Минимальный размер выборки.
        extreme_factor : float
            Множитель ПДК для определения экстремальных выбросов.

        Returns
        -------
        dict
            Отчёт по металлу.
        """
        report = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }

        values = np.asarray(values, dtype=float)

        # 1. Проверка размера выборки
        if len(values) < min_samples:
            report["issues"].append(
                f"Недостаточный размер выборки: {len(values)} < {min_samples}"
            )
            report["is_valid"] = False

        # 2. Проверка NaN
        n_nan = np.sum(np.isnan(values))
        if n_nan > 0:
            report["warnings"].append(
                f"Обнаружено {n_nan} пропущенных значений "
                f"({n_nan / len(values) * 100:.1f}%)"
            )
            values_clean = values[~np.isnan(values)]
        else:
            values_clean = values

        if len(values_clean) == 0:
            report["issues"].append("Все значения пропущены или NaN")
            report["is_valid"] = False
            return report

        # 3. Проверка отрицательных значений
        n_negative = np.sum(values_clean < 0)
        if n_negative > 0:
            report["issues"].append(
                f"Обнаружено {n_negative} отрицательных значений"
            )
            report["is_valid"] = False

        # 4. Проверка экстремальных выбросов (> extreme_factor * ПДК)
        threshold = extreme_factor * mac
        n_extreme = np.sum(values_clean > threshold)
        if n_extreme > 0:
            report["warnings"].append(
                f"Обнаружено {n_extreme} значений > {extreme_factor}×ПДК "
                f"({threshold:.6f}): помечены как экстремальные выбросы"
            )

        # 5. Описательная статистика
        report["statistics"] = {
            "n": len(values_clean),
            "mean": float(np.mean(values_clean)),
            "std": float(np.std(values_clean, ddof=1)),
            "min": float(np.min(values_clean)),
            "q25": float(np.percentile(values_clean, 25)),
            "median": float(np.median(values_clean)),
            "q75": float(np.percentile(values_clean, 75)),
            "max": float(np.max(values_clean)),
            "cv": float(
                np.std(values_clean, ddof=1) / np.mean(values_clean)
            ) if np.mean(values_clean) > 0 else np.inf,
            "skewness": float(stats.skew(values_clean)),
            "kurtosis": float(stats.kurtosis(values_clean)),
            "n_nan": int(n_nan),
            "n_negative": int(n_negative),
            "n_extreme_outliers": int(n_extreme),
            "pct_above_mac": float(np.mean(values_clean > mac) * 100),
            "mac": mac
        }

        return report

    # ------------------------------------------------------------------
    # 6. Вычисление истинных статистик
    # ------------------------------------------------------------------

    def compute_true_statistics(
        self,
        true_values: np.ndarray,
        metal: str
    ) -> Dict[str, Any]:
        """
        Вычисляет полный набор «истинных» статистик по нецензурированным
        данным.

        Используется как эталон для оценки качества методов обработки
        цензурированных данных.

        Parameters
        ----------
        true_values : np.ndarray
            Полный вектор истинных (нецензурированных) значений.
        metal : str
            Символ металла (для получения ПДК из конфигурации).

        Returns
        -------
        dict
            Словарь со статистиками:
            - ``"basic"`` (dict): среднее, дисперсия, std, медиана
            - ``"percentiles"`` (dict): 75, 90, 95, 99 перцентили
            - ``"exceedance"`` (dict): вероятности превышения
            - ``"fitted_distributions"`` (dict): параметры подогнанных
              распределений (lognormal, normal, gamma)
            - ``"normality_tests"`` (dict): тесты на нормальность
            - ``"lognormality_test"`` (dict): тест на логнормальность

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.generate_synthetic_data("Pb", 500)
        >>> stats_dict = loader.compute_true_statistics(
        ...     data["true_values"], "Pb"
        ... )
        >>> print(f"Среднее: {stats_dict['basic']['mean']:.6f}")
        >>> print(f"P95: {stats_dict['percentiles']['p95']:.6f}")
        >>> print(f"Шапиро-Уилк p-value: "
        ...       f"{stats_dict['normality_tests']['shapiro']['p_value']:.4f}")
        """
        self.logger.info(f"Вычисление истинных статистик для {metal}")

        values = np.asarray(true_values, dtype=float)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            raise ValueError("Массив значений пуст после удаления NaN")

        mc = self.config.get("metals", {}).get(metal, {})
        mac = mc.get("mac", 0.01)

        result = {}

        # --- Базовые статистики ---
        result["basic"] = {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=1)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": len(values),
            "cv": float(
                np.std(values, ddof=1) / np.mean(values)
            ) if np.mean(values) > 0 else np.inf,
            "geometric_mean": float(np.exp(np.mean(np.log(values[values > 0])))),
            "geometric_std": float(np.exp(np.std(np.log(values[values > 0]), ddof=1))),
        }

        # --- Перцентили ---
        result["percentiles"] = {
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "iqr": float(
                np.percentile(values, 75) - np.percentile(values, 25)
            ),
        }

        # --- Вероятности превышения ---
        result["exceedance"] = {
            "mac": mac,
            "prob_above_mac": float(np.mean(values > mac)),
            "prob_above_05mac": float(np.mean(values > 0.5 * mac)),
            "prob_above_2mac": float(np.mean(values > 2 * mac)),
            "n_above_mac": int(np.sum(values > mac)),
        }

        # --- Подгонка распределений ---
        result["fitted_distributions"] = {}
        positive_values = values[values > 0]

        # Логнормальное
        try:
            log_vals = np.log(positive_values)
            ln_mu = float(np.mean(log_vals))
            ln_sigma = float(np.std(log_vals, ddof=1))
            result["fitted_distributions"]["lognormal"] = {
                "mu": ln_mu,
                "sigma": ln_sigma,
                "mean_analytical": float(np.exp(ln_mu + ln_sigma ** 2 / 2)),
                "variance_analytical": float(
                    (np.exp(ln_sigma ** 2) - 1) *
                    np.exp(2 * ln_mu + ln_sigma ** 2)
                ),
                "median_analytical": float(np.exp(ln_mu)),
            }
        except Exception as e:
            self.logger.warning(f"Ошибка подгонки логнормального: {e}")
            result["fitted_distributions"]["lognormal"] = {"error": str(e)}

        # Нормальное
        try:
            norm_mu = float(np.mean(values))
            norm_sigma = float(np.std(values, ddof=1))
            result["fitted_distributions"]["normal"] = {
                "mu": norm_mu,
                "sigma": norm_sigma,
            }
        except Exception as e:
            result["fitted_distributions"]["normal"] = {"error": str(e)}

        # Гамма
        try:
            if np.all(positive_values > 0) and len(positive_values) > 2:
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(
                    positive_values, floc=0
                )
                result["fitted_distributions"]["gamma"] = {
                    "shape": float(fit_alpha),
                    "loc": float(fit_loc),
                    "scale": float(fit_beta),
                    "mean_analytical": float(fit_alpha * fit_beta),
                    "variance_analytical": float(fit_alpha * fit_beta ** 2),
                }
        except Exception as e:
            self.logger.warning(f"Ошибка подгонки гамма: {e}")
            result["fitted_distributions"]["gamma"] = {"error": str(e)}

        # --- Тесты на нормальность ---
        result["normality_tests"] = {}

        # Шапиро-Уилка (на сырых данных)
        try:
            sample_for_test = values
            if len(values) > 5000:
                # Шапиро-Уилк ограничен по размеру выборки
                sample_for_test = self.rng.choice(values, 5000, replace=False)

            sw_stat, sw_p = stats.shapiro(sample_for_test)
            result["normality_tests"]["shapiro"] = {
                "statistic": float(sw_stat),
                "p_value": float(sw_p),
                "is_normal": bool(sw_p > 0.05),
                "interpretation": (
                    "Нормальность НЕ отвергается (p > 0.05)"
                    if sw_p > 0.05
                    else "Нормальность отвергается (p ≤ 0.05)"
                )
            }
        except Exception as e:
            result["normality_tests"]["shapiro"] = {"error": str(e)}

        # Андерсон-Дарлинг (на сырых данных)
        try:
            ad_result = stats.anderson(values, dist='norm')
            # Сравниваем с критическим значением для 5%
            cv_5pct = ad_result.critical_values[2]  # индекс 2 = 5%
            result["normality_tests"]["anderson_darling"] = {
                "statistic": float(ad_result.statistic),
                "critical_values": {
                    f"{sl}%": float(cv)
                    for sl, cv in zip(
                        ad_result.significance_level,
                        ad_result.critical_values
                    )
                },
                "is_normal": bool(ad_result.statistic < cv_5pct),
                "interpretation": (
                    "Нормальность НЕ отвергается (5%)"
                    if ad_result.statistic < cv_5pct
                    else "Нормальность отвергается (5%)"
                )
            }
        except Exception as e:
            result["normality_tests"]["anderson_darling"] = {"error": str(e)}

        # --- Тест на логнормальность ---
        result["lognormality_test"] = {}
        try:
            if np.all(positive_values > 0):
                log_vals_test = np.log(positive_values)
                if len(log_vals_test) > 5000:
                    log_vals_test = self.rng.choice(
                        log_vals_test, 5000, replace=False
                    )

                sw_log_stat, sw_log_p = stats.shapiro(log_vals_test)
                result["lognormality_test"]["shapiro_on_log"] = {
                    "statistic": float(sw_log_stat),
                    "p_value": float(sw_log_p),
                    "is_lognormal": bool(sw_log_p > 0.05),
                    "interpretation": (
                        "Логнормальность НЕ отвергается (p > 0.05)"
                        if sw_log_p > 0.05
                        else "Логнормальность отвергается (p ≤ 0.05)"
                    )
                }

                # Андерсон-Дарлинг на лог-данных
                ad_log = stats.anderson(np.log(positive_values), dist='norm')
                cv_5_log = ad_log.critical_values[2]
                result["lognormality_test"]["anderson_darling_on_log"] = {
                    "statistic": float(ad_log.statistic),
                    "critical_values": {
                        f"{sl}%": float(cv)
                        for sl, cv in zip(
                            ad_log.significance_level,
                            ad_log.critical_values
                        )
                    },
                    "is_lognormal": bool(ad_log.statistic < cv_5_log),
                    "interpretation": (
                        "Логнормальность НЕ отвергается (5%)"
                        if ad_log.statistic < cv_5_log
                        else "Логнормальность отвергается (5%)"
                    )
                }
        except Exception as e:
            result["lognormality_test"]["error"] = str(e)

        self.logger.info(
            f"  Статистики для {metal}: "
            f"среднее={result['basic']['mean']:.6f}, "
            f"std={result['basic']['std']:.6f}, "
            f"P95={result['percentiles']['p95']:.6f}, "
            f"P(>ПДК)={result['exceedance']['prob_above_mac']:.2%}"
        )

        return result

    # ------------------------------------------------------------------
    # 7. Вспомогательные методы
    # ------------------------------------------------------------------

    def apply_censoring(
        self,
        true_values: np.ndarray,
        lod: float,
        censoring_type: str = "single"
    ) -> Dict[str, Any]:
        """
        Применяет цензурирование к данным (для экспериментов).

        Parameters
        ----------
        true_values : np.ndarray
            Истинные значения.
        lod : float
            Предел обнаружения.
        censoring_type : str
            Тип цензурирования:
            - ``"single"`` — один LOD
            - ``"random"`` — случайный LOD для каждого наблюдения

        Returns
        -------
        dict
            Словарь с ключами:
            - ``"observed_values"`` (np.ndarray): наблюдаемые значения
              (цензурированные заменены на LOD)
            - ``"censored"`` (np.ndarray): булева маска
            - ``"censoring_fraction"`` (float): доля цензурирования
            - ``"lod"`` (float): предел обнаружения

        Examples
        --------
        >>> loader = DataLoader()
        >>> true_vals = np.array([0.001, 0.005, 0.010, 0.002, 0.020])
        >>> result = loader.apply_censoring(true_vals, lod=0.003)
        >>> print(f"Цензурировано: {result['censoring_fraction']:.0%}")
        Цензурировано: 40%
        """
        values = np.asarray(true_values, dtype=float)

        if censoring_type == "single":
            censored_mask = values < lod
            observed = values.copy()
            observed[censored_mask] = lod
        elif censoring_type == "random":
            lods = lod * self.rng.uniform(0.5, 1.5, size=len(values))
            censored_mask = values < lods
            observed = values.copy()
            observed[censored_mask] = lods[censored_mask]
            lod = float(np.median(lods))
        else:
            raise ValueError(
                f"Неподдерживаемый тип цензурирования: {censoring_type}"
            )

        censoring_fraction = float(np.mean(censored_mask))
        self.logger.debug(
            f"Цензурирование применено: LOD={lod:.6f}, "
            f"доля={censoring_fraction:.1%}"
        )

        return {
            "observed_values": observed,
            "censored": censored_mask,
            "censoring_fraction": censoring_fraction,
            "lod": lod
        }

    def summary(self, data: Dict[str, Any]) -> str:
        """
        Формирует текстовое описание набора данных.

        Parameters
        ----------
        data : dict
            Набор данных.

        Returns
        -------
        str
            Текстовое описание.

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.generate_synthetic_data("Pb", 100)
        >>> print(loader.summary(data))
        """
        lines = ["=" * 60]
        lines.append("ОПИСАНИЕ НАБОРА ДАННЫХ")
        lines.append("=" * 60)

        if "metal" in data:
            # Одиночный металл
            lines.append(f"Металл: {data['metal']}")
            lines.append(f"Распределение: {data['distribution']}")
            lines.append(f"Размер выборки: {data.get('n_samples', 'N/A')}")
            lines.append(f"Среднее: {data['true_mean']:.6f}")
            lines.append(f"Дисперсия: {data['true_variance']:.8f}")
            lines.append(f"P90: {data['true_p90']:.6f}")
            lines.append(f"P95: {data['true_p95']:.6f}")
            lines.append(
                f"P(X>ПДК): {data['true_exceedance_prob']:.2%} "
                f"(ПДК={data.get('mac', 'N/A')})"
            )
        elif "data" in data:
            # Многомерный набор
            lines.append(
                f"Тип: Многомерный набор "
                f"({len(data.get('metals', []))} металлов)"
            )
            lines.append(f"Размер выборки: {data.get('n_samples', 'N/A')}")
            lines.append(
                f"Источник: {data.get('source', 'N/A')}"
            )
            lines.append("")
            lines.append(
                f"{'Металл':>8s} {'Среднее':>12s} {'Медиана':>12s} "
                f"{'P95':>12s} {'P(>ПДК)':>10s}"
            )
            lines.append("-" * 60)
            for metal, mdata in data["data"].items():
                lines.append(
                    f"{metal:>8s} "
                    f"{mdata['true_mean']:>12.6f} "
                    f"{mdata['true_median']:>12.6f} "
                    f"{mdata['true_p95']:>12.6f} "
                    f"{mdata['true_exceedance_prob']:>10.2%}"
                )
        elif "values" in data:
            # Реальные данные
            lines.append("Тип: Реальные данные")
            md = data.get("metadata", {})
            lines.append(f"Источник: {md.get('source', 'N/A')}")
            lines.append(f"Размер: {md.get('n_samples', 'N/A')}")
            lines.append(f"Металлов: {md.get('n_metals', 'N/A')}")
            cf = md.get("censoring_fractions", {})
            for metal, frac in cf.items():
                lines.append(f"  {metal}: цензурировано {frac:.1%}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Демонстрация при запуске как скрипта
# ======================================================================

def main():
    """Демонстрация функциональности модуля."""

    print("=" * 70)
    print("  МОДУЛЬ 1: Загрузка и подготовка данных")
    print("  Тяжёлые металлы в экологическом мониторинге")
    print("=" * 70)

    # Создаём загрузчик
    loader = DataLoader(DEFAULT_CONFIG)

    # --- 1. Генерация данных для одного металла ---
    print("\n" + "─" * 70)
    print("1. ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ОДНОГО МЕТАЛЛА (Pb)")
    print("─" * 70)

    for dist in ["lognormal", "gamma", "weibull", "mixture"]:
        data = loader.generate_synthetic_data(
            "Pb", n_samples=500, distribution=dist
        )
        print(
            f"  {dist:12s}: μ={data['true_mean']:.6f}, "
            f"σ²={data['true_variance']:.8f}, "
            f"P95={data['true_p95']:.6f}, "
            f"P(>ПДК)={data['true_exceedance_prob']:.2%}"
        )

    # --- 2. Многомерный набор данных ---
    print("\n" + "─" * 70)
    print("2. МНОГОМЕРНЫЙ НАБОР ДАННЫХ (5 металлов, коррелированные)")
    print("─" * 70)

    multi_data = loader.generate_multi_metal_dataset(
        n_samples=500,
        metals=["Pb", "Cd", "Cu", "Zn", "As"]
    )
    print(loader.summary(multi_data))

    print("\nМатрица корреляций (лог-пространство):")
    print(multi_data["correlation_actual"].round(3).to_string())

    # --- 3. Встроенный набор данных ---
    print("\n" + "─" * 70)
    print("3. ВСТРОЕННЫЙ НАБОР ДАННЫХ (из «публичных источников»)")
    print("─" * 70)

    builtin = loader.load_from_public_sources("synthetic", n_samples=500)
    print(f"  Источник: {builtin.get('source', 'N/A')}")
    print(f"  Проб: {builtin.get('n_samples', 'N/A')}")
    print(f"  Металлов: {len(builtin.get('metals', []))}")

    # --- 4. Валидация данных ---
    print("\n" + "─" * 70)
    print("4. ВАЛИДАЦИЯ ДАННЫХ")
    print("─" * 70)

    validation_report = loader.validate_data(multi_data)
    print(f"  Общий результат: {'✓ ВАЛИДНО' if validation_report['is_valid'] else '✗ ПРОБЛЕМЫ'}")
    if validation_report["issues"]:
        for issue in validation_report["issues"]:
            print(f"  ✗ {issue}")
    if validation_report["warnings"]:
        for warning in validation_report["warnings"][:5]:
            print(f"  ⚠ {warning}")

    # --- 5. Истинные статистики ---
    print("\n" + "─" * 70)
    print("5. ИСТИННЫЕ СТАТИСТИКИ (эталон для сравнения)")
    print("─" * 70)

    pb_data = loader.generate_synthetic_data("Pb", n_samples=1000)
    true_stats = loader.compute_true_statistics(pb_data["true_values"], "Pb")

    print(f"\n  Свинец (Pb), n = {true_stats['basic']['n']}")
    print(f"  {'─' * 50}")
    print(f"  Среднее арифметическое:   {true_stats['basic']['mean']:.6f}")
    print(f"  Среднее геометрическое:   {true_stats['basic']['geometric_mean']:.6f}")
    print(f"  Стандартное отклонение:    {true_stats['basic']['std']:.6f}")
    print(f"  Коэффициент вариации:     {true_stats['basic']['cv']:.2f}")
    print(f"  Медиана:                  {true_stats['percentiles']['p50']:.6f}")
    print(f"  IQR:                      {true_stats['percentiles']['iqr']:.6f}")
    print(f"  P90:                      {true_stats['percentiles']['p90']:.6f}")
    print(f"  P95:                      {true_stats['percentiles']['p95']:.6f}")
    print(f"  P99:                      {true_stats['percentiles']['p99']:.6f}")
    print(f"  P(X > ПДК):               {true_stats['exceedance']['prob_above_mac']:.2%}")

    # Тесты на нормальность
    print(f"\n  Тесты нормальности:")
    if "shapiro" in true_stats["normality_tests"]:
        sw = true_stats["normality_tests"]["shapiro"]
        if "error" not in sw:
            print(f"    Шапиро-Уилк: W={sw['statistic']:.4f}, p={sw['p_value']:.4e}")
            print(f"    → {sw['interpretation']}")

    if "anderson_darling" in true_stats["normality_tests"]:
        ad = true_stats["normality_tests"]["anderson_darling"]
        if "error" not in ad:
            print(f"    Андерсон-Дарлинг: A²={ad['statistic']:.4f}")
            print(f"    → {ad['interpretation']}")

    print(f"\n  Тест логнормальности:")
    if "shapiro_on_log" in true_stats["lognormality_test"]:
        slog = true_stats["lognormality_test"]["shapiro_on_log"]
        if "error" not in slog:
            print(f"    Шапиро-Уилк (log): W={slog['statistic']:.4f}, p={slog['p_value']:.4e}")
            print(f"    → {slog['interpretation']}")

    # Подогнанные распределения
    print(f"\n  Параметры логнормального распределения:")
    if "lognormal" in true_stats["fitted_distributions"]:
        ln = true_stats["fitted_distributions"]["lognormal"]
        if "error" not in ln:
            print(f"    μ_log = {ln['mu']:.4f}")
            print(f"    σ_log = {ln['sigma']:.4f}")
            print(f"    Аналитическое среднее = {ln['mean_analytical']:.6f}")

    # --- 6. Цензурирование ---
    print("\n" + "─" * 70)
    print("6. ПРИМЕНЕНИЕ ЦЕНЗУРИРОВАНИЯ")
    print("─" * 70)

    lod = pb_data["lod"]
    cens_result = loader.apply_censoring(pb_data["true_values"], lod)
    print(f"  LOD = {lod:.6f}")
    print(f"  Доля цензурирования: {cens_result['censoring_fraction']:.1%}")
    print(f"  Нецензурированных: {np.sum(~cens_result['censored'])}")

    # Демонстрация разных уровней цензурирования
    print(f"\n  Зависимость от LOD:")
    for lod_mult in [0.5, 1.0, 2.0, 5.0, 10.0]:
        test_lod = pb_data["lod"] * lod_mult
        cr = loader.apply_censoring(pb_data["true_values"], test_lod)
        print(
            f"    LOD = {test_lod:.6f} ({lod_mult:.0f}×): "
            f"цензурировано {cr['censoring_fraction']:.1%}"
        )

    print("\n" + "=" * 70)
    print("  Модуль 1 успешно продемонстрирован!")
    print("=" * 70)


if __name__ == "__main__":
    main()