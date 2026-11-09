"""
Модуль 11: Монте-Карло эксперимент по сравнению методов обработки цензурированных данных.

Объединяет все предыдущие модули (1-10) и проводит систематический
Монте-Карло эксперимент: многократно генерирует данные, цензурирует их,
применяет все методы и собирает статистику ошибок.

Это главный "оркестратор" проекта.
"""

import os
import sys
import time
import signal
import pickle
import logging
import hashlib
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Импорт модулей проекта
try:
    from module_1_generation import DataGenerator
except ImportError:
    DataGenerator = None

try:
    from module_2_censoring import CensoringMechanism
except ImportError:
    CensoringMechanism = None

try:
    from module_3_substitution import SubstitutionMethods
except ImportError:
    SubstitutionMethods = None

try:
    from module_4_mle import MLECensored
except ImportError:
    MLECensored = None

try:
    from module_5_km_ros import KMROSEstimator
except ImportError:
    KMROSEstimator = None

try:
    from module_6_bayesian import BayesianCensored
except ImportError:
    BayesianCensored = None

try:
    from module_7_imputation import MultipleImputation
except ImportError:
    MultipleImputation = None

try:
    from module_8_nonparametric import NonparametricCensored
except ImportError:
    NonparametricCensored = None

try:
    from module_9_evaluation import MethodEvaluator
except ImportError:
    MethodEvaluator = None

try:
    from module_10_visualization import ResultVisualizer
except ImportError:
    ResultVisualizer = None


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Вспомогательные структуры
# ─────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Конфигурация Монте-Карло эксперимента."""
    metal: str = "Pb"
    n_simulations: int = 1000
    sample_sizes: List[int] = field(default_factory=lambda: [50, 100, 200])
    censoring_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    censoring_types: List[str] = field(default_factory=lambda: ["fixed_lod"])
    distributions: List[str] = field(default_factory=lambda: ["lognormal"])
    methods: Optional[List[str]] = None
    n_jobs: int = 1
    seed: int = 42
    save_raw: bool = False
    checkpoint_every: int = 100
    max_memory_mb: float = 2048.0

    def to_dict(self) -> dict:
        return {
            "metal": self.metal,
            "n_simulations": self.n_simulations,
            "sample_sizes": self.sample_sizes,
            "censoring_rates": self.censoring_rates,
            "censoring_types": self.censoring_types,
            "distributions": self.distributions,
            "methods": self.methods,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "save_raw": self.save_raw,
            "checkpoint_every": self.checkpoint_every,
        }

    def experiment_hash(self) -> str:
        """Уникальный хэш конфигурации для checkpoint."""
        cfg_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(cfg_str.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────
# Список всех доступных методов
# ─────────────────────────────────────────────────────────
ALL_METHODS = [
    "substitution_zero",
    "substitution_lod_div_2",
    "substitution_lod_div_sqrt2",
    "substitution_lod",
    "mle_lognormal",
    "mle_gamma",
    "km_kaplan_meier",
    "ros_regression_on_order",
    "bayesian_lognormal",
    "imputation_multiple",
    "nonparametric_bootstrap",
]


def _get_default_config() -> dict:
    """Конфигурация по умолчанию для всех модулей."""
    return {
        "metal": "Pb",
        "seed": 42,
        "mpc": 0.01,
        # Параметры генерации
        "generation": {
            "lognormal": {"mu_log": -4.6, "sigma_log": 0.8},
            "gamma": {"shape": 2.0, "scale": 0.005},
            "weibull": {"shape": 1.5, "scale": 0.01},
        },
        # Параметры MLE
        "mle": {"max_iter": 500, "tol": 1e-6},
        # Параметры Байеса
        "bayesian": {
            "n_samples": 2000,
            "n_burnin": 500,
            "n_chains": 2,
        },
        # Параметры множественной импутации
        "imputation": {"n_imputations": 10, "n_burnin": 50},
        # Параметры бутстрэпа
        "bootstrap": {"n_bootstrap": 500},
        # Параметры оценки
        "evaluation": {"confidence_level": 0.95},
    }


class InterruptHandler:
    """Обработчик прерывания Ctrl+C с сохранением промежуточных результатов."""

    def __init__(self):
        self.interrupted = False
        self._original_handler = None

    def __enter__(self):
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._original_handler)
        return False

    def _handler(self, signum, frame):
        if self.interrupted:
            logger.warning("Повторное прерывание — завершение немедленно.")
            sys.exit(1)
        self.interrupted = True
        logger.warning(
            "Прерывание получено. Завершаем текущую итерацию и сохраняем результаты..."
        )


# ─────────────────────────────────────────────────────────
# ОСНОВНОЙ КЛАСС
# ─────────────────────────────────────────────────────────
class MonteCarloSimulation:
    """
    Полный Монте-Карло эксперимент по сравнению методов обработки
    цензурированных данных.

    Parameters
    ----------
    config : dict
        Словарь конфигурации. Если None — используются значения по умолчанию.

    Examples
    --------
    >>> mc = MonteCarloSimulation({"seed": 42, "mpc": 0.01})
    >>> results = mc.run_monte_carlo(
    ...     n_simulations=100,
    ...     sample_sizes=[50],
    ...     censoring_rates=[0.2, 0.5],
    ... )
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = _get_default_config()
        if config:
            self.config.update(config)

        self.seed = self.config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.mpc = self.config.get("mpc", 0.01)

        self._setup_logging()
        self._init_modules()

        # Для checkpoint / restart
        self._checkpoint_dir = Path(self.config.get("checkpoint_dir", ".checkpoints"))
        self._checkpoint_dir.mkdir(exist_ok=True)

        logger.info("MonteCarloSimulation инициализирован (seed=%d)", self.seed)

    # ─────────────────────── Инициализация ───────────────────────

    def _setup_logging(self) -> None:
        """Настройка логирования."""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _init_modules(self) -> None:
        """Инициализация всех модулей проекта."""
        self.generator = DataGenerator(self.config) if DataGenerator else None
        self.censorer = CensoringMechanism(self.config) if CensoringMechanism else None
        self.substitution = SubstitutionMethods(self.config) if SubstitutionMethods else None
        self.mle = MLECensored(self.config) if MLECensored else None
        self.km_ros = KMROSEstimator(self.config) if KMROSEstimator else None
        self.bayesian = BayesianCensored(self.config) if BayesianCensored else None
        self.imputation = MultipleImputation(self.config) if MultipleImputation else None
        self.nonparametric = NonparametricCensored(self.config) if NonparametricCensored else None
        self.evaluator = MethodEvaluator(self.config) if MethodEvaluator else None
        self.visualizer = ResultVisualizer(self.config) if ResultVisualizer else None

        missing = []
        if not self.generator:
            missing.append("module_1_generation")
        if not self.censorer:
            missing.append("module_2_censoring")
        if not self.evaluator:
            missing.append("module_9_evaluation")
        if missing:
            logger.warning("Недоступные модули: %s. Используем fallback.", missing)

    # ─────────────────────── Генерация данных (fallback) ───────────────────────

    def _generate_sample(
        self,
        distribution: str,
        sample_size: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Генерация выборки из заданного распределения.

        Если module_1 доступен — делегирует ему.
        Иначе — fallback на scipy.

        Parameters
        ----------
        distribution : str
            Название распределения: "lognormal", "gamma", "weibull".
        sample_size : int
            Размер выборки.
        seed : int, optional
            Seed для воспроизводимости.

        Returns
        -------
        np.ndarray
            Сгенерированная выборка.
        """
        rng = np.random.default_rng(seed)
        gen_cfg = self.config.get("generation", {})

        if self.generator is not None:
            try:
                return self.generator.generate(
                    distribution=distribution,
                    n=sample_size,
                    seed=seed,
                )
            except Exception as e:
                logger.debug("Fallback генерация: %s", e)

        if distribution == "lognormal":
            params = gen_cfg.get("lognormal", {"mu_log": -4.6, "sigma_log": 0.8})
            return rng.lognormal(
                mean=params["mu_log"],
                sigma=params["sigma_log"],
                size=sample_size,
            )
        elif distribution == "gamma":
            params = gen_cfg.get("gamma", {"shape": 2.0, "scale": 0.005})
            return rng.gamma(
                shape=params["shape"],
                scale=params["scale"],
                size=sample_size,
            )
        elif distribution == "weibull":
            params = gen_cfg.get("weibull", {"shape": 1.5, "scale": 0.01})
            return params["scale"] * rng.weibull(params["shape"], size=sample_size)
        else:
            raise ValueError(f"Неизвестное распределение: {distribution}")

    # ─────────────────────── Цензурирование (fallback) ───────────────────────

    def _censor_data(
        self,
        data: np.ndarray,
        censoring_rate: float,
        censoring_type: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Цензурирование данных.

        Parameters
        ----------
        data : np.ndarray
            Исходные данные.
        censoring_rate : float
            Доля цензурирования.
        censoring_type : str
            Тип цензурирования: "fixed_lod", "variable_lod", "random".

        Returns
        -------
        observed : np.ndarray
            Наблюдаемые значения (цензурированные заменены на LOD).
        censored : np.ndarray[bool]
            Маска цензурирования (True = цензурировано).
        lod_values : np.ndarray
            Значения LOD для каждого наблюдения.
        """
        if self.censorer is not None:
            try:
                result = self.censorer.censor(
                    data=data,
                    censoring_rate=censoring_rate,
                    censoring_type=censoring_type,
                    seed=seed,
                )
                if isinstance(result, dict):
                    return (
                        result.get("observed", data),
                        result.get("censored", np.zeros(len(data), dtype=bool)),
                        result.get("lod", np.zeros(len(data))),
                    )
                elif isinstance(result, tuple) and len(result) >= 3:
                    return result[0], result[1], result[2]
                elif isinstance(result, tuple) and len(result) == 2:
                    return result[0], result[1], np.full(len(data), np.nan)
            except Exception as e:
                logger.debug("Fallback цензурирование: %s", e)

        # Fallback реализация
        rng = np.random.default_rng(seed)
        sorted_data = np.sort(data)

        if censoring_type == "fixed_lod":
            idx = int(np.clip(censoring_rate * len(data), 0, len(data) - 1))
            lod = sorted_data[idx]
            censored = data < lod
            observed = data.copy()
            observed[censored] = lod
            lod_values = np.full(len(data), lod)

        elif censoring_type == "variable_lod":
            target_q = np.quantile(data, censoring_rate)
            noise = rng.uniform(0.8, 1.2, size=len(data))
            lod_values = target_q * noise
            censored = data < lod_values
            observed = data.copy()
            observed[censored] = lod_values[censored]

        elif censoring_type == "random":
            censored = rng.random(len(data)) < censoring_rate
            lod = np.quantile(data, censoring_rate)
            lod_values = np.full(len(data), lod)
            observed = data.copy()
            observed[censored] = lod

        else:
            raise ValueError(f"Неизвестный тип цензурирования: {censoring_type}")

        return observed, censored, lod_values

    # ─────────────────────── Вычисление истинных статистик ───────────────────────

    @staticmethod
    def _compute_true_statistics(data: np.ndarray, mpc: float) -> dict:
        """
        Вычисление истинных статистик по полным (нецензурированным) данным.

        Parameters
        ----------
        data : np.ndarray
            Полные данные.
        mpc : float
            Предельно допустимая концентрация.

        Returns
        -------
        dict
            Словарь истинных статистик.
        """
        return {
            "mean": float(np.mean(data)),
            "variance": float(np.var(data, ddof=1)) if len(data) > 1 else 0.0,
            "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            "median": float(np.median(data)),
            "p90": float(np.percentile(data, 90)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "exceedance_prob": float(np.mean(data > mpc)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "full_data": data.copy(),
        }

    # ─────────────────────── Применение методов ───────────────────────

    def _apply_method(
        self,
        method_name: str,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> dict:
        """
        Применение одного метода обработки цензурированных данных.

        Parameters
        ----------
        method_name : str
            Название метода.
        observed : np.ndarray
            Наблюдаемые значения.
        censored : np.ndarray[bool]
            Маска цензурирования.
        lod_values : np.ndarray
            Значения LOD.

        Returns
        -------
        dict
            Результаты метода: оценки статистик, время, статус сходимости.
        """
        t0 = time.perf_counter()
        result = {
            "converged": True,
            "estimated_mean": np.nan,
            "estimated_variance": np.nan,
            "estimated_std": np.nan,
            "estimated_median": np.nan,
            "estimated_p90": np.nan,
            "estimated_p95": np.nan,
            "estimated_p99": np.nan,
            "estimated_exceedance": np.nan,
            "imputed_data": None,
            "ci_mean": (np.nan, np.nan),
            "ci_p95": (np.nan, np.nan),
            "computation_time": np.nan,
        }

        try:
            imputed = self._get_imputed_data(method_name, observed, censored, lod_values)

            if imputed is None or len(imputed) == 0 or np.all(np.isnan(imputed)):
                result["converged"] = False
                result["computation_time"] = time.perf_counter() - t0
                return result

            # Очистка: заменяем отрицательные на 0 (концентрации не бывают отрицательными)
            imputed = np.array(imputed, dtype=float)
            valid = np.isfinite(imputed)
            if np.sum(valid) < 2:
                result["converged"] = False
                result["computation_time"] = time.perf_counter() - t0
                return result

            imputed_clean = imputed[valid]
            imputed_clean = np.clip(imputed_clean, 0, None)

            result["estimated_mean"] = float(np.mean(imputed_clean))
            result["estimated_variance"] = float(np.var(imputed_clean, ddof=1))
            result["estimated_std"] = float(np.std(imputed_clean, ddof=1))
            result["estimated_median"] = float(np.median(imputed_clean))
            result["estimated_p90"] = float(np.percentile(imputed_clean, 90))
            result["estimated_p95"] = float(np.percentile(imputed_clean, 95))
            result["estimated_p99"] = float(np.percentile(imputed_clean, 99))
            result["estimated_exceedance"] = float(np.mean(imputed_clean > self.mpc))
            result["imputed_data"] = imputed_clean

            # Bootstrap CI для mean и p95
            ci_mean, ci_p95 = self._bootstrap_ci(imputed_clean)
            result["ci_mean"] = ci_mean
            result["ci_p95"] = ci_p95

        except Exception as e:
            logger.debug("Метод %s не сошёлся: %s", method_name, e)
            result["converged"] = False

        result["computation_time"] = time.perf_counter() - t0
        return result

    def _get_imputed_data(
        self,
        method_name: str,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Получение импутированных данных для заданного метода.

        Returns
        -------
        np.ndarray or None
            Импутированные данные или None при неудаче.
        """
        n = len(observed)
        uncensored_mask = ~censored

        # --- Substitution methods ---
        if method_name.startswith("substitution_"):
            lod_scalar = np.nanmax(lod_values[censored]) if np.any(censored) else np.nanmax(lod_values)
            if np.isnan(lod_scalar):
                lod_scalar = np.nanmin(observed[uncensored_mask]) if np.any(uncensored_mask) else 0.001

            if self.substitution is not None:
                try:
                    sub_type = method_name.replace("substitution_", "")
                    res = self.substitution.substitute(
                        observed=observed,
                        censored=censored,
                        lod=lod_scalar,
                        method=sub_type,
                    )
                    if isinstance(res, np.ndarray):
                        return res
                    elif isinstance(res, dict):
                        return res.get("imputed", None)
                except Exception:
                    pass

            # Fallback substitution
            imputed = observed.copy().astype(float)
            if method_name == "substitution_zero":
                imputed[censored] = 0.0
            elif method_name == "substitution_lod_div_2":
                imputed[censored] = lod_scalar / 2.0
            elif method_name == "substitution_lod_div_sqrt2":
                imputed[censored] = lod_scalar / np.sqrt(2.0)
            elif method_name == "substitution_lod":
                imputed[censored] = lod_scalar
            return imputed

        # --- MLE ---
        elif method_name.startswith("mle_"):
            dist = method_name.replace("mle_", "")
            if self.mle is not None:
                try:
                    res = self.mle.fit(
                        observed=observed,
                        censored=censored,
                        distribution=dist,
                    )
                    if isinstance(res, dict) and "imputed" in res:
                        return res["imputed"]
                    elif isinstance(res, dict) and "params" in res:
                        # Генерируем по оцененным параметрам
                        return self._generate_from_params(
                            dist, res["params"], observed, censored, lod_values
                        )
                except Exception:
                    pass

            # Fallback MLE
            return self._fallback_mle(dist, observed, censored, lod_values)

        # --- Kaplan-Meier ---
        elif method_name == "km_kaplan_meier":
            if self.km_ros is not None:
                try:
                    res = self.km_ros.kaplan_meier(
                        observed=observed, censored=censored
                    )
                    if isinstance(res, np.ndarray):
                        return res
                    elif isinstance(res, dict):
                        return res.get("imputed", None)
                except Exception:
                    pass
            return self._fallback_km(observed, censored, lod_values)

        # --- ROS ---
        elif method_name == "ros_regression_on_order":
            if self.km_ros is not None:
                try:
                    res = self.km_ros.ros(observed=observed, censored=censored)
                    if isinstance(res, np.ndarray):
                        return res
                    elif isinstance(res, dict):
                        return res.get("imputed", None)
                except Exception:
                    pass
            return self._fallback_ros(observed, censored, lod_values)

        # --- Bayesian ---
        elif method_name == "bayesian_lognormal":
            if self.bayesian is not None:
                try:
                    res = self.bayesian.fit(
                        observed=observed,
                        censored=censored,
                        distribution="lognormal",
                    )
                    if isinstance(res, dict) and "imputed" in res:
                        return res["imputed"]
                    elif isinstance(res, np.ndarray):
                        return res
                except Exception:
                    pass
            return self._fallback_bayesian(observed, censored, lod_values)

        # --- Multiple imputation ---
        elif method_name == "imputation_multiple":
            if self.imputation is not None:
                try:
                    res = self.imputation.impute(
                        observed=observed, censored=censored
                    )
                    if isinstance(res, np.ndarray):
                        return res
                    elif isinstance(res, dict):
                        return res.get("imputed", res.get("pooled", None))
                except Exception:
                    pass
            return self._fallback_imputation(observed, censored, lod_values)

        # --- Nonparametric bootstrap ---
        elif method_name == "nonparametric_bootstrap":
            if self.nonparametric is not None:
                try:
                    res = self.nonparametric.estimate(
                        observed=observed, censored=censored
                    )
                    if isinstance(res, np.ndarray):
                        return res
                    elif isinstance(res, dict):
                        return res.get("imputed", None)
                except Exception:
                    pass
            return self._fallback_nonparametric(observed, censored, lod_values)

        else:
            raise ValueError(f"Неизвестный метод: {method_name}")

    # ─────────────────────── Fallback реализации методов ───────────────────────

    def _fallback_mle(
        self,
        distribution: str,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback MLE для лог-нормального и гамма-распределений."""
        uncensored = observed[~censored]
        if len(uncensored) < 2:
            return None

        imputed = observed.copy().astype(float)

        if distribution == "lognormal":
            log_uncensored = np.log(np.clip(uncensored, 1e-15, None))
            mu_hat = np.mean(log_uncensored)
            sigma_hat = max(np.std(log_uncensored, ddof=1), 1e-10)

            if np.any(censored):
                lods = lod_values[censored]
                for i, idx in enumerate(np.where(censored)[0]):
                    lod_i = lods[i] if np.isfinite(lods[i]) else np.min(uncensored)
                    log_lod = np.log(max(lod_i, 1e-15))
                    # E[X | X < lod] для lognormal
                    z = (log_lod - mu_hat) / sigma_hat
                    if stats.norm.cdf(z) > 1e-10:
                        truncated_mean = np.exp(
                            mu_hat + sigma_hat ** 2 / 2
                        ) * stats.norm.cdf(z - sigma_hat) / stats.norm.cdf(z)
                        imputed[idx] = truncated_mean
                    else:
                        imputed[idx] = lod_i / 2.0

        elif distribution == "gamma":
            try:
                shape, _, scale = stats.gamma.fit(uncensored, floc=0)
                if np.any(censored):
                    lods = lod_values[censored]
                    for i, idx in enumerate(np.where(censored)[0]):
                        lod_i = lods[i] if np.isfinite(lods[i]) else np.min(uncensored)
                        prob = stats.gamma.cdf(lod_i, shape, scale=scale)
                        if prob > 1e-10:
                            # Численный E[X | X < lod]
                            x_grid = np.linspace(1e-15, lod_i, 200)
                            pdf_vals = stats.gamma.pdf(x_grid, shape, scale=scale)
                            imputed[idx] = np.trapz(x_grid * pdf_vals, x_grid) / prob
                        else:
                            imputed[idx] = lod_i / 2.0
            except Exception:
                return None
        else:
            return None

        return imputed

    def _fallback_km(
        self,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback Kaplan-Meier: простое KM-перераспределение."""
        n = len(observed)
        if n == 0:
            return None

        # Для левого цензурирования: разворачиваем через зеркало
        max_val = np.max(observed) * 1.1
        reversed_data = max_val - observed
        reversed_event = ~censored  # в KM: событие = наблюдается

        # Сортируем
        order = np.argsort(reversed_data)
        sorted_times = reversed_data[order]
        sorted_events = reversed_event[order]
        sorted_censored_orig = censored[order]

        # KM оценка выживаемости
        km_survival = np.ones(n)
        at_risk = n
        for i in range(n):
            if sorted_events[i]:
                km_survival[i] = km_survival[i - 1] * (1 - 1 / at_risk) if i > 0 else (1 - 1 / at_risk)
            else:
                km_survival[i] = km_survival[i - 1] if i > 0 else 1.0
            at_risk -= 1

        # Простая импутация: цензурированные заменяются на условное среднее
        imputed = observed.copy().astype(float)
        uncensored_vals = observed[~censored]
        if len(uncensored_vals) > 0 and np.any(censored):
            lod_max = np.max(lod_values[censored]) if np.isfinite(lod_values[censored]).any() else np.min(uncensored_vals)
            below_lod = uncensored_vals[uncensored_vals <= lod_max]
            if len(below_lod) > 0:
                fill_val = np.mean(below_lod)
            else:
                fill_val = lod_max / 2.0
            imputed[censored] = fill_val
        elif np.any(censored):
            lod_scalar = np.nanmax(lod_values) if np.any(np.isfinite(lod_values)) else 0.001
            imputed[censored] = lod_scalar / 2.0

        return imputed

    def _fallback_ros(
        self,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback ROS: Regression on Order Statistics."""
        n = len(observed)
        uncensored = observed[~censored]
        if len(uncensored) < 2:
            return None

        imputed = observed.copy().astype(float)

        # Plotting positions для нецензурированных
        m = len(uncensored)
        sorted_unc = np.sort(uncensored)
        pp = (np.arange(1, m + 1) - 0.375) / (m + 0.25)

        # Регрессия log(x) vs Phi^{-1}(pp)
        log_sorted = np.log(np.clip(sorted_unc, 1e-15, None))
        z_scores = stats.norm.ppf(pp)

        slope, intercept, _, _, _ = stats.linregress(z_scores, log_sorted)

        if np.any(censored):
            n_cens = int(np.sum(censored))
            # Plotting positions для цензурированных
            cens_pp = (np.arange(1, n_cens + 1) - 0.375) / (n + 0.25)
            cens_z = stats.norm.ppf(np.clip(cens_pp, 0.001, 0.999))
            cens_log_vals = intercept + slope * cens_z
            cens_vals = np.exp(cens_log_vals)

            cens_indices = np.where(censored)[0]
            # Сортируем цензурированные индексы по lod
            cens_order = np.argsort(lod_values[cens_indices])
            for i, order_idx in enumerate(cens_order):
                if i < len(cens_vals):
                    imputed[cens_indices[order_idx]] = max(cens_vals[i], 0.0)

        return imputed

    def _fallback_bayesian(
        self,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback Bayesian: простой Gibbs sampler для lognormal."""
        uncensored = observed[~censored]
        if len(uncensored) < 2:
            return None

        log_unc = np.log(np.clip(uncensored, 1e-15, None))
        mu = np.mean(log_unc)
        sigma = max(np.std(log_unc, ddof=1), 0.01)

        rng = np.random.default_rng(self.seed + 7)
        imputed = observed.copy().astype(float)

        n_iter = self.config.get("bayesian", {}).get("n_samples", 500)

        cens_idx = np.where(censored)[0]
        if len(cens_idx) == 0:
            return imputed

        # Инициализация цензурированных
        for idx in cens_idx:
            lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else np.min(uncensored)
            imputed[idx] = lod_i / 2.0

        # Gibbs sampling
        for _ in range(n_iter):
            all_log = np.log(np.clip(imputed, 1e-15, None))
            mu = np.mean(all_log)
            sigma = max(np.std(all_log, ddof=1), 0.01)

            for idx in cens_idx:
                lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else np.min(uncensored)
                log_lod = np.log(max(lod_i, 1e-15))
                z = (log_lod - mu) / sigma
                u = rng.uniform(0, stats.norm.cdf(z))
                log_val = mu + sigma * stats.norm.ppf(max(u, 1e-10))
                imputed[idx] = np.exp(log_val)

        return imputed

    def _fallback_imputation(
        self,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback множественная импутация."""
        n_imp = self.config.get("imputation", {}).get("n_imputations", 10)
        uncensored = observed[~censored]
        if len(uncensored) < 2:
            return None

        rng = np.random.default_rng(self.seed + 13)
        log_unc = np.log(np.clip(uncensored, 1e-15, None))
        mu = np.mean(log_unc)
        sigma = max(np.std(log_unc, ddof=1), 0.01)

        cens_idx = np.where(censored)[0]
        all_imputed = np.zeros((n_imp, len(observed)))

        for m in range(n_imp):
            imp = observed.copy().astype(float)
            # Параметрическая неопределённость
            mu_draw = rng.normal(mu, sigma / np.sqrt(len(uncensored)))
            sigma_draw = sigma * np.sqrt(len(uncensored) / rng.chisquare(max(len(uncensored) - 1, 1)))

            for idx in cens_idx:
                lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else np.min(uncensored)
                log_lod = np.log(max(lod_i, 1e-15))
                z = (log_lod - mu_draw) / sigma_draw
                u = rng.uniform(0, max(stats.norm.cdf(z), 1e-10))
                log_val = mu_draw + sigma_draw * stats.norm.ppf(max(u, 1e-10))
                imp[idx] = np.exp(log_val)

            all_imputed[m] = imp

        return np.mean(all_imputed, axis=0)

    def _fallback_nonparametric(
        self,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fallback непараметрическая оценка через бутстрэп."""
        uncensored = observed[~censored]
        if len(uncensored) < 2:
            return None

        rng = np.random.default_rng(self.seed + 17)
        imputed = observed.copy().astype(float)
        cens_idx = np.where(censored)[0]

        if len(cens_idx) == 0:
            return imputed

        # Для каждого цензурированного: случайный выбор из наблюдений ниже LOD
        for idx in cens_idx:
            lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else np.min(uncensored)
            candidates = uncensored[uncensored <= lod_i]
            if len(candidates) > 0:
                imputed[idx] = rng.choice(candidates)
            else:
                imputed[idx] = rng.uniform(0, lod_i)

        return imputed

    def _generate_from_params(
        self,
        distribution: str,
        params: dict,
        observed: np.ndarray,
        censored: np.ndarray,
        lod_values: np.ndarray,
    ) -> np.ndarray:
        """Генерация импутированных данных из оценённых параметров MLE."""
        imputed = observed.copy().astype(float)
        rng = np.random.default_rng(self.seed + 3)

        cens_idx = np.where(censored)[0]
        if len(cens_idx) == 0:
            return imputed

        if distribution == "lognormal":
            mu = params.get("mu", params.get("mu_log", -4.6))
            sigma = params.get("sigma", params.get("sigma_log", 0.8))
            for idx in cens_idx:
                lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else 0.001
                log_lod = np.log(max(lod_i, 1e-15))
                z = (log_lod - mu) / max(sigma, 1e-10)
                u = rng.uniform(0, max(stats.norm.cdf(z), 1e-10))
                log_val = mu + sigma * stats.norm.ppf(max(u, 1e-10))
                imputed[idx] = np.exp(log_val)

        elif distribution == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 0.005)
            for idx in cens_idx:
                lod_i = lod_values[idx] if np.isfinite(lod_values[idx]) else 0.001
                p = stats.gamma.cdf(lod_i, shape, scale=scale)
                u = rng.uniform(0, max(p, 1e-10))
                imputed[idx] = stats.gamma.ppf(u, shape, scale=scale)

        return imputed

    # ─────────────────────── Bootstrap CI ───────────────────────

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        n_boot: int = 200,
        alpha: float = 0.05,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Bootstrap доверительные интервалы для среднего и p95.

        Returns
        -------
        ci_mean : (float, float)
        ci_p95 : (float, float)
        """
        rng = np.random.default_rng(self.seed + 99)
        n = len(data)
        if n < 2:
            return (np.nan, np.nan), (np.nan, np.nan)

        means = np.empty(n_boot)
        p95s = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.choice(data, size=n, replace=True)
            means[b] = np.mean(sample)
            p95s[b] = np.percentile(sample, 95)

        ci_mean = (
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
        )
        ci_p95 = (
            float(np.percentile(p95s, 100 * alpha / 2)),
            float(np.percentile(p95s, 100 * (1 - alpha / 2))),
        )
        return ci_mean, ci_p95

    # ─────────────────────── Один эксперимент ───────────────────────

    def run_single_experiment(
        self,
        true_values: np.ndarray,
        censoring_rate: float,
        censoring_type: str,
        methods: List[str],
        mpc: float,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Один эксперимент (одна реализация Монте-Карло).

        Parameters
        ----------
        true_values : np.ndarray
            Полные (нецензурированные) данные.
        censoring_rate : float
            Доля цензурирования (0, 1).
        censoring_type : str
            Тип цензурирования.
        methods : list of str
            Список методов для применения.
        mpc : float
            Предельно допустимая концентрация.
        seed : int, optional
            Seed для воспроизводимости цензурирования.

        Returns
        -------
        dict
            Результаты эксперимента для всех методов.
        """
        # a) Цензурировать данные
        observed, censored_mask, lod_values = self._censor_data(
            true_values, censoring_rate, censoring_type, seed=seed
        )

        actual_cens_rate = float(np.mean(censored_mask))

        # Истинные статистики
        true_stats = self._compute_true_statistics(true_values, mpc)

        # b) Для каждого метода
        method_results = {}
        for method_name in methods:
            res = self._apply_method(method_name, observed, censored_mask, lod_values)

            # c) Оценка
            evaluation = {}
            if res["converged"]:
                evaluation["mean_bias"] = res["estimated_mean"] - true_stats["mean"]
                evaluation["mean_error_sq"] = evaluation["mean_bias"] ** 2
                evaluation["variance_bias"] = res["estimated_variance"] - true_stats["variance"]
                evaluation["variance_error_sq"] = evaluation["variance_bias"] ** 2
                evaluation["p90_bias"] = res["estimated_p90"] - true_stats["p90"]
                evaluation["p95_bias"] = res["estimated_p95"] - true_stats["p95"]
                evaluation["p99_bias"] = res["estimated_p99"] - true_stats["p99"]
                evaluation["exceedance_error"] = (
                    res["estimated_exceedance"] - true_stats["exceedance_prob"]
                )
                evaluation["exceedance_error_abs"] = abs(evaluation["exceedance_error"])

                # KS statistic между импутированным и истинным распределением
                if res["imputed_data"] is not None and len(res["imputed_data"]) > 1:
                    ks_stat, ks_p = stats.ks_2samp(
                        true_stats["full_data"], res["imputed_data"]
                    )
                    evaluation["ks_statistic"] = ks_stat
                    evaluation["ks_pvalue"] = ks_p
                else:
                    evaluation["ks_statistic"] = np.nan
                    evaluation["ks_pvalue"] = np.nan

                # Coverage: попадает ли истинное значение в CI
                ci_lo, ci_hi = res["ci_mean"]
                evaluation["coverage_mean"] = (
                    1.0 if ci_lo <= true_stats["mean"] <= ci_hi else 0.0
                )
                ci_lo95, ci_hi95 = res["ci_p95"]
                evaluation["coverage_p95"] = (
                    1.0 if ci_lo95 <= true_stats["p95"] <= ci_hi95 else 0.0
                )

                # Относительные ошибки
                if abs(true_stats["mean"]) > 1e-15:
                    evaluation["mean_relative_error"] = (
                        evaluation["mean_bias"] / true_stats["mean"]
                    )
                else:
                    evaluation["mean_relative_error"] = np.nan
            else:
                for key in [
                    "mean_bias", "mean_error_sq", "variance_bias",
                    "variance_error_sq", "p90_bias", "p95_bias", "p99_bias",
                    "exceedance_error", "exceedance_error_abs", "ks_statistic",
                    "ks_pvalue", "coverage_mean", "coverage_p95",
                    "mean_relative_error",
                ]:
                    evaluation[key] = np.nan

            method_results[method_name] = {
                "estimates": res,
                "evaluation": evaluation,
                "converged": res["converged"],
                "computation_time": res["computation_time"],
            }

        return {
            "actual_censoring_rate": actual_cens_rate,
            "true_statistics": {k: v for k, v in true_stats.items() if k != "full_data"},
            "methods": method_results,
        }

    # ─────────────────────── Checkpoint ───────────────────────

    def _save_checkpoint(
        self,
        experiment_hash: str,
        scenario_key: tuple,
        sim_index: int,
        accumulated: dict,
    ) -> None:
        """Сохранение промежуточных результатов."""
        cp_path = self._checkpoint_dir / f"cp_{experiment_hash}_{hash(scenario_key)}_{sim_index}.pkl"
        try:
            with open(cp_path, "wb") as f:
                pickle.dump(
                    {
                        "scenario_key": scenario_key,
                        "sim_index": sim_index,
                        "accumulated": accumulated,
                    },
                    f,
                )
            logger.debug("Checkpoint сохранён: %s", cp_path)
        except Exception as e:
            logger.warning("Не удалось сохранить checkpoint: %s", e)

    def _load_checkpoint(
        self,
        experiment_hash: str,
        scenario_key: tuple,
    ) -> Optional[dict]:
        """Загрузка checkpoint если существует."""
        pattern = f"cp_{experiment_hash}_{hash(scenario_key)}_"
        candidates = sorted(self._checkpoint_dir.glob(pattern + "*.pkl"))
        if not candidates:
            return None
        try:
            with open(candidates[-1], "rb") as f:
                data = pickle.load(f)
            logger.info("Checkpoint загружен: %s (sim_index=%d)", candidates[-1], data["sim_index"])
            return data
        except Exception as e:
            logger.warning("Не удалось загрузить checkpoint: %s", e)
            return None

    def _clean_checkpoints(self, experiment_hash: str) -> None:
        """Удаление checkpoint-файлов после завершения."""
        for p in self._checkpoint_dir.glob(f"cp_{experiment_hash}_*"):
            try:
                p.unlink()
            except Exception:
                pass

    # ─────────────────────── Одна симуляция (для joblib) ───────────────────────

    def _run_one_simulation(
        self,
        sim_idx: int,
        distribution: str,
        sample_size: int,
        censoring_rate: float,
        censoring_type: str,
        methods: List[str],
        base_seed: int,
    ) -> dict:
        """Запуск одной симуляции (thread-safe для joblib)."""
        sim_seed = base_seed + sim_idx

        # Генерация
        true_values = self._generate_sample(distribution, sample_size, seed=sim_seed)

        # Эксперимент
        result = self.run_single_experiment(
            true_values=true_values,
            censoring_rate=censoring_rate,
            censoring_type=censoring_type,
            methods=methods,
            mpc=self.mpc,
            seed=sim_seed + 1_000_000,
        )

        return result

    # ─────────────────────── ПОЛНЫЙ МОНТЕ-КАРЛО ───────────────────────

    def run_monte_carlo(
        self,
        metal: str = "Pb",
        n_simulations: int = 1000,
        sample_sizes: Optional[List[int]] = None,
        censoring_rates: Optional[List[float]] = None,
        censoring_types: Optional[List[str]] = None,
        distributions: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        n_jobs: int = 1,
    ) -> dict:
        """
        Полная Монте-Карло симуляция.

        Parameters
        ----------
        metal : str
            Металл (для контекста).
        n_simulations : int
            Количество симуляций на каждую комбинацию параметров.
        sample_sizes : list of int
            Размеры выборок.
        censoring_rates : list of float
            Доли цензурирования.
        censoring_types : list of str
            Типы цензурирования.
        distributions : list of str
            Распределения.
        methods : list of str
            Методы. Если None — все доступные.
        n_jobs : int
            Количество процессов для joblib.

        Returns
        -------
        dict
            Полные результаты эксперимента.
        """
        # Значения по умолчанию
        if sample_sizes is None:
            sample_sizes = [50, 100, 200]
        if censoring_rates is None:
            censoring_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if censoring_types is None:
            censoring_types = ["fixed_lod"]
        if distributions is None:
            distributions = ["lognormal"]
        if methods is None:
            methods = ALL_METHODS.copy()

        # Конфигурация эксперимента
        exp_config = ExperimentConfig(
            metal=metal,
            n_simulations=n_simulations,
            sample_sizes=sample_sizes,
            censoring_rates=censoring_rates,
            censoring_types=censoring_types,
            distributions=distributions,
            methods=methods,
            n_jobs=n_jobs,
            seed=self.seed,
            save_raw=n_simulations <= 1000,
        )
        exp_hash = exp_config.experiment_hash()

        logger.info("=" * 70)
        logger.info("ЗАПУСК МОНТЕ-КАРЛО ЭКСПЕРИМЕНТА")
        logger.info("Хэш: %s", exp_hash)
        logger.info("Симуляций: %d", n_simulations)
        logger.info("Размеры выборок: %s", sample_sizes)
        logger.info("Доли цензурирования: %s", censoring_rates)
        logger.info("Распределения: %s", distributions)
        logger.info("Методы: %s", methods)
        total_scenarios = (
            len(distributions)
            * len(sample_sizes)
            * len(censoring_types)
            * len(censoring_rates)
        )
        total_runs = total_scenarios * n_simulations
        logger.info("Всего сценариев: %d, всего запусков: %d", total_scenarios, total_runs)
        logger.info("=" * 70)

        # Хранилища
        results = {}
        raw_data = {} if exp_config.save_raw else None
        start_time = time.perf_counter()

        # Прогресс
        scenario_pbar = tqdm(
            total=total_scenarios,
            desc="Сценарии",
            position=0,
            leave=True,
        )

        with InterruptHandler() as ih:
            for distribution in distributions:
                for sample_size in sample_sizes:
                    for censoring_type in censoring_types:
                        for censoring_rate in censoring_rates:
                            if ih.interrupted:
                                break

                            scenario_key = (
                                distribution,
                                sample_size,
                                censoring_type,
                                censoring_rate,
                            )

                            logger.info(
                                "Сценарий: dist=%s, n=%d, type=%s, rate=%.2f",
                                *scenario_key,
                            )

                            # Проверка checkpoint
                            cp = self._load_checkpoint(exp_hash, scenario_key)
                            start_sim = 0
                            accumulated = self._init_accumulator(methods)
                            if cp is not None:
                                start_sim = cp["sim_index"] + 1
                                accumulated = cp["accumulated"]
                                logger.info(
                                    "Продолжение с симуляции %d/%d",
                                    start_sim,
                                    n_simulations,
                                )

                            sim_raw = [] if exp_config.save_raw else None

                            # Запуск симуляций
                            remaining = n_simulations - start_sim
                            base_seed = (
                                self.seed
                                + hash(scenario_key) % (2 ** 31)
                            )

                            use_parallel = (
                                HAS_JOBLIB
                                and n_jobs != 1
                                and remaining > 10
                            )

                            if use_parallel:
                                batch_size = min(remaining, exp_config.checkpoint_every)
                                for batch_start in range(start_sim, n_simulations, batch_size):
                                    if ih.interrupted:
                                        break
                                    batch_end = min(batch_start + batch_size, n_simulations)
                                    batch_indices = range(batch_start, batch_end)

                                    batch_results = Parallel(
                                        n_jobs=n_jobs, prefer="threads"
                                    )(
                                        delayed(self._run_one_simulation)(
                                            sim_idx=i,
                                            distribution=distribution,
                                            sample_size=sample_size,
                                            censoring_rate=censoring_rate,
                                            censoring_type=censoring_type,
                                            methods=methods,
                                            base_seed=base_seed,
                                        )
                                        for i in batch_indices
                                    )

                                    for sim_result in batch_results:
                                        self._accumulate(accumulated, sim_result, methods)
                                        if sim_raw is not None:
                                            sim_raw.append(sim_result)

                                    # Checkpoint
                                    self._save_checkpoint(
                                        exp_hash, scenario_key, batch_end - 1, accumulated
                                    )

                            else:
                                sim_pbar = tqdm(
                                    range(start_sim, n_simulations),
                                    desc=f"  n={sample_size}, cens={censoring_rate:.0%}",
                                    position=1,
                                    leave=False,
                                )
                                for i in sim_pbar:
                                    if ih.interrupted:
                                        break

                                    sim_result = self._run_one_simulation(
                                        sim_idx=i,
                                        distribution=distribution,
                                        sample_size=sample_size,
                                        censoring_rate=censoring_rate,
                                        censoring_type=censoring_type,
                                        methods=methods,
                                        base_seed=base_seed,
                                    )

                                    self._accumulate(accumulated, sim_result, methods)
                                    if sim_raw is not None:
                                        sim_raw.append(sim_result)

                                    # Промежуточное сохранение
                                    if (i + 1) % exp_config.checkpoint_every == 0:
                                        self._save_checkpoint(
                                            exp_hash, scenario_key, i, accumulated
                                        )

                                    # ETA
                                    elapsed = time.perf_counter() - start_time
                                    done_total = scenario_pbar.n * n_simulations + (i - start_sim + 1)
                                    if done_total > 0:
                                        eta = elapsed / done_total * (total_runs - done_total)
                                        sim_pbar.set_postfix(
                                            ETA=str(timedelta(seconds=int(eta)))
                                        )

                                sim_pbar.close()

                            # Агрегация результатов сценария
                            scenario_results = self._finalize_accumulator(
                                accumulated, methods
                            )
                            results[scenario_key] = scenario_results

                            if raw_data is not None and sim_raw is not None:
                                raw_data[scenario_key] = sim_raw

                            scenario_pbar.update(1)

                        if ih.interrupted:
                            break
                    if ih.interrupted:
                        break
                if ih.interrupted:
                    break

        scenario_pbar.close()

        total_time = time.perf_counter() - start_time
        logger.info("Монте-Карло завершён за %.1f сек (%.1f мин)", total_time, total_time / 60)

        # Формирование итоговых таблиц
        summary_tables = self._build_summary_tables(results, methods)

        # Рекомендации
        recommendations = self.generate_recommendation(results)

        # Очистка checkpoint-ов
        if not ih.interrupted:
            self._clean_checkpoints(exp_hash)

        output = {
            "experiment_config": exp_config.to_dict(),
            "results": results,
            "summary_tables": summary_tables,
            "recommendations": recommendations,
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
        }
        if raw_data is not None:
            output["raw_data"] = raw_data

        return output

    # ─────────────────────── Аккумулятор результатов ───────────────────────

    def _init_accumulator(self, methods: List[str]) -> dict:
        """Инициализация аккумулятора для агрегации результатов."""
        metrics = [
            "mean_bias", "mean_error_sq", "variance_bias", "variance_error_sq",
            "p90_bias", "p95_bias", "p99_bias",
            "exceedance_error", "exceedance_error_abs",
            "ks_statistic", "coverage_mean", "coverage_p95",
            "computation_time", "mean_relative_error",
        ]
        acc = {}
        for method in methods:
            acc[method] = {
                "count": 0,
                "converged_count": 0,
            }
            for m in metrics:
                acc[method][m] = []
        return acc

    def _accumulate(
        self,
        accumulator: dict,
        sim_result: dict,
        methods: List[str],
    ) -> None:
        """Добавление результата одной симуляции в аккумулятор."""
        for method in methods:
            if method not in sim_result["methods"]:
                continue
            mr = sim_result["methods"][method]
            acc = accumulator[method]
            acc["count"] += 1

            if mr["converged"]:
                acc["converged_count"] += 1
                ev = mr["evaluation"]
                for key in [
                    "mean_bias", "mean_error_sq", "variance_bias",
                    "variance_error_sq", "p90_bias", "p95_bias", "p99_bias",
                    "exceedance_error", "exceedance_error_abs", "ks_statistic",
                    "coverage_mean", "coverage_p95", "mean_relative_error",
                ]:
                    val = ev.get(key, np.nan)
                    if np.isfinite(val):
                        acc[key].append(val)

            acc["computation_time"].append(mr.get("computation_time", np.nan))

    def _finalize_accumulator(
        self,
        accumulator: dict,
        methods: List[str],
    ) -> dict:
        """Финализация аккумулятора: вычисление средних, RMSE и т.д."""
        result = {}
        for method in methods:
            acc = accumulator[method]
            n_total = acc["count"]
            n_converged = acc["converged_count"]
            convergence_rate = n_converged / n_total if n_total > 0 else 0.0

            def _safe_mean(lst):
                return float(np.mean(lst)) if len(lst) > 0 else np.nan

            def _safe_std(lst):
                return float(np.std(lst, ddof=1)) if len(lst) > 1 else np.nan

            mean_bias = _safe_mean(acc["mean_bias"])
            mean_rmse = float(np.sqrt(_safe_mean(acc["mean_error_sq"])))
            variance_bias = _safe_mean(acc["variance_bias"])
            variance_rmse = float(np.sqrt(_safe_mean(acc["variance_error_sq"])))

            result[method] = {
                "mean_bias": mean_bias,
                "mean_rmse": mean_rmse,
                "mean_bias_std": _safe_std(acc["mean_bias"]),
                "variance_bias": variance_bias,
                "variance_rmse": variance_rmse,
                "p90_bias": _safe_mean(acc["p90_bias"]),
                "p95_bias": _safe_mean(acc["p95_bias"]),
                "p99_bias": _safe_mean(acc["p99_bias"]),
                "exceedance_error": _safe_mean(acc["exceedance_error"]),
                "exceedance_error_abs": _safe_mean(acc["exceedance_error_abs"]),
                "ks_statistic": _safe_mean(acc["ks_statistic"]),
                "coverage_mean": _safe_mean(acc["coverage_mean"]),
                "coverage_p95": _safe_mean(acc["coverage_p95"]),
                "convergence_rate": convergence_rate,
                "computation_time": _safe_mean(acc["computation_time"]),
                "mean_relative_error": _safe_mean(acc["mean_relative_error"]),
                "n_simulations": n_total,
                "n_converged": n_converged,
            }

        return result

    # ─────────────────────── Сводные таблицы ───────────────────────

    def _build_summary_tables(
        self,
        results: dict,
        methods: List[str],
    ) -> dict:
        """
        Построение сводных таблиц из результатов.

        Returns
        -------
        dict
            Словарь с DataFrames: bias_table, rmse_table, ranking_table,
            recommendation_table.
        """
        rows_bias = []
        rows_rmse = []

        for scenario_key, scenario_results in results.items():
            dist, n, cens_type, cens_rate = scenario_key

            bias_row = {
                "distribution": dist,
                "sample_size": n,
                "censoring_type": cens_type,
                "censoring_rate": cens_rate,
            }
            rmse_row = bias_row.copy()

            for method in methods:
                if method in scenario_results:
                    bias_row[method] = scenario_results[method].get("mean_bias", np.nan)
                    rmse_row[method] = scenario_results[method].get("mean_rmse", np.nan)
                else:
                    bias_row[method] = np.nan
                    rmse_row[method] = np.nan

            rows_bias.append(bias_row)
            rows_rmse.append(rmse_row)

        bias_table = pd.DataFrame(rows_bias) if rows_bias else pd.DataFrame()
        rmse_table = pd.DataFrame(rows_rmse) if rows_rmse else pd.DataFrame()

        # Ranking table: для каждого сценария ранжируем методы по RMSE
        ranking_rows = []
        for _, row in rmse_table.iterrows():
            method_rmses = {m: row.get(m, np.nan) for m in methods if m in row}
            valid = {k: v for k, v in method_rmses.items() if np.isfinite(v)}
            if valid:
                sorted_methods = sorted(valid.items(), key=lambda x: x[1])
                ranking_row = {
                    "distribution": row.get("distribution"),
                    "sample_size": row.get("sample_size"),
                    "censoring_type": row.get("censoring_type"),
                    "censoring_rate": row.get("censoring_rate"),
                }
                for rank, (method, rmse_val) in enumerate(sorted_methods, 1):
                    ranking_row[f"rank_{rank}"] = method
                    ranking_row[f"rmse_{rank}"] = rmse_val
                ranking_rows.append(ranking_row)

        ranking_table = pd.DataFrame(ranking_rows) if ranking_rows else pd.DataFrame()

        # Recommendation table: средний ранг каждого метода
        method_ranks = defaultdict(list)
        for _, row in rmse_table.iterrows():
            method_rmses = {m: row.get(m, np.nan) for m in methods if m in row}
            valid = {k: v for k, v in method_rmses.items() if np.isfinite(v)}
            if valid:
                sorted_methods = sorted(valid.items(), key=lambda x: x[1])
                for rank, (method, _) in enumerate(sorted_methods, 1):
                    method_ranks[method].append(rank)

        rec_rows = []
        for method in methods:
            ranks = method_ranks.get(method, [])
            rec_rows.append({
                "method": method,
                "mean_rank": float(np.mean(ranks)) if ranks else np.nan,
                "median_rank": float(np.median(ranks)) if ranks else np.nan,
                "best_rank": int(np.min(ranks)) if ranks else np.nan,
                "worst_rank": int(np.max(ranks)) if ranks else np.nan,
                "n_scenarios": len(ranks),
                "n_first_place": sum(1 for r in ranks if r == 1),
            })

        recommendation_table = pd.DataFrame(rec_rows).sort_values(
            "mean_rank", ascending=True, ignore_index=True
        )

        return {
            "bias_table": bias_table,
            "rmse_table": rmse_table,
            "ranking_table": ranking_table,
            "recommendation_table": recommendation_table,
        }

    # ─────────────────────── Рекомендации ───────────────────────

    def generate_recommendation(self, results: dict) -> dict:
        """
        Генерация рекомендаций на основе результатов Монте-Карло.

        Parameters
        ----------
        results : dict
            Результаты из run_monte_carlo (ключ 'results' или весь словарь).

        Returns
        -------
        dict
            Рекомендации.
        """
        # Если передан полный output, извлекаем results
        if "results" in results and isinstance(results["results"], dict):
            res_data = results["results"]
        else:
            res_data = results

        # Собираем средние RMSE по всем сценариям
        method_rmse_all = defaultdict(list)
        method_rmse_by_rate = defaultdict(lambda: defaultdict(list))
        method_metrics = defaultdict(lambda: defaultdict(list))

        for scenario_key, scenario_results in res_data.items():
            if not isinstance(scenario_results, dict):
                continue

            cens_rate = scenario_key[3] if isinstance(scenario_key, tuple) and len(scenario_key) > 3 else 0.0

            for method, mr in scenario_results.items():
                if not isinstance(mr, dict):
                    continue

                rmse = mr.get("mean_rmse", np.nan)
                if np.isfinite(rmse):
                    method_rmse_all[method].append(rmse)

                    if cens_rate < 0.2:
                        method_rmse_by_rate["low"][method].append(rmse)
                    elif cens_rate <= 0.5:
                        method_rmse_by_rate["moderate"][method].append(rmse)
                    else:
                        method_rmse_by_rate["high"][method].append(rmse)

                # По метрикам
                var_rmse = mr.get("variance_rmse", np.nan)
                if np.isfinite(var_rmse):
                    method_metrics["variance_estimation"][method].append(var_rmse)

                p95_bias = mr.get("p95_bias", np.nan)
                if np.isfinite(p95_bias):
                    method_metrics["upper_tail"][method].append(abs(p95_bias))

                exc = mr.get("exceedance_error_abs", np.nan)
                if np.isfinite(exc):
                    method_metrics["exceedance_probability"][method].append(exc)

        def _best_method(method_scores: dict) -> str:
            if not method_scores:
                return "N/A"
            avg = {m: np.mean(v) for m, v in method_scores.items() if len(v) > 0}
            if not avg:
                return "N/A"
            return min(avg, key=avg.get)

        # Лучший в целом
        best_overall = _best_method(method_rmse_all)

        # Лучший по уровню цензурирования
        best_by_censoring = {}
        for level, label in [("low", "low (< 20%)"), ("moderate", "moderate (20-50%)"), ("high", "high (> 50%)")]:
            best_by_censoring[label] = _best_method(method_rmse_by_rate.get(level, {}))

        # Лучший по метрике
        best_by_metric = {
            "mean_estimation": best_overall,
        }
        for metric in ["variance_estimation", "upper_tail", "exceedance_probability"]:
            best_by_metric[metric] = _best_method(method_metrics.get(metric, {}))

        # Методы, которых стоит избегать (средний RMSE в 2+ раз хуже лучшего)
        avoid = []
        if method_rmse_all:
            avg_rmse = {m: np.mean(v) for m, v in method_rmse_all.items() if len(v) > 0}
            if avg_rmse:
                best_rmse_val = min(avg_rmse.values())
                for method, rmse_val in avg_rmse.items():
                    if rmse_val > 2 * best_rmse_val and best_rmse_val > 0:
                        avoid.append(method)

        # Комментарий
        commentary_parts = [
            f"Лучший метод в целом: {best_overall}.",
        ]
        if avoid:
            commentary_parts.append(
                f"Методы, которые не рекомендуются из-за высокой ошибки: {', '.join(avoid)}."
            )
        if best_by_censoring.get("high (> 50%)") != best_overall:
            commentary_parts.append(
                f"При высоком цензурировании (>50%) лучше использовать "
                f"{best_by_censoring.get('high (> 50%)', 'N/A')}."
            )
        commentary_parts.append(
            "Результаты основаны на Монте-Карло симуляции. "
            "Рекомендации могут зависеть от конкретных условий задачи."
        )

        return {
            "best_overall": best_overall,
            "best_by_censoring_rate": best_by_censoring,
            "best_by_metric": best_by_metric,
            "avoid": avoid,
            "commentary": " ".join(commentary_parts),
        }

    # ─────────────────────── Сохранение отчёта ───────────────────────

    def save_full_report(
        self,
        results: dict,
        output_dir: str = "reports",
    ) -> str:
        """
        Сохранение полного отчёта эксперимента.

        Parameters
        ----------
        results : dict
            Результаты из run_monte_carlo.
        output_dir : str
            Директория для сохранения.

        Returns
        -------
        str
            Путь к директории с отчётом.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(output_dir) / f"mc_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Сохранение отчёта в %s", report_dir)

        # 1. Таблицы в CSV
        tables_dir = report_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        summary_tables = results.get("summary_tables", {})
        for name, df in summary_tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                csv_path = tables_dir / f"{name}.csv"
                df.to_csv(csv_path, index=False)
                logger.info("  Сохранено: %s", csv_path)

                # Excel
                try:
                    xlsx_path = tables_dir / f"{name}.xlsx"
                    df.to_excel(xlsx_path, index=False)
                except Exception:
                    pass

        # 2. Графики
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if self.visualizer is not None:
            try:
                self.visualizer.plot_all(
                    results=results,
                    output_dir=str(plots_dir),
                )
            except Exception as e:
                logger.warning("Не удалось создать графики: %s", e)
        else:
            # Минимальные графики через matplotlib напрямую
            self._create_basic_plots(results, plots_dir)

        # 3. Текстовый отчёт в Markdown
        md_path = report_dir / "report.md"
        self._write_markdown_report(results, md_path)

        # 4. Рекомендации
        rec_path = report_dir / "recommendations.json"
        recommendations = results.get("recommendations", {})
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False, default=str)

        # 5. Метаданные
        meta_path = report_dir / "metadata.json"
        metadata = {
            "experiment_config": results.get("experiment_config", {}),
            "total_time_seconds": results.get("total_time_seconds", 0),
            "timestamp": results.get("timestamp", timestamp),
            "n_scenarios": len(results.get("results", {})),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        # 6. Полные данные (pickle)
        pickle_path = report_dir / "full_results.pkl"
        results_to_save = {k: v for k, v in results.items() if k != "raw_data"}
        with open(pickle_path, "wb") as f:
            pickle.dump(results_to_save, f)

        logger.info("Отчёт сохранён в %s", report_dir)
        return str(report_dir)

    def _create_basic_plots(self, results: dict, plots_dir: Path) -> None:
        """Создание базовых графиков без модуля визуализации."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            summary = results.get("summary_tables", {})
            rmse_table = summary.get("rmse_table")
            if rmse_table is None or rmse_table.empty:
                return

            # RMSE по методам и долям цензурирования
            methods_in_table = [
                c for c in rmse_table.columns
                if c not in ["distribution", "sample_size", "censoring_type", "censoring_rate"]
            ]

            if not methods_in_table:
                return

            fig, ax = plt.subplots(figsize=(14, 8))
            for method in methods_in_table:
                grouped = rmse_table.groupby("censoring_rate")[method].mean()
                ax.plot(grouped.index, grouped.values, marker="o", label=method[:20])
            ax.set_xlabel("Доля цензурирования")
            ax.set_ylabel("RMSE (среднее)")
            ax.set_title("RMSE по методам vs. доля цензурирования")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(plots_dir / "rmse_vs_censoring.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Recommendation ranking
            rec_table = summary.get("recommendation_table")
            if rec_table is not None and not rec_table.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                methods_sorted = rec_table.sort_values("mean_rank")
                ax.barh(
                    methods_sorted["method"],
                    methods_sorted["mean_rank"],
                    color="steelblue",
                )
                ax.set_xlabel("Средний ранг (меньше = лучше)")
                ax.set_title("Рейтинг методов")
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(plots_dir / "method_ranking.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            logger.info("Базовые графики сохранены в %s", plots_dir)

        except ImportError:
            logger.warning("matplotlib недоступен — графики не созданы")
        except Exception as e:
            logger.warning("Ошибка создания графиков: %s", e)

    def _write_markdown_report(self, results: dict, path: Path) -> None:
        """Генерация Markdown отчёта."""
        config = results.get("experiment_config", {})
        recommendations = results.get("recommendations", {})
        summary = results.get("summary_tables", {})

        lines = [
            "# Отчёт Монте-Карло эксперимента",
            "",
            f"**Дата:** {results.get('timestamp', 'N/A')}",
            f"**Время выполнения:** {results.get('total_time_seconds', 0):.1f} сек",
            "",
            "## Конфигурация",
            "",
            f"- Количество симуляций: {config.get('n_simulations', 'N/A')}",
            f"- Размеры выборок: {config.get('sample_sizes', 'N/A')}",
            f"- Доли цензурирования: {config.get('censoring_rates', 'N/A')}",
            f"- Распределения: {config.get('distributions', 'N/A')}",
            f"- Методы: {config.get('methods', 'N/A')}",
            "",
            "## Рекомендации",
            "",
            f"**Лучший метод в целом:** {recommendations.get('best_overall', 'N/A')}",
            "",
            "### По уровню цензурирования:",
            "",
        ]

        for level, method in recommendations.get("best_by_censoring_rate", {}).items():
            lines.append(f"- {level}: **{method}**")

        lines += [
            "",
            "### По метрике:",
            "",
        ]
        for metric, method in recommendations.get("best_by_metric", {}).items():
            lines.append(f"- {metric}: **{method}**")

        avoid = recommendations.get("avoid", [])
        if avoid:
            lines += [
                "",
                "### Методы, которые НЕ рекомендуются:",
                "",
            ]
            for m in avoid:
                lines.append(f"- {m}")

        lines += [
            "",
            "### Комментарий:",
            "",
            recommendations.get("commentary", ""),
            "",
        ]

        # Таблица ранжирования
        rec_table = summary.get("recommendation_table")
        if rec_table is not None and not rec_table.empty:
            lines += [
                "## Рейтинг методов",
                "",
                rec_table.to_markdown(index=False),
                "",
            ]

        # Сводная таблица RMSE
        rmse_table = summary.get("rmse_table")
        if rmse_table is not None and not rmse_table.empty:
            lines += [
                "## Таблица RMSE",
                "",
            ]
            # Ограничиваем ширину
            if len(rmse_table) <= 50:
                lines.append(rmse_table.to_markdown(index=False))
            else:
                lines.append(f"*(Таблица слишком большая — {len(rmse_table)} строк. См. CSV файл.)*")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Markdown отчёт: %s", path)

    # ─────────────────────── Анализ чувствительности ───────────────────────

    def run_sensitivity_analysis(
        self,
        base_results: Optional[dict] = None,
        n_simulations: int = 200,
        sample_size: int = 100,
        methods: Optional[List[str]] = None,
    ) -> dict:
        """
        Анализ чувствительности.

        Проверяет:
        a) Чувствительность к форме распределения (model misspecification).
        b) Чувствительность к размеру выборки.
        c) Чувствительность к типу цензурирования (fixed vs variable LOD).
        d) Чувствительность к выбросам.

        Parameters
        ----------
        base_results : dict, optional
            Базовые результаты для сравнения.
        n_simulations : int
            Количество симуляций для каждого теста.
        sample_size : int
            Размер выборки.
        methods : list, optional
            Методы для тестирования.

        Returns
        -------
        dict
            Результаты анализа чувствительности.
        """
        if methods is None:
            methods = ALL_METHODS.copy()

        logger.info("=" * 50)
        logger.info("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        logger.info("=" * 50)

        sensitivity_results = {}

        # a) Чувствительность к форме распределения
        logger.info("(a) Чувствительность к форме распределения...")
        dist_sensitivity = self.run_monte_carlo(
            n_simulations=n_simulations,
            sample_sizes=[sample_size],
            censoring_rates=[0.2, 0.4, 0.6],
            distributions=["lognormal", "gamma", "weibull"],
            methods=methods,
            n_jobs=1,
        )
        sensitivity_results["distribution_sensitivity"] = dist_sensitivity.get("results", {})

        # Анализ misspecification: данные из gamma, метод предполагает lognormal
        logger.info("  Model misspecification test...")
        misspec_results = {}
        for cens_rate in [0.2, 0.4, 0.6]:
            acc = self._init_accumulator(methods)
            for i in range(n_simulations):
                # Данные из gamma
                true_data = self._generate_sample("gamma", sample_size, seed=self.seed + i + 50000)
                result = self.run_single_experiment(
                    true_values=true_data,
                    censoring_rate=cens_rate,
                    censoring_type="fixed_lod",
                    methods=methods,
                    mpc=self.mpc,
                    seed=self.seed + i + 60000,
                )
                self._accumulate(acc, result, methods)

            misspec_results[cens_rate] = self._finalize_accumulator(acc, methods)

        sensitivity_results["model_misspecification"] = misspec_results

        # b) Чувствительность к размеру выборки
        logger.info("(b) Чувствительность к размеру выборки...")
        size_sensitivity = self.run_monte_carlo(
            n_simulations=n_simulations,
            sample_sizes=[20, 50, 100, 200, 500],
            censoring_rates=[0.3],
            distributions=["lognormal"],
            methods=methods,
            n_jobs=1,
        )
        sensitivity_results["sample_size_sensitivity"] = size_sensitivity.get("results", {})

        # c) Чувствительность к типу цензурирования
        logger.info("(c) Чувствительность к типу цензурирования...")
        cens_type_results = {}
        for cens_type in ["fixed_lod", "variable_lod", "random"]:
            acc = self._init_accumulator(methods)
            for i in range(n_simulations):
                true_data = self._generate_sample("lognormal", sample_size, seed=self.seed + i + 70000)
                result = self.run_single_experiment(
                    true_values=true_data,
                    censoring_rate=0.3,
                    censoring_type=cens_type,
                    methods=methods,
                    mpc=self.mpc,
                    seed=self.seed + i + 80000,
                )
                self._accumulate(acc, result, methods)

            cens_type_results[cens_type] = self._finalize_accumulator(acc, methods)

        sensitivity_results["censoring_type_sensitivity"] = cens_type_results

        # d) Чувствительность к выбросам
        logger.info("(d) Чувствительность к выбросам...")
        outlier_results = {}
        for outlier_fraction in [0.0, 0.02, 0.05, 0.10]:
            acc = self._init_accumulator(methods)
            for i in range(n_simulations):
                true_data = self._generate_sample("lognormal", sample_size, seed=self.seed + i + 90000)

                # Добавляем выбросы
                n_outliers = int(outlier_fraction * sample_size)
                if n_outliers > 0:
                    rng = np.random.default_rng(self.seed + i + 100000)
                    outlier_indices = rng.choice(sample_size, n_outliers, replace=False)
                    # Выбросы = 10-100x от максимума
                    outlier_mult = rng.uniform(10, 100, n_outliers)
                    true_data[outlier_indices] *= outlier_mult

                result = self.run_single_experiment(
                    true_values=true_data,
                    censoring_rate=0.3,
                    censoring_type="fixed_lod",
                    methods=methods,
                    mpc=self.mpc,
                    seed=self.seed + i + 110000,
                )
                self._accumulate(acc, result, methods)

            outlier_results[outlier_fraction] = self._finalize_accumulator(acc, methods)

        sensitivity_results["outlier_sensitivity"] = outlier_results

        logger.info("Анализ чувствительности завершён.")
        return sensitivity_results


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Монте-Карло эксперимент по сравнению методов обработки цензурированных данных"
    )
    parser.add_argument("--n-simulations", type=int, default=100, help="Количество симуляций")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[50, 100], help="Размеры выборок")
    parser.add_argument(
        "--censoring-rates",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7],
        help="Доли цензурирования",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="Количество процессов")
    parser.add_argument("--output-dir", type=str, default="reports", help="Директория для отчёта")
    parser.add_argument("--mpc", type=float, default=0.01, help="ПДК")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Запустить анализ чувствительности",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Методы (по умолчанию — все)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый запуск (малые параметры)",
    )

    args = parser.parse_args()

    if args.quick:
        args.n_simulations = 20
        args.sample_sizes = [50]
        args.censoring_rates = [0.2, 0.5]

    config = {
        "seed": args.seed,
        "mpc": args.mpc,
    }

    mc = MonteCarloSimulation(config)

    print("=" * 70)
    print("МОНТЕ-КАРЛО ЭКСПЕРИМЕНТ: СРАВНЕНИЕ МЕТОДОВ ОБРАБОТКИ")
    print("ЦЕНЗУРИРОВАННЫХ ДАННЫХ")
    print("=" * 70)
    print(f"Симуляций: {args.n_simulations}")
    print(f"Размеры выборок: {args.sample_sizes}")
    print(f"Доли цензурирования: {args.censoring_rates}")
    print(f"Методы: {args.methods or 'все'}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Основной эксперимент
    results = mc.run_monte_carlo(
        n_simulations=args.n_simulations,
        sample_sizes=args.sample_sizes,
        censoring_rates=args.censoring_rates,
        methods=args.methods,
        n_jobs=args.n_jobs,
    )

    # Рекомендации
    recommendations = results.get("recommendations", {})
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 70)
    print(f"Лучший метод в целом: {recommendations.get('best_overall', 'N/A')}")
    print("\nПо уровню цензурирования:")
    for level, method in recommendations.get("best_by_censoring_rate", {}).items():
        print(f"  {level}: {method}")
    print("\nПо метрике:")
    for metric, method in recommendations.get("best_by_metric", {}).items():
        print(f"  {metric}: {method}")
    avoid = recommendations.get("avoid", [])
    if avoid:
        print(f"\nНе рекомендуются: {', '.join(avoid)}")
    print(f"\n{recommendations.get('commentary', '')}")

    # Таблица ранжирования
    rec_table = results.get("summary_tables", {}).get("recommendation_table")
    if rec_table is not None and not rec_table.empty:
        print("\n" + "=" * 70)
        print("РЕЙТИНГ МЕТОДОВ")
        print("=" * 70)
        print(rec_table.to_string(index=False))

    # Анализ чувствительности
    if args.sensitivity:
        print("\n" + "=" * 70)
        print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        print("=" * 70)
        sens = mc.run_sensitivity_analysis(
            base_results=results,
            n_simulations=max(args.n_simulations // 5, 20),
            methods=args.methods,
        )
        results["sensitivity_analysis"] = sens

    # Сохранение отчёта
    report_path = mc.save_full_report(results, output_dir=args.output_dir)
    print(f"\nОтчёт сохранён: {report_path}")
    print(f"Общее время: {results.get('total_time_seconds', 0):.1f} сек")