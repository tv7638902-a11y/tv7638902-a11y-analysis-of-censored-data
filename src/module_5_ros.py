"""
src/module_5_ros.py

Реализация метода ROS (Regression on Order Statistics) для обработки
цензурированных данных (данные с пределом обнаружения, LOD / non-detects).

Метод ROS — полупараметрический метод, рекомендованный EPA (ProUCL) и
Helsel (2005, 2012). Он строит вероятностный график (probability plot)
по нецензурированным данным, подгоняет регрессионную прямую и
экстраполирует её для получения оценок цензурированных значений.

Математическое описание алгоритма
---------------------------------
1. Упорядочить все наблюдения (включая цензурированные как ≤ LOD).
2. Вычислить plotting positions методом Blom:
       pᵢ = (i - 0.375) / (n + 0.25)
   Для цензурированных наблюдений используются модифицированные позиции
   (метод Helsel-Cohn).
3. Построить вероятностный график:
       X-ось: Φ⁻¹(pᵢ)  — квантили стандартного нормального распределения
       Y-ось: наблюдённые значения (или их логарифмы для логнормального)
4. Подогнать регрессионную прямую ТОЛЬКО по нецензурированным точкам:
       y = a + b · Φ⁻¹(p)
5. Экстраполировать прямую для цензурированных:
       x_imputed = a + b · Φ⁻¹(pᵢ)
6. Для логнормального ROS: x_final = exp(x_imputed).

Литература:
- Helsel, D.R. (2005, 2012). Statistics for Censored Environmental Data
  Using Minitab and R.
- USEPA (2015). ProUCL Technical Guide.
- Lee, L. & Helsel, D. (2005). Computer programs for the analysis of
  multiply-censored environmental data.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.figure


# ---------------------------------------------------------------------------
# Настройка логгера модуля
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class ROSMethod:
    """
    Класс для применения метода ROS (Regression on Order Statistics).

    Parameters
    ----------
    distribution : str
        Предполагаемое распределение данных:
        - "lognormal" (по умолчанию) — работаем с ln(x);
        - "normal"    — работаем непосредственно с x.
    """

    SUPPORTED_DISTRIBUTIONS = ("lognormal", "normal")
    SUPPORTED_PP_METHODS = ("blom", "weibull", "hazen", "cunnane")
    MIN_UNCENSORED = 3
    R2_WARN_THRESHOLD = 0.7

    def __init__(self, distribution: str = "lognormal"):
        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self.SUPPORTED_DISTRIBUTIONS}, "
                f"got '{distribution}'."
            )
        self.distribution = distribution
        self.logger = logger
        self.logger.info(f"ROSMethod инициализирован (distribution={distribution}).")

    # ------------------------------------------------------------------ #
    # Внутренние методы
    # ------------------------------------------------------------------ #
    def _compute_plotting_positions(
        self,
        n_total: int,
        censored_flags: np.ndarray,
        method: str = "blom",
    ) -> np.ndarray:
        """
        Вычисляет plotting positions для каждого (отсортированного) наблюдения.

        Формулы:
          * blom:     pᵢ = (i - 0.375) / (n + 0.25)   (по умолчанию)
          * weibull:  pᵢ = i / (n + 1)
          * hazen:    pᵢ = (i - 0.5) / n
          * cunnane:  pᵢ = (i - 0.4) / (n + 0.2)

        Для цензурированных наблюдений реализован метод Helsel-Cohn:
        plotting positions цензурированных перераспределяются так, чтобы
        они занимали нижнюю часть распределения, пропорциональную доле
        наблюдений ≤ LOD.

        Parameters
        ----------
        n_total : int
            Общее количество наблюдений.
        censored_flags : np.ndarray[bool]
            Массив флагов (True = цензурированное наблюдение).
            Предполагается, что данные уже отсортированы по возрастанию.
        method : str
            Метод plotting positions.

        Returns
        -------
        np.ndarray
            Массив plotting positions длиной n_total, pᵢ ∈ (0, 1).
        """
        if method not in self.SUPPORTED_PP_METHODS:
            raise ValueError(
                f"method must be one of {self.SUPPORTED_PP_METHODS}, got '{method}'."
            )

        n = n_total
        ranks = np.arange(1, n + 1, dtype=float)

        if method == "blom":
            a_const = 0.375
            pp_full = (ranks - a_const) / (n + 1 - 2 * a_const)
        elif method == "weibull":
            pp_full = ranks / (n + 1)
        elif method == "hazen":
            pp_full = (ranks - 0.5) / n
        else:  # cunnane
            pp_full = (ranks - 0.4) / (n + 0.2)

        # Модификация Helsel-Cohn для цензурированных:
        # Доля цензурированных определяет верхнюю границу их plotting positions.
        n_censored = int(np.sum(censored_flags))
        n_uncensored = n - n_censored

        if n_censored == 0 or n_uncensored == 0:
            return pp_full

        # F(LOD) — доля значений, не превышающих LOD.
        f_lod = n_censored / n

        # Цензурированные позиции: равномерно распределены в (0, F(LOD))
        censored_ranks = np.arange(1, n_censored + 1, dtype=float)
        pp_censored = (censored_ranks - 0.375) / (n_censored + 0.25) * f_lod

        # Нецензурированные позиции: в (F(LOD), 1), по формуле Blom со сдвигом
        uncens_ranks = np.arange(1, n_uncensored + 1, dtype=float)
        pp_uncensored = f_lod + (uncens_ranks - 0.375) / (n_uncensored + 0.25) * (
            1 - f_lod
        )

        pp = np.empty(n, dtype=float)
        pp[censored_flags] = pp_censored
        pp[~censored_flags] = pp_uncensored

        # Защита от 0 и 1 (для Φ⁻¹)
        eps = 1e-6
        pp = np.clip(pp, eps, 1 - eps)
        return pp

    def _fit_regression(
        self,
        plotting_positions: np.ndarray,
        observed_values: np.ndarray,
        censored_flags: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Подгоняет регрессионную прямую y = a + b·Φ⁻¹(p) по нецензурированным
        наблюдениям.

        Для lognormal: y = ln(x); для normal: y = x.

        Returns
        -------
        dict
            {
              "intercept": float,
              "slope": float,
              "r_squared": float,
              "std_error": float,
              "residuals": np.ndarray
            }
        """
        uncens_mask = ~censored_flags
        n_uncens = int(np.sum(uncens_mask))

        if n_uncens < self.MIN_UNCENSORED:
            self.logger.warning(
                f"Недостаточно нецензурированных наблюдений для надёжной "
                f"регрессии (n={n_uncens} < {self.MIN_UNCENSORED}). "
                f"Fallback: используем простые оценки."
            )
            # Fallback — используем среднее и std по имеющимся
            vals = observed_values[uncens_mask]
            if self.distribution == "lognormal":
                vals = np.log(np.clip(vals, 1e-12, None))
            mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            return {
                "intercept": mu,
                "slope": sd,
                "r_squared": float("nan"),
                "std_error": float("nan"),
                "residuals": np.array([]),
                "fallback": True,
            }

        p_uncens = plotting_positions[uncens_mask]
        x_uncens = observed_values[uncens_mask]

        if self.distribution == "lognormal":
            # Защита от нулей
            if np.any(x_uncens <= 0):
                raise ValueError(
                    "Для lognormal ROS все нецензурированные значения должны быть > 0."
                )
            y = np.log(x_uncens)
        else:
            y = x_uncens.copy()

        # Теоретические квантили
        z = stats.norm.ppf(p_uncens)

        # Проверка на вырожденный случай (все значения одинаковы)
        if np.std(y) < 1e-12:
            self.logger.warning(
                "Все нецензурированные значения одинаковы — slope=0."
            )
            return {
                "intercept": float(np.mean(y)),
                "slope": 0.0,
                "r_squared": float("nan"),
                "std_error": 0.0,
                "residuals": np.zeros_like(y),
                "fallback": False,
            }

        X = z.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Стандартная ошибка оценки
        dof = max(n_uncens - 2, 1)
        std_error = float(np.sqrt(ss_res / dof))

        intercept = float(model.intercept_)
        slope = float(model.coef_[0])

        if not np.isnan(r_squared) and r_squared < self.R2_WARN_THRESHOLD:
            self.logger.warning(
                f"Низкое R² = {r_squared:.3f} (< {self.R2_WARN_THRESHOLD}). "
                f"Распределение, возможно, выбрано некорректно."
            )

        return {
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_squared,
            "std_error": std_error,
            "residuals": residuals,
            "fallback": False,
        }

    def _extrapolate_censored(
        self,
        regression_params: Dict[str, Any],
        plotting_positions: np.ndarray,
        censored_flags: np.ndarray,
        lod_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Экстраполирует значения для цензурированных наблюдений по
        подогнанной регрессионной прямой.

        Parameters
        ----------
        regression_params : dict
            Результат _fit_regression.
        plotting_positions : np.ndarray
            Полный массив plotting positions.
        censored_flags : np.ndarray[bool]
            Флаги цензурирования.
        lod_values : np.ndarray, optional
            Значения LOD для каждого наблюдения (если переменные LOD).

        Returns
        -------
        np.ndarray
            Массив импутированных значений длиной равной числу
            цензурированных наблюдений.
        """
        a = regression_params["intercept"]
        b = regression_params["slope"]

        cens_idx = np.where(censored_flags)[0]
        if len(cens_idx) == 0:
            return np.array([])

        p_cens = plotting_positions[cens_idx]
        z_cens = stats.norm.ppf(p_cens)
        y_imp = a + b * z_cens

        if self.distribution == "lognormal":
            x_imp = np.exp(y_imp)
        else:
            x_imp = y_imp

        # Защита: значения должны быть > 0 и < LOD
        x_imp = np.where(x_imp <= 0, 1e-12, x_imp)

        if lod_values is not None:
            lod_cens = lod_values[cens_idx]
            over_mask = x_imp > lod_cens
            if np.any(over_mask):
                self.logger.debug(
                    f"{np.sum(over_mask)} импутированных значений > LOD, "
                    f"обрезаются до 0.99·LOD."
                )
                x_imp[over_mask] = lod_cens[over_mask] * 0.99

        return x_imp

    # ------------------------------------------------------------------ #
    # Публичный интерфейс
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_censored_data(censored_data: Dict[str, Any]):
        """
        Унифицированный парсинг входных данных.

        Ожидаемый формат censored_data:
            {
              "values":   array-like,   # все наблюдения; цензурированные
                                         # хранятся как значение LOD
              "censored": array-like[bool],
              "lod":      float или array-like (optional)
            }

        Returns
        -------
        values : np.ndarray
        censored_flags : np.ndarray[bool]
        lod_values : np.ndarray  (длиной n)
        """
        values = np.asarray(censored_data["values"], dtype=float)
        censored_flags = np.asarray(censored_data["censored"], dtype=bool)

        if "lod" in censored_data and censored_data["lod"] is not None:
            lod = censored_data["lod"]
            if np.isscalar(lod):
                lod_values = np.full_like(values, float(lod), dtype=float)
            else:
                lod_values = np.asarray(lod, dtype=float)
        else:
            # Если LOD явно не задан — берём значения в точках цензурирования.
            lod_values = values.copy()

        if values.shape != censored_flags.shape:
            raise ValueError("values и censored должны иметь одинаковую длину.")

        return values, censored_flags, lod_values

    def fit(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод: применяет полный алгоритм ROS.

        Шаги:
          a) сортировка данных,
          b) вычисление plotting positions,
          c) подгонка регрессии по нецензурированным,
          d) экстраполяция цензурированных,
          e) объединение в полный «заполненный» набор,
          f) вычисление всех статистик.

        Parameters
        ----------
        censored_data : dict
            См. _parse_censored_data.

        Returns
        -------
        dict
            См. раздел docstring модуля.
        """
        values, censored_flags, lod_values = self._parse_censored_data(censored_data)
        n_total = len(values)

        if n_total == 0:
            raise ValueError("Пустой входной массив.")

        # (a) Сортировка — цензурированные занимают нижние ранги.
        # Сортируем сначала по "эффективному" значению; цензурированные
        # считаем ≤ LOD, помещая их в начало при равенстве.
        effective = np.where(censored_flags, lod_values - 1e-12, values)
        order = np.argsort(effective, kind="mergesort")
        values_sorted = values[order]
        cens_sorted = censored_flags[order]
        lod_sorted = lod_values[order]

        # (b) Plotting positions
        pp = self._compute_plotting_positions(n_total, cens_sorted, method="blom")

        # (c) Регрессия
        reg = self._fit_regression(pp, values_sorted, cens_sorted)

        # (d) Экстраполяция цензурированных
        imputed_cens = self._extrapolate_censored(
            reg, pp, cens_sorted, lod_values=lod_sorted
        )

        # (e) Полный заполненный набор
        imputed_values_sorted = values_sorted.copy()
        imputed_values_sorted[cens_sorted] = imputed_cens

        # Возвращаем в исходный порядок
        inverse_order = np.argsort(order)
        imputed_full = imputed_values_sorted[inverse_order]

        # (f) Статистики
        est_mean = float(np.mean(imputed_full))
        est_var = float(np.var(imputed_full, ddof=1)) if n_total > 1 else 0.0
        est_std = float(np.sqrt(est_var))
        est_median = float(np.median(imputed_full))
        percentiles = {
            p: float(np.percentile(imputed_full, p)) for p in (75, 90, 95, 99)
        }

        # Оценка вероятности превышения заданного порога (используем макс. LOD
        # как контекстный порог; пользователь может пересчитать сам).
        threshold = float(np.max(lod_values))
        if self.distribution == "lognormal" and reg["slope"] > 0:
            mu, sigma = reg["intercept"], reg["slope"]
            exceed_prob = float(
                1.0 - stats.norm.cdf((np.log(threshold) - mu) / sigma)
            )
        elif self.distribution == "normal" and reg["slope"] > 0:
            mu, sigma = reg["intercept"], reg["slope"]
            exceed_prob = float(1.0 - stats.norm.cdf((threshold - mu) / sigma))
        else:
            exceed_prob = float(np.mean(imputed_full > threshold))

        # Данные для probability plot (в исходном порядке сортировки)
        prob_plot_data = {
            "sorted_values": values_sorted,
            "imputed_sorted": imputed_values_sorted,
            "plotting_positions": pp,
            "theoretical_quantiles": stats.norm.ppf(pp),
            "censored_mask_sorted": cens_sorted,
            "lod_sorted": lod_sorted,
        }

        return {
            "method_name": f"ros_{self.distribution}",
            "imputed_values": imputed_full,
            "regression_params": reg,
            "plotting_positions": pp,
            "estimated_mean": est_mean,
            "estimated_variance": est_var,
            "estimated_std": est_std,
            "estimated_median": est_median,
            "estimated_percentiles": percentiles,
            "estimated_exceedance_prob": exceed_prob,
            "r_squared": reg["r_squared"],
            "probability_plot_data": prob_plot_data,
        }

    def fit_with_variable_lod(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ROS для данных с несколькими порогами LOD (multiple detection limits).

        Реализация: адаптация метода Helsel (2005). Для каждой уникальной
        LOD оценивается доля наблюдений ≤ LOD, затем plotting positions
        цензурированных перераспределяются по стратам.

        Parameters
        ----------
        censored_data : dict
            Должен содержать массив "lod" длиной n с индивидуальными LOD.
        """
        values, censored_flags, lod_values = self._parse_censored_data(censored_data)
        n_total = len(values)

        if n_total == 0:
            raise ValueError("Пустой входной массив.")

        # Сортировка
        effective = np.where(censored_flags, lod_values - 1e-12, values)
        order = np.argsort(effective, kind="mergesort")
        values_sorted = values[order]
        cens_sorted = censored_flags[order]
        lod_sorted = lod_values[order]

        # Расчёт модифицированных plotting positions по методу Helsel-Cohn
        # для multiple detection limits.
        pp = self._multi_lod_plotting_positions(
            values_sorted, cens_sorted, lod_sorted
        )

        reg = self._fit_regression(pp, values_sorted, cens_sorted)
        imputed_cens = self._extrapolate_censored(
            reg, pp, cens_sorted, lod_values=lod_sorted
        )

        imputed_values_sorted = values_sorted.copy()
        imputed_values_sorted[cens_sorted] = imputed_cens
        inverse_order = np.argsort(order)
        imputed_full = imputed_values_sorted[inverse_order]

        est_mean = float(np.mean(imputed_full))
        est_var = float(np.var(imputed_full, ddof=1)) if n_total > 1 else 0.0
        est_std = float(np.sqrt(est_var))
        est_median = float(np.median(imputed_full))
        percentiles = {
            p: float(np.percentile(imputed_full, p)) for p in (75, 90, 95, 99)
        }

        threshold = float(np.max(lod_values))
        if self.distribution == "lognormal" and reg["slope"] > 0:
            mu, sigma = reg["intercept"], reg["slope"]
            exceed_prob = float(
                1.0 - stats.norm.cdf((np.log(threshold) - mu) / sigma)
            )
        elif self.distribution == "normal" and reg["slope"] > 0:
            mu, sigma = reg["intercept"], reg["slope"]
            exceed_prob = float(1.0 - stats.norm.cdf((threshold - mu) / sigma))
        else:
            exceed_prob = float(np.mean(imputed_full > threshold))

        prob_plot_data = {
            "sorted_values": values_sorted,
            "imputed_sorted": imputed_values_sorted,
            "plotting_positions": pp,
            "theoretical_quantiles": stats.norm.ppf(pp),
            "censored_mask_sorted": cens_sorted,
            "lod_sorted": lod_sorted,
        }

        return {
            "method_name": f"ros_{self.distribution}_multi_lod",
            "imputed_values": imputed_full,
            "regression_params": reg,
            "plotting_positions": pp,
            "estimated_mean": est_mean,
            "estimated_variance": est_var,
            "estimated_std": est_std,
            "estimated_median": est_median,
            "estimated_percentiles": percentiles,
            "estimated_exceedance_prob": exceed_prob,
            "r_squared": reg["r_squared"],
            "probability_plot_data": prob_plot_data,
        }

    # ------------------------------------------------------------------ #
    # Multi-LOD plotting positions (Helsel-Cohn)
    # ------------------------------------------------------------------ #
    def _multi_lod_plotting_positions(
        self,
        values_sorted: np.ndarray,
        cens_sorted: np.ndarray,
        lod_sorted: np.ndarray,
    ) -> np.ndarray:
        """
        Вычисление plotting positions по методу Helsel-Cohn для данных
        с несколькими detection limits.

        Алгоритм (упрощённо):
          1. Определить уникальные пороги LOD (включая +∞).
          2. Для каждой «страты» (интервала между соседними LOD)
             вычислить вероятность попадания A_j.
          3. Распределить plotting positions внутри страты.
        """
        n = len(values_sorted)
        unique_lods = np.sort(np.unique(lod_sorted))
        # Добавим бесконечность как верхнюю границу
        thresholds = np.concatenate([unique_lods, [np.inf]])

        pp = np.zeros(n, dtype=float)

        # Для каждой страты j: интервал (thresholds[j-1], thresholds[j]]
        # (используя соглашение: страта 0 — ≤ thresholds[0])
        # Подсчитаем:
        #   B_j — число нецензурированных в страте j
        #   C_j — число цензурированных с LOD = thresholds[j]
        # и вероятность A_j методом Helsel-Cohn.
        A = np.zeros(len(thresholds))
        # Снизу вверх
        above = 0  # число наблюдений с значением > thresholds[j]
        # Проходим страты сверху вниз
        probs_above = 0.0
        # Для вычисления используем классический алгоритм ROS multi-LOD.
        # Упрощённая версия: присваиваем каждому цензурированному наблюдению
        # ранг внутри подмножества значений ≤ его LOD.

        # Нецензурированные — обычные Blom-позиции на фоне всех данных
        uncens_mask = ~cens_sorted
        n_uncens = int(np.sum(uncens_mask))
        if n_uncens > 0:
            u_ranks = np.arange(1, n_uncens + 1, dtype=float)
            pp[uncens_mask] = (u_ranks - 0.375) / (n + 0.25)

        # Цензурированные — позиции в (0, F(LOD_i))
        cens_idx = np.where(cens_sorted)[0]
        for i, idx in enumerate(cens_idx):
            lod_i = lod_sorted[idx]
            # Доля наблюдений (всех) ≤ lod_i
            f_lod_i = float(np.sum(values_sorted <= lod_i)) / n
            # Ранг данного цензурированного среди цензурированных с LOD ≤ lod_i
            same_or_lower = np.sum(
                cens_sorted & (lod_sorted <= lod_i) & (np.arange(n) <= idx)
            )
            n_same = max(int(np.sum(cens_sorted & (lod_sorted <= lod_i))), 1)
            pp[idx] = (same_or_lower - 0.375) / (n_same + 0.25) * f_lod_i

        eps = 1e-6
        pp = np.clip(pp, eps, 1 - eps)
        return pp

    # ------------------------------------------------------------------ #
    # Визуализация
    # ------------------------------------------------------------------ #
    def plot_probability_plot(
        self,
        censored_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> matplotlib.figure.Figure:
        """
        Строит вероятностный график (Q-Q plot) для результатов ROS.

        - Синие кружки: нецензурированные наблюдения.
        - Красные треугольники: цензурированные (экстраполированные).
        - Чёрная линия: подогнанная регрессия.
        - Пунктирная горизонтальная линия: LOD.
        """
        _, _, lod_values = self._parse_censored_data(censored_data)
        pp_data = result["probability_plot_data"]
        reg = result["regression_params"]

        z = pp_data["theoretical_quantiles"]
        cens_mask = pp_data["censored_mask_sorted"]
        values_sorted = pp_data["sorted_values"]
        imputed_sorted = pp_data["imputed_sorted"]

        if self.distribution == "lognormal":
            y_obs = np.log(np.clip(values_sorted, 1e-12, None))
            y_imp = np.log(np.clip(imputed_sorted, 1e-12, None))
            ylabel = "ln(значение)"
            lod_line = np.log(np.max(lod_values))
        else:
            y_obs = values_sorted
            y_imp = imputed_sorted
            ylabel = "значение"
            lod_line = float(np.max(lod_values))

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            z[~cens_mask],
            y_obs[~cens_mask],
            marker="o",
            color="tab:blue",
            label="Нецензурированные",
            zorder=3,
        )
        ax.scatter(
            z[cens_mask],
            y_imp[cens_mask],
            marker="^",
            color="tab:red",
            label="Цензурированные (ROS)",
            zorder=3,
        )

        # Линия регрессии
        z_line = np.linspace(z.min(), z.max(), 100)
        y_line = reg["intercept"] + reg["slope"] * z_line
        r2 = reg.get("r_squared", float("nan"))
        ax.plot(
            z_line,
            y_line,
            color="black",
            linewidth=2,
            label=f"Регрессия (R² = {r2:.3f})"
            if not np.isnan(r2)
            else "Регрессия",
        )

        ax.axhline(
            lod_line,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"LOD = {np.max(lod_values):g}",
        )

        ax.set_xlabel("Теоретический квантиль Φ⁻¹(p)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Probability plot — ROS ({self.distribution})"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Пример использования
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Генерация синтетических данных ---
    rng = np.random.default_rng(42)
    true_mu, true_sigma = 1.0, 0.7
    n_samples = 100
    true_samples = rng.lognormal(mean=true_mu, sigma=true_sigma, size=n_samples)

    LOD = 1.5
    censored_flags = true_samples < LOD
    observed = np.where(censored_flags, LOD, true_samples)

    print("=" * 60)
    print("Пример применения ROSMethod")
    print("=" * 60)
    print(f"n = {n_samples}, цензурировано: {int(censored_flags.sum())}")
    print(f"Истинные параметры: μ_log={true_mu}, σ_log={true_sigma}")
    print(f"LOD = {LOD}")

    data = {
        "values": observed,
        "censored": censored_flags,
        "lod": LOD,
    }

    ros = ROSMethod(distribution="lognormal")
    result = ros.fit(data)

    print("\n--- Результаты ROS (lognormal) ---")
    print(f"Оцен. mean    : {result['estimated_mean']:.4f}")
    print(f"Оцен. std     : {result['estimated_std']:.4f}")
    print(f"Оцен. median  : {result['estimated_median']:.4f}")
    print(f"R²            : {result['r_squared']:.4f}")
    print(f"μ_log (intercept): {result['regression_params']['intercept']:.4f}")
    print(f"σ_log (slope)    : {result['regression_params']['slope']:.4f}")
    print(f"Процентили       : {result['estimated_percentiles']}")

    print(f"\nДля сравнения, истинные mean = "
          f"{np.exp(true_mu + true_sigma**2 / 2):.4f}, "
          f"true std ≈ "
          f"{np.sqrt((np.exp(true_sigma**2)-1)*np.exp(2*true_mu+true_sigma**2)):.4f}")

    # Пример multi-LOD
    lods = np.where(rng.random(n_samples) < 0.5, 1.0, 2.0)
    censored_flags_m = true_samples < lods
    observed_m = np.where(censored_flags_m, lods, true_samples)
    data_m = {"values": observed_m, "censored": censored_flags_m, "lod": lods}

    result_m = ros.fit_with_variable_lod(data_m)
    print("\n--- Результаты ROS с variable LOD ---")
    print(f"Оцен. mean : {result_m['estimated_mean']:.4f}")
    print(f"R²         : {result_m['r_squared']:.4f}")

    # Визуализация
    try:
        fig = ros.plot_probability_plot(data, result)
        fig.savefig("ros_probability_plot.png", dpi=120)
        print("\nProbability plot сохранён в ros_probability_plot.png")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")