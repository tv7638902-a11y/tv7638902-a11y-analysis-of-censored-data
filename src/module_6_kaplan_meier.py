"""
Модуль реализации непараметрического метода Каплана-Мейера
для оценки статистик по цензурированным данным.

Метод Каплана-Мейера (KM) — непараметрический метод оценки функции
выживания/распределения без предположений о форме распределения.
Для левоцензурированных экологических данных применяется "flipping trick".

Ссылки:
    Helsel, D. R. (2005). Nondetects and Data Analysis: Statistics for
    Censored Environmental Data. Wiley-Interscience.

    Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from
    Incomplete Observations. JASA, 53(282), 457-481.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class KaplanMeierCensored:
    """
    Непараметрический оценщик Каплана-Мейера для левоцензурированных данных.

    Для левоцензурирования используется "flipping trick":
        x' = C - x, где C = max(x) + min(x)
    После "переворота" левое цензурирование становится правым, и
    применяется стандартный KM-алгоритм.
    """

    def __init__(self) -> None:
        """Инициализирует логгер."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Валидация входных данных
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_censored_data(censored_data: dict) -> tuple:
        """
        Извлекает и валидирует данные из словаря.

        Args:
            censored_data: словарь с ключами:
                - 'observed_values': np.ndarray наблюдаемых значений
                  (для цензурированных = LOD)
                - 'censored_flags': np.ndarray булевых флагов
                  (True = значение цензурировано)
                - 'lod_values': np.ndarray LOD для каждого наблюдения
                  (может быть None)

        Returns:
            Кортеж (observed, censored, lod)
        """
        observed = np.asarray(censored_data["observed_values"], dtype=float)
        censored = np.asarray(censored_data["censored_flags"], dtype=bool)
        lod = censored_data.get("lod_values", None)
        if lod is None:
            lod = observed.copy()
        else:
            lod = np.asarray(lod, dtype=float)

        if len(observed) != len(censored):
            raise ValueError(
                "Размеры observed_values и censored_flags не совпадают"
            )
        return observed, censored, lod

    # ------------------------------------------------------------------
    # Основной алгоритм KM с "flipping trick"
    # ------------------------------------------------------------------
    def _turnbull_left_censored(
        self,
        observed_values: np.ndarray,
        censored_flags: np.ndarray,
        lod_values: np.ndarray,
    ) -> dict:
        """
        Реализует оценку Каплана-Мейера для левоцензурированных данных.

        Используется "flipping trick":
            x_flipped = C - x
        После чего применяется стандартный KM (для правого цензурирования),
        и результат переворачивается обратно.

        Args:
            observed_values: наблюдаемые значения (для цензурированных = LOD)
            censored_flags: True если значение левоцензурировано
            lod_values: пороги LOD для каждого наблюдения

        Returns:
            Словарь с ключами:
                - "survival_function": pd.DataFrame (time, S, F, se)
                - "unique_times": np.ndarray
                - "n_at_risk": np.ndarray
                - "n_events": np.ndarray
                - "flip_constant": float (для обратного преобразования)
        """
        n = len(observed_values)
        if n == 0:
            self.logger.warning("Пустой набор данных")
            return self._empty_result()

        # Проверка: все ли данные цензурированы?
        if np.all(censored_flags):
            self.logger.warning(
                "Все данные цензурированы — KM не может оценить распределение"
            )
            return self._empty_result()

        # Проверка: нет цензурированных вообще — обычный ECDF
        all_values = observed_values.copy()

        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))
        # Константа для переворота
        flip_constant = max_val + min_val

        # Перевёрнутые значения
        flipped = flip_constant - observed_values
        # Левое цензурирование -> правое цензурирование
        # В терминах lifelines/survival: event_observed = NOT censored
        event_observed = ~censored_flags

        # Сортируем по возрастанию flipped
        order = np.argsort(flipped, kind="mergesort")
        t_sorted = flipped[order]
        e_sorted = event_observed[order].astype(int)

        # Уникальные моменты событий (включая цензурированные)
        unique_times = np.unique(t_sorted)

        n_at_risk_list = []
        n_events_list = []
        n_censored_list = []
        S_list = []
        var_sum_list = []  # сумма для Гринвуда

        S = 1.0
        var_sum = 0.0

        for t in unique_times:
            # Число "в группе риска" перед t
            at_risk = int(np.sum(t_sorted >= t))
            # События (нецензурированные) ровно в момент t
            mask_t = t_sorted == t
            d = int(np.sum(e_sorted[mask_t]))

            if at_risk > 0 and d > 0:
                factor = 1.0 - d / at_risk
                S *= factor
                if at_risk - d > 0:
                    var_sum += d / (at_risk * (at_risk - d))
                else:
                    var_sum += np.nan

            n_at_risk_list.append(at_risk)
            n_events_list.append(d)
            n_censored_list.append(int(np.sum(mask_t) - d))
            S_list.append(S)
            var_sum_list.append(var_sum)

        n_at_risk = np.array(n_at_risk_list)
        n_events = np.array(n_events_list)
        S_arr = np.array(S_list)
        var_sum_arr = np.array(var_sum_list)

        # Дисперсия Гринвуда для S(t) в "перевёрнутом" пространстве
        greenwood_var = (S_arr ** 2) * var_sum_arr
        se = np.sqrt(np.clip(greenwood_var, 0, None))

        # Теперь переворачиваем обратно: исходное время
        # В исходном пространстве: F_orig(t) = S_flipped(C - t)
        original_times = flip_constant - unique_times
        # Отсортируем по возрастанию исходного времени
        rev_order = np.argsort(original_times)
        original_times_sorted = original_times[rev_order]
        F_orig = S_arr[rev_order]       # S в перевёрнутых = F в исходных
        S_orig = 1.0 - F_orig
        se_orig = se[rev_order]
        n_at_risk_orig = n_at_risk[rev_order]
        n_events_orig = n_events[rev_order]

        surv_df = pd.DataFrame(
            {
                "time": original_times_sorted,
                "S": S_orig,
                "F": F_orig,
                "se": se_orig,
            }
        )

        return {
            "survival_function": surv_df,
            "unique_times": original_times_sorted,
            "n_at_risk": n_at_risk_orig,
            "n_events": n_events_orig,
            "flip_constant": flip_constant,
            "_greenwood_var": greenwood_var[rev_order],
        }

    @staticmethod
    def _empty_result() -> dict:
        """Возвращает "пустой" результат для случаев, когда KM не применим."""
        empty_df = pd.DataFrame(columns=["time", "S", "F", "se"])
        return {
            "survival_function": empty_df,
            "unique_times": np.array([]),
            "n_at_risk": np.array([]),
            "n_events": np.array([]),
            "flip_constant": np.nan,
            "_greenwood_var": np.array([]),
        }

    # ------------------------------------------------------------------
    # Дисперсия Гринвуда
    # ------------------------------------------------------------------
    def _greenwood_variance(self, survival_data: dict) -> np.ndarray:
        """
        Вычисляет дисперсию оценки S(t) по формуле Гринвуда.

        Var(Ŝ(t)) = Ŝ(t)² × Σ_{tᵢ≤t} dᵢ / (nᵢ × (nᵢ - dᵢ))

        Args:
            survival_data: результат _turnbull_left_censored

        Returns:
            np.ndarray — дисперсии в порядке unique_times
        """
        if "_greenwood_var" in survival_data:
            return survival_data["_greenwood_var"]
        n_risk = survival_data["n_at_risk"]
        n_evt = survival_data["n_events"]
        S = survival_data["survival_function"]["S"].values
        denom = n_risk * (n_risk - n_evt)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(denom > 0, n_evt / denom, 0.0)
        cum = np.cumsum(terms)
        return (S ** 2) * cum

    # ------------------------------------------------------------------
    # Доверительные интервалы
    # ------------------------------------------------------------------
    def _confidence_intervals_km(
        self, survival_data: dict, confidence_level: float = 0.95
    ) -> dict:
        """
        Вычисляет доверительные интервалы для S(t) методом log(-log).

        log(-log(S(t))) ± z × se(log(-log(S(t))))
        где se(log(-log(S))) = sqrt(Var(S)) / (S × |log(S)|)

        Args:
            survival_data: результат KM-оценки
            confidence_level: уровень доверия (по умолчанию 0.95)

        Returns:
            Словарь с ключами 'lower', 'upper', 'time'
        """
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(1.0 - alpha / 2.0)

        df = survival_data["survival_function"]
        if df.empty:
            return {
                "time": np.array([]),
                "lower_S": np.array([]),
                "upper_S": np.array([]),
                "lower_F": np.array([]),
                "upper_F": np.array([]),
            }

        S = df["S"].values
        se = df["se"].values
        t = df["time"].values

        # Защита от крайних значений
        S_safe = np.clip(S, 1e-10, 1.0 - 1e-10)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_log_S = np.log(-np.log(S_safe))
            se_loglog = se / (S_safe * np.abs(np.log(S_safe)))

        lower_loglog = log_log_S - z * se_loglog
        upper_loglog = log_log_S + z * se_loglog

        lower_S = np.exp(-np.exp(upper_loglog))
        upper_S = np.exp(-np.exp(lower_loglog))

        # Для крайних значений — подстановка
        lower_S = np.where(np.isnan(lower_S), S, lower_S)
        upper_S = np.where(np.isnan(upper_S), S, upper_S)

        lower_S = np.clip(lower_S, 0.0, 1.0)
        upper_S = np.clip(upper_S, 0.0, 1.0)

        return {
            "time": t,
            "lower_S": lower_S,
            "upper_S": upper_S,
            "lower_F": 1.0 - upper_S,
            "upper_F": 1.0 - lower_S,
        }

    # ------------------------------------------------------------------
    # Статистики на основе KM-оценки
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_statistics_from_km(
        surv_df: pd.DataFrame, mpc: Optional[float] = None
    ) -> dict:
        """
        Вычисляет описательные статистики из KM-функции распределения.

        Среднее: μ = Σ tᵢ × ΔF(tᵢ)
        Дисперсия: σ² = Σ (tᵢ - μ)² × ΔF(tᵢ)
        Медиана, перцентили: обратная функция F.

        Args:
            surv_df: DataFrame с колонками time, S, F
            mpc: предельно-допустимая концентрация

        Returns:
            Словарь статистик
        """
        if surv_df.empty:
            return {
                "estimated_mean": np.nan,
                "estimated_variance": np.nan,
                "estimated_std": np.nan,
                "estimated_median": np.nan,
                "estimated_percentiles": {
                    75: np.nan, 90: np.nan, 95: np.nan, 99: np.nan
                },
                "estimated_exceedance_prob": np.nan,
            }

        t = surv_df["time"].values
        F = surv_df["F"].values

        # ΔF для каждого скачка (F(t_i) - F(t_{i-1}))
        F_prev = np.concatenate(([0.0], F[:-1]))
        dF = F - F_prev

        # Нормировка в случае, если максимальный F < 1 (из-за цензурирования)
        total_mass = F[-1] if F[-1] > 0 else 1.0
        dF_norm = dF / total_mass

        mean = float(np.sum(t * dF_norm))
        variance = float(np.sum((t - mean) ** 2 * dF_norm))
        std = float(np.sqrt(variance)) if variance >= 0 else np.nan

        # Медиана и перцентили через линейную интерполяцию обратной F
        def _percentile(p: float) -> float:
            target = p / 100.0 * total_mass
            if target <= F[0]:
                return float(t[0])
            if target >= F[-1]:
                return float(t[-1])
            idx = np.searchsorted(F, target)
            # Линейная интерполяция между (t[idx-1], F[idx-1]) и (t[idx], F[idx])
            t0, t1 = t[idx - 1], t[idx]
            f0, f1 = F[idx - 1], F[idx]
            if f1 == f0:
                return float(t0)
            return float(t0 + (target - f0) * (t1 - t0) / (f1 - f0))

        median = _percentile(50.0)
        percentiles = {p: _percentile(p) for p in (75, 90, 95, 99)}

        # Вероятность превышения ПДК
        exceedance = np.nan
        if mpc is not None:
            if mpc <= t[0]:
                exceedance = 1.0
            elif mpc >= t[-1]:
                exceedance = float(1.0 - F[-1])
            else:
                idx = np.searchsorted(t, mpc)
                t0, t1 = t[idx - 1], t[idx]
                f0, f1 = F[idx - 1], F[idx]
                if t1 == t0:
                    F_mpc = f0
                else:
                    F_mpc = f0 + (mpc - t0) * (f1 - f0) / (t1 - t0)
                exceedance = float(1.0 - F_mpc)

        return {
            "estimated_mean": mean,
            "estimated_variance": variance,
            "estimated_std": std,
            "estimated_median": median,
            "estimated_percentiles": percentiles,
            "estimated_exceedance_prob": exceedance,
        }

    # ------------------------------------------------------------------
    # Основной метод fit
    # ------------------------------------------------------------------
    def fit(self, censored_data: dict) -> dict:
        """
        Основной метод — оценивает статистики с помощью KM.

        Args:
            censored_data: словарь с данными (см. _validate_censored_data).
                Может содержать 'mpc' — ПДК для расчёта вероятности превышения.

        Returns:
            Словарь с оценками (см. docstring модуля).
        """
        observed, censored, lod = self._validate_censored_data(censored_data)
        mpc = censored_data.get("mpc", None)

        self.logger.info(
            "KM-оценка: n=%d, цензурировано=%d (%.1f%%)",
            len(observed),
            int(np.sum(censored)),
            100.0 * np.sum(censored) / len(observed) if len(observed) else 0.0,
        )

        km_data = self._turnbull_left_censored(observed, censored, lod)
        surv_df = km_data["survival_function"]

        stats_dict = self._compute_statistics_from_km(surv_df, mpc=mpc)
        ci = self._confidence_intervals_km(km_data, confidence_level=0.95)
        greenwood = self._greenwood_variance(km_data)

        return {
            "method_name": "kaplan_meier",
            **stats_dict,
            "survival_function": surv_df,
            "confidence_bands": ci,
            "greenwood_variance": greenwood,
            "n_at_risk": km_data["n_at_risk"],
            "n_events": km_data["n_events"],
        }

    # ------------------------------------------------------------------
    # Бутстрап
    # ------------------------------------------------------------------
    def bootstrap_km(
        self,
        censored_data: dict,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> dict:
        """
        Бутстрап для доверительных интервалов всех статистик.

        Args:
            censored_data: исходные данные
            n_bootstrap: число бутстрап-итераций
            confidence_level: уровень доверия
            random_state: seed для воспроизводимости

        Returns:
            Словарь с бутстрап-перцентилями для каждой статистики.
        """
        observed, censored, lod = self._validate_censored_data(censored_data)
        mpc = censored_data.get("mpc", None)
        n = len(observed)
        rng = np.random.default_rng(random_state)

        means, variances, medians, exceedances = [], [], [], []
        pct_dict = {75: [], 90: [], 95: [], 99: []}

        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_data = {
                "observed_values": observed[idx],
                "censored_flags": censored[idx],
                "lod_values": lod[idx],
                "mpc": mpc,
            }
            try:
                res = self.fit(boot_data)
                means.append(res["estimated_mean"])
                variances.append(res["estimated_variance"])
                medians.append(res["estimated_median"])
                for p in pct_dict:
                    pct_dict[p].append(res["estimated_percentiles"][p])
                exceedances.append(res["estimated_exceedance_prob"])
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Бутстрап-итерация %d пропущена: %s", b, exc)

        alpha = 1.0 - confidence_level
        q_low, q_high = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

        def _ci(arr):
            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return (np.nan, np.nan)
            return (float(np.percentile(arr, q_low)),
                    float(np.percentile(arr, q_high)))

        return {
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "mean_ci": _ci(means),
            "variance_ci": _ci(variances),
            "median_ci": _ci(medians),
            "percentiles_ci": {p: _ci(v) for p, v in pct_dict.items()},
            "exceedance_prob_ci": _ci(exceedances),
            "bootstrap_samples": {
                "mean": np.array(means),
                "variance": np.array(variances),
                "median": np.array(medians),
                "percentiles": {p: np.array(v) for p, v in pct_dict.items()},
                "exceedance_prob": np.array(exceedances),
            },
        }

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------
    def plot_survival_function(
        self,
        result: dict,
        mpc: Optional[float] = None,
        lod: Optional[float] = None,
        title: str = "",
        plot_cdf: bool = True,
    ) -> plt.Figure:
        """
        Строит график функции распределения / выживания.

        Args:
            result: результат fit()
            mpc: ПДК (вертикальная линия)
            lod: LOD (вертикальная линия)
            title: заголовок графика
            plot_cdf: True — рисовать F(t), False — S(t)

        Returns:
            matplotlib.figure.Figure
        """
        surv_df = result["survival_function"]
        ci = result.get("confidence_bands", {})

        fig, ax = plt.subplots(figsize=(9, 6))

        if surv_df.empty:
            ax.text(0.5, 0.5, "Нет данных для построения",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title or "KM-оценка")
            return fig

        t = surv_df["time"].values
        if plot_cdf:
            y = surv_df["F"].values
            ylabel = "F(t) — функция распределения"
            if ci:
                lower = ci["lower_F"]
                upper = ci["upper_F"]
        else:
            y = surv_df["S"].values
            ylabel = "S(t) — функция выживания"
            if ci:
                lower = ci["lower_S"]
                upper = ci["upper_S"]

        ax.step(t, y, where="post", color="C0", linewidth=2,
                label="KM-оценка")

        if ci and len(ci.get("time", [])) > 0:
            ax.fill_between(t, lower, upper, step="post",
                            alpha=0.25, color="C0", label="95% ДИ")

        if lod is not None:
            ax.axvline(lod, color="orange", linestyle="--",
                       label=f"LOD = {lod:g}")
        if mpc is not None:
            ax.axvline(mpc, color="red", linestyle="--",
                       label=f"ПДК = {mpc:g}")

        # Горизонтальные линии перцентилей
        if plot_cdf:
            for q, color in [(0.5, "grey"), (0.95, "grey")]:
                ax.axhline(q, color=color, linestyle=":", alpha=0.6)

        ax.set_xlabel("Значение концентрации")
        ax.set_ylabel(ylabel)
        ax.set_title(title or "Каплан-Мейер: оценка распределения")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Сравнение с lifelines (опционально)
    # ------------------------------------------------------------------
    def compare_with_lifelines(self, censored_data: dict) -> dict:
        """
        Сравнивает собственную реализацию с библиотекой lifelines.

        Args:
            censored_data: входные данные

        Returns:
            Словарь со сравнением оценок.
        """
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            self.logger.warning("lifelines не установлен — сравнение невозможно")
            return {"available": False}

        observed, censored, _ = self._validate_censored_data(censored_data)
        min_val = float(np.min(observed))
        max_val = float(np.max(observed))
        flip_c = max_val + min_val
        flipped = flip_c - observed
        event = (~censored).astype(int)

        kmf = KaplanMeierFitter()
        kmf.fit(flipped, event_observed=event)
        life_sf = kmf.survival_function_

        own = self.fit(censored_data)
        own_df = own["survival_function"]

        return {
            "available": True,
            "own_survival_function": own_df,
            "lifelines_survival_function_flipped": life_sf,
            "lifelines_median_survival_time_flipped": float(
                kmf.median_survival_time_
            ),
            "flip_constant": flip_c,
            "own_mean": own["estimated_mean"],
            "own_median": own["estimated_median"],
        }


# ----------------------------------------------------------------------
# Демонстрация
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    # Истинные данные: логнормальное распределение
    true_data = rng.lognormal(mean=1.0, sigma=0.6, size=200)

    # Левое цензурирование: несколько порогов LOD
    lod1, lod2 = 1.5, 2.0
    lod_assigned = rng.choice([lod1, lod2], size=len(true_data))
    censored_flags = true_data < lod_assigned
    observed_values = np.where(censored_flags, lod_assigned, true_data)

    print(f"Всего наблюдений: {len(true_data)}")
    print(f"Цензурировано: {int(np.sum(censored_flags))} "
          f"({100 * np.mean(censored_flags):.1f}%)")

    data = {
        "observed_values": observed_values,
        "censored_flags": censored_flags,
        "lod_values": lod_assigned,
        "mpc": 5.0,
    }

    km = KaplanMeierCensored()
    result = km.fit(data)

    print("\n=== Результаты KM ===")
    print(f"Среднее:   {result['estimated_mean']:.4f}")
    print(f"Std:       {result['estimated_std']:.4f}")
    print(f"Медиана:   {result['estimated_median']:.4f}")
    print(f"95-й pct:  {result['estimated_percentiles'][95]:.4f}")
    print(f"P(X>ПДК):  {result['estimated_exceedance_prob']:.4f}")
    print(f"\nИстинное среднее:  {np.mean(true_data):.4f}")
    print(f"Истинная медиана:  {np.median(true_data):.4f}")
    print(f"Истинный 95-й pct: {np.percentile(true_data, 95):.4f}")

    # Бутстрап
    print("\n=== Бутстрап-ДИ (n=200) ===")
    boot = km.bootstrap_km(data, n_bootstrap=200, random_state=0)
    print(f"Mean CI:   {boot['mean_ci']}")
    print(f"Median CI: {boot['median_ci']}")
    print(f"95-pct CI: {boot['percentiles_ci'][95]}")

    # Сравнение с lifelines
    cmp = km.compare_with_lifelines(data)
    if cmp.get("available"):
        print("\nСравнение с lifelines: OK")
    else:
        print("\nlifelines недоступен")

    # График
    fig = km.plot_survival_function(
        result, mpc=5.0, lod=lod2, title="Демо: KM для левоцензурированных данных"
    )
    out = "km_demo.png"
    fig.savefig(out, dpi=120)
    print(f"\nГрафик сохранён: {out}")