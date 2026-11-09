"""
src/module_3_substitution.py

Модуль 3: Простые методы подстановки для обработки цензурированных данных.

Реализует baseline-методы замены значений ниже предела обнаружения (LOD):
    - подстановка 0
    - подстановка LOD
    - подстановка LOD/2
    - подстановка LOD/sqrt(2)
    - подстановка LOD * fraction (кастом)

Литература:
    - Helsel, D.R. (2005). Nondetects and Data Analysis: Statistics for
      Censored Environmental Data. Wiley.
    - Hornung, R.W., Reed, L.D. (1990). Estimation of average concentration
      in the presence of nondetectable values. Appl. Occup. Environ. Hyg.
      5(1): 46-51.
    - US EPA (2000). Guidance for Data Quality Assessment, EPA/600/R-96/084.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class SubstitutionMethods:
    """
    Класс, объединяющий простые методы подстановки для цензурированных данных.

    Все методы принимают словарь censored_data со структурой:
        {
            "observed_values": np.ndarray,  # наблюдаемые значения (для цензурированных обычно LOD)
            "censored_flags": np.ndarray(bool),  # True == цензурировано (< LOD)
            "lod_value": float или np.ndarray,  # фиксированный LOD или массив LOD
            "true_values": np.ndarray (optional)  # истинные значения (для синтетики)
            "mpc": float (optional)  # ПДК (MPC)
        }

    Общее ограничение методов подстановки:
        - Все методы занижают дисперсию (искусственно уменьшают разброс).
        - substitute_zero занижает среднее, substitute_lod — завышает.
        - substitute_half_lod и substitute_lod_sqrt2 — компромисс.
        - При доле цензурирования > 50% все методы подстановки сильно смещают оценки
          (см. Helsel, 2005). В таких случаях рекомендуются KM, ROS или MLE.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        self.available_methods = {
            "substitution_zero": self.substitute_zero,
            "substitution_lod": self.substitute_lod,
            "substitution_half_lod": self.substitute_half_lod,
            "substitution_lod_sqrt2": self.substitute_lod_sqrt2,
            "substitution_custom": self.substitute_custom,
        }

        self.logger.info(
            f"SubstitutionMethods initialized with {len(self.available_methods)} methods."
        )

    # ------------------------------------------------------------------ #
    # Вспомогательные методы
    # ------------------------------------------------------------------ #
    def _extract_arrays(self, censored_data: Dict[str, Any]):
        """Аккуратно извлекает observed, flags, lod из входного словаря."""
        observed = np.asarray(censored_data.get("observed_values", []), dtype=float)
        flags = np.asarray(censored_data.get("censored_flags", []), dtype=bool)

        if observed.size != flags.size:
            raise ValueError(
                f"Размерности observed_values ({observed.size}) и "
                f"censored_flags ({flags.size}) не совпадают."
            )

        lod = censored_data.get("lod_value", None)
        if lod is None:
            raise ValueError("В censored_data отсутствует 'lod_value'.")

        if np.isscalar(lod):
            lod_arr = np.full(observed.shape, float(lod))
        else:
            lod_arr = np.asarray(lod, dtype=float)
            if lod_arr.size == 1:
                lod_arr = np.full(observed.shape, float(lod_arr.item()))
            elif lod_arr.size != observed.size:
                raise ValueError(
                    "lod_value должен быть скаляром или массивом длины observed_values."
                )

        return observed, flags, lod_arr

    def _compute_confidence_intervals(
        self, values: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, tuple]:
        """
        Доверительные интервалы для среднего (t-распределение)
        и дисперсии (хи-квадрат) в предположении нормальности.

        Для цензурированных данных ДИ приближённые, так как подстановка
        искажает выборку. Используются как справочные.
        """
        n = values.size
        if n < 2:
            return {
                "mean_95ci": (float("nan"), float("nan")),
                "variance_95ci": (float("nan"), float("nan")),
            }

        mean = float(np.mean(values))
        var = float(np.var(values, ddof=1))
        std = float(np.sqrt(var))
        alpha = 1.0 - confidence_level

        # CI для среднего — t-распределение
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        se = std / np.sqrt(n)
        mean_ci = (mean - t_crit * se, mean + t_crit * se)

        # CI для дисперсии — chi-square
        chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
        if chi2_lower > 0 and chi2_upper > 0:
            var_ci = (
                (n - 1) * var / chi2_upper,
                (n - 1) * var / chi2_lower,
            )
        else:
            var_ci = (float("nan"), float("nan"))

        return {"mean_95ci": mean_ci, "variance_95ci": var_ci}

    def _compute_statistics(
        self, values: np.ndarray, mpc: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Вычисление базовых статистик по заполненному массиву.
        """
        if values.size == 0:
            empty_result = {
                "estimated_mean": float("nan"),
                "estimated_variance": float("nan"),
                "estimated_std": float("nan"),
                "estimated_median": float("nan"),
                "estimated_percentiles": {
                    75: float("nan"),
                    90: float("nan"),
                    95: float("nan"),
                    99: float("nan"),
                },
                "estimated_exceedance_prob": float("nan"),
                "confidence_intervals": {
                    "mean_95ci": (float("nan"), float("nan")),
                    "variance_95ci": (float("nan"), float("nan")),
                },
            }
            return empty_result

        mean_val = float(np.mean(values))
        var_val = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        std_val = float(np.sqrt(var_val))
        median_val = float(np.median(values))

        percentiles = {
            int(p): float(np.percentile(values, p))
            for p in (75, 90, 95, 99)
        }

        exceedance = float("nan")
        if mpc is not None and values.size > 0:
            exceedance = float(np.mean(values > mpc))

        ci = self._compute_confidence_intervals(values)

        return {
            "estimated_mean": mean_val,
            "estimated_variance": var_val,
            "estimated_std": std_val,
            "estimated_median": median_val,
            "estimated_percentiles": percentiles,
            "estimated_exceedance_prob": exceedance,
            "confidence_intervals": ci,
        }

    def _substitute_with_factor(
        self,
        censored_data: Dict[str, Any],
        factor: float,
        method_name: str,
    ) -> Dict[str, Any]:
        """
        Общая реализация подстановки: цензурированные значения заменяются на LOD * factor.
        При variable LOD каждое цензурированное наблюдение получает свой LOD * factor.
        """
        observed, flags, lod_arr = self._extract_arrays(censored_data)

        imputed = observed.copy()
        if observed.size > 0:
            sub_values = lod_arr * factor
            imputed[flags] = sub_values[flags]

        n_substituted = int(np.sum(flags))

        # Для отчёта в поле substitution_value:
        # если LOD варьируется, возвращаем "представительное" значение
        if np.unique(lod_arr).size == 1 and lod_arr.size > 0:
            substitution_value = float(lod_arr[0] * factor)
        else:
            substitution_value = float(np.mean(lod_arr) * factor) if lod_arr.size else float("nan")

        mpc = censored_data.get("mpc", None)
        stats_dict = self._compute_statistics(imputed, mpc=mpc)

        result = {
            "method_name": method_name,
            "imputed_values": imputed,
            "substitution_value": substitution_value,
            "n_substituted": n_substituted,
            **stats_dict,
        }

        self.logger.debug(
            f"[{method_name}] n={observed.size}, n_substituted={n_substituted}, "
            f"factor={factor}, mean={stats_dict['estimated_mean']:.4g}"
        )

        return result

    # ------------------------------------------------------------------ #
    # Методы подстановки
    # ------------------------------------------------------------------ #
    def substitute_zero(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подстановка нулями: X_cens -> 0.

        Допущения:
            - Истинные значения считаются пренебрежимо малыми.
        Проблемы:
            - Систематически занижает среднее.
            - Занижает дисперсию (искусственная масса в точке 0).
            - Нереалистичен для логнормальных концентраций.
        Когда применять:
            - Как нижняя граница (consersative lower bound).
            - Доля цензурирования очень мала (< 5%).
        Литература:
            - Helsel (2005), глава 6.
        """
        return self._substitute_with_factor(
            censored_data, factor=0.0, method_name="substitution_zero"
        )

    def substitute_lod(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подстановка значением LOD: X_cens -> LOD.

        Допущения:
            - Все цензурированные значения равны пределу обнаружения.
        Свойства:
            - Завышает среднее по сравнению с истинным.
            - Используется как консервативная (worst-case) оценка
              для регуляторных задач, оценок риска.
        Литература:
            - US EPA (2000), Guidance for Data Quality Assessment.
        """
        return self._substitute_with_factor(
            censored_data, factor=1.0, method_name="substitution_lod"
        )

    def substitute_half_lod(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подстановка LOD/2.

        Допущения:
            - Истинные значения распределены равномерно на [0, LOD],
              тогда среднее равно LOD/2.
        Свойства:
            - Наиболее популярный метод на практике.
            - Относительно малое смещение при < 15-20% цензурирования.
            - При сильно скошенных (логнормальных) распределениях завышает
              среднее по цензурированным наблюдениям.
        Литература:
            - US EPA guidance; Helsel (2005).
        """
        return self._substitute_with_factor(
            censored_data, factor=0.5, method_name="substitution_half_lod"
        )

    def substitute_lod_sqrt2(self, censored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подстановка LOD/sqrt(2) ≈ 0.7071 * LOD.

        Допущения:
            - Истинные значения распределены логнормально.
            - Малая доля цензурирования (< 15-20%).
        Свойства:
            - Hornung & Reed (1990) показали, что для логнормальных данных
              LOD/sqrt(2) минимизирует смещение среднего при малой доле цензурирования.
            - При сильно скошенных и/или сильно цензурированных данных метод
              также смещён; предпочтительнее MLE/ROS.
        Литература:
            - Hornung, R.W. and Reed, L.D. (1990).
        """
        factor = 1.0 / np.sqrt(2.0)
        return self._substitute_with_factor(
            censored_data, factor=factor, method_name="substitution_lod_sqrt2"
        )

    def substitute_custom(
        self, censored_data: Dict[str, Any], fraction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Подстановка LOD * fraction, где fraction ∈ (0, 1].

        Обобщение всех предыдущих методов, позволяет пользователю задать
        собственную долю (например, 0.3, 0.7 и т.д.).

        Параметры:
            fraction: доля от LOD, которой заменяются цензурированные значения.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                f"fraction должен быть в диапазоне (0, 1], получено {fraction}."
            )
        method_name = f"substitution_custom_{fraction:g}"
        return self._substitute_with_factor(
            censored_data, factor=float(fraction), method_name=method_name
        )

    # ------------------------------------------------------------------ #
    # Сравнительный анализ
    # ------------------------------------------------------------------ #
    def apply_all_substitution_methods(
        self,
        censored_data: Dict[str, Any],
        mpc: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Применяет все методы подстановки и возвращает сравнительную таблицу.

        Если задан mpc — он прокидывается в censored_data для расчёта P(X > MPC).
        Если в censored_data есть true_values, добавляется строка с истинными
        статистиками для сравнения.
        """
        local_data = dict(censored_data)  # копия верхнего уровня
        if mpc is not None:
            local_data["mpc"] = mpc

        rows = []

        try:
            observed, flags, _ = self._extract_arrays(local_data)
            total_n = observed.size
            censored_rate = float(np.mean(flags)) if total_n > 0 else float("nan")
        except Exception as e:
            self.logger.warning(f"Не удалось извлечь массивы: {e}")
            total_n = 0
            censored_rate = float("nan")

        # Основные методы (без кастомного — там нужен fraction)
        method_calls = [
            ("substitution_zero", self.substitute_zero),
            ("substitution_lod", self.substitute_lod),
            ("substitution_half_lod", self.substitute_half_lod),
            ("substitution_lod_sqrt2", self.substitute_lod_sqrt2),
        ]

        for name, func in method_calls:
            try:
                res = func(local_data)
                rows.append(self._result_to_row(res, mpc=mpc))
            except Exception as e:
                self.logger.error(f"Ошибка в методе {name}: {e}")

        # Если переданы true_values — добавим строку с «идеальной» оценкой
        true_values = local_data.get("true_values", None)
        if true_values is not None:
            true_values = np.asarray(true_values, dtype=float)
            if true_values.size > 0:
                true_stats = self._compute_statistics(true_values, mpc=mpc)
                rows.append(
                    {
                        "method_name": "true_values",
                        "substitution_value": float("nan"),
                        "estimated_mean": true_stats["estimated_mean"],
                        "estimated_variance": true_stats["estimated_variance"],
                        "estimated_std": true_stats["estimated_std"],
                        "estimated_p90": true_stats["estimated_percentiles"][90],
                        "estimated_p95": true_stats["estimated_percentiles"][95],
                        "estimated_exceedance_prob": true_stats["estimated_exceedance_prob"],
                    }
                )

        df = pd.DataFrame(rows)
        self.logger.info(
            f"apply_all_substitution_methods: n={total_n}, "
            f"censored_rate={censored_rate:.2%}, methods={len(rows)}"
        )
        return df

    @staticmethod
    def _result_to_row(result: Dict[str, Any], mpc: Optional[float]) -> Dict[str, Any]:
        """Преобразует результат одного метода в строку DataFrame."""
        perc = result.get("estimated_percentiles", {})
        return {
            "method_name": result["method_name"],
            "substitution_value": result.get("substitution_value", float("nan")),
            "estimated_mean": result.get("estimated_mean", float("nan")),
            "estimated_variance": result.get("estimated_variance", float("nan")),
            "estimated_std": result.get("estimated_std", float("nan")),
            "estimated_p90": perc.get(90, float("nan")),
            "estimated_p95": perc.get(95, float("nan")),
            "estimated_exceedance_prob": result.get(
                "estimated_exceedance_prob", float("nan")
            ),
        }


# ---------------------------------------------------------------------- #
# Пример использования
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)

    # Генерация синтетических данных: логнормальное распределение
    true_values = rng.lognormal(mean=0.0, sigma=1.0, size=500)
    LOD = 0.5
    MPC = 3.0

    censored_flags = true_values < LOD
    observed_values = np.where(censored_flags, LOD, true_values)

    censored_data = {
        "observed_values": observed_values,
        "censored_flags": censored_flags,
        "lod_value": LOD,
        "true_values": true_values,
    }

    print(f"\nДоля цензурированных: {censored_flags.mean():.2%}")
    print(f"Истинное среднее: {true_values.mean():.4f}")
    print(f"Истинная дисперсия: {true_values.var(ddof=1):.4f}\n")

    sm = SubstitutionMethods()

    # Отдельный вызов
    res_half = sm.substitute_half_lod(censored_data)
    print("=== Пример одиночного вызова: substitute_half_lod ===")
    print(f"  mean = {res_half['estimated_mean']:.4f}")
    print(f"  var  = {res_half['estimated_variance']:.4f}")
    print(f"  CI mean = {res_half['confidence_intervals']['mean_95ci']}")
    print(f"  n_substituted = {res_half['n_substituted']}\n")

    # Сравнение всех методов
    df = sm.apply_all_substitution_methods(censored_data, mpc=MPC)
    print("=== Сравнительная таблица методов подстановки ===")
    print(df.to_string(index=False))

    # Пример variable LOD
    print("\n=== Пример с переменным LOD ===")
    variable_lod = rng.uniform(0.3, 0.7, size=true_values.size)
    v_flags = true_values < variable_lod
    v_obs = np.where(v_flags, variable_lod, true_values)
    v_data = {
        "observed_values": v_obs,
        "censored_flags": v_flags,
        "lod_value": variable_lod,
        "true_values": true_values,
    }
    df2 = sm.apply_all_substitution_methods(v_data, mpc=MPC)
    print(df2.to_string(index=False))

    # Кастомный fraction
    print("\n=== Кастомная подстановка (fraction=0.3) ===")
    res_custom = sm.substitute_custom(censored_data, fraction=0.3)
    print(f"  method = {res_custom['method_name']}")
    print(f"  mean   = {res_custom['estimated_mean']:.4f}")
    print(f"  substitution_value = {res_custom['substitution_value']}")