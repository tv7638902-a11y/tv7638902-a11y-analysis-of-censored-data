"""
src/module_9_evaluation.py

КЛЮЧЕВОЙ модуль для сравнения всех методов обработки цензурированных данных.

Этот модуль является центральным в проекте. Он:
1. Берёт истинные (полные) данные с известными статистиками
2. Создаёт цензурированные версии (через модуль 2)
3. Применяет все методы обработки (модули 3-8)
4. Вычисляет метрики качества для каждого метода
5. Создаёт сравнительные таблицы и отчёты

Автор: Project Censored Data Analysis
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# Настройка логирования
# ============================================================================
def _setup_logger(name: str = "MethodEvaluator", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ============================================================================
# Основной класс
# ============================================================================
class MethodEvaluator:
    """
    Класс для оценки и сравнения методов обработки цензурированных данных.

    Поддерживаемые методы:
      - substitution_zero, substitution_half, substitution_lod
      - mle_normal, mle_lognormal
      - km (Kaplan-Meier), ros (Regression on Order Statistics)
      - multiple_imputation
      - bayesian

    Поддерживаемые метрики:
      - Смещение (абсолютное, относительное)
      - RMSE
      - Coverage probability
      - Статистики распределений (среднее, медиана, дисперсия, перцентили)
      - Вероятность превышения ПДК
      - KS и Cramer-von Mises для сравнения распределений
    """

    # Список известных методов — для маршрутизации
    SUPPORTED_METHODS = [
        "substitution_zero",
        "substitution_half",
        "substitution_lod",
        "mle_normal",
        "mle_lognormal",
        "km",
        "ros",
        "multiple_imputation",
        "bayesian",
    ]

    # Ключевые статистики, по которым оцениваем качество восстановления
    STAT_KEYS = ["mean", "median", "std", "variance", "p90", "p95"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        config : dict, optional
            Конфигурация. Поддерживаемые ключи:
              - 'mpc' : dict {metal_name: value} — ПДК для металлов
              - 'log_level' : int — уровень логирования
              - 'random_seed' : int — зерно для воспроизводимости
        """
        self.config = config or {}
        self.logger = _setup_logger(
            "MethodEvaluator",
            level=self.config.get("log_level", logging.INFO),
        )

        # ПДК (предельно допустимые концентрации) для металлов
        self.mpc_dict: Dict[str, float] = self.config.get("mpc", {})

        # Зерно для воспроизводимости
        self.random_seed = self.config.get("random_seed", 42)

        self.logger.info("MethodEvaluator initialised")
        if self.mpc_dict:
            self.logger.info(f"Loaded MPC values for metals: {list(self.mpc_dict.keys())}")

    # ------------------------------------------------------------------
    # 2. Смещение
    # ------------------------------------------------------------------
    def compute_bias(self, true_value: float, estimated_value: float) -> Dict[str, float]:
        """
        Вычисляет смещение оценки.

        Returns
        -------
        dict :
          - absolute_bias : estimated - true
          - relative_bias_percent : 100 * (estimated - true) / true
          - absolute_relative_bias : 100 * |estimated - true| / true
        """
        if true_value is None or estimated_value is None:
            return {
                "absolute_bias": np.nan,
                "relative_bias_percent": np.nan,
                "absolute_relative_bias": np.nan,
            }
        if np.isnan(true_value) or np.isnan(estimated_value):
            return {
                "absolute_bias": np.nan,
                "relative_bias_percent": np.nan,
                "absolute_relative_bias": np.nan,
            }

        abs_bias = estimated_value - true_value
        if abs(true_value) > 1e-12:
            rel_bias = 100.0 * abs_bias / true_value
            abs_rel_bias = 100.0 * abs(abs_bias) / abs(true_value)
        else:
            rel_bias = np.nan
            abs_rel_bias = np.nan

        return {
            "absolute_bias": abs_bias,
            "relative_bias_percent": rel_bias,
            "absolute_relative_bias": abs_rel_bias,
        }

    # ------------------------------------------------------------------
    # 3. RMSE
    # ------------------------------------------------------------------
    def compute_rmse(self, true_value: float, estimates: np.ndarray) -> float:
        """
        RMSE по множеству повторений (Монте-Карло).
        RMSE = sqrt(mean((estimates - true)^2))
        """
        estimates = np.asarray(estimates, dtype=float)
        estimates = estimates[~np.isnan(estimates)]
        if len(estimates) == 0 or true_value is None or np.isnan(true_value):
            return np.nan
        return float(np.sqrt(np.mean((estimates - true_value) ** 2)))

    # ------------------------------------------------------------------
    # 4. Coverage probability
    # ------------------------------------------------------------------
    def compute_coverage_probability(
        self,
        true_value: float,
        confidence_intervals: List[Tuple[float, float]],
    ) -> float:
        """
        Доля ДИ, покрывающих истинное значение.
        Идеальное значение: 0.95 для 95% ДИ.
        """
        if not confidence_intervals or true_value is None or np.isnan(true_value):
            return np.nan

        valid_ci = [
            (lo, hi) for (lo, hi) in confidence_intervals
            if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi))
        ]
        if not valid_ci:
            return np.nan

        covered = sum(1 for (lo, hi) in valid_ci if lo <= true_value <= hi)
        return covered / len(valid_ci)

    # ------------------------------------------------------------------
    # 5. Оценка одного метода
    # ------------------------------------------------------------------
    def evaluate_single_method(
        self,
        true_stats: Dict[str, Any],
        method_result: Dict[str, Any],
        mpc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает один метод по всем метрикам для одного эксперимента.

        Parameters
        ----------
        true_stats : dict
            Истинные значения: mean, median, std, variance, p90, p95,
            а также 'data' (истинные данные) и опционально
            'exceedance_prob' (истинная вероятность превышения ПДК).
        method_result : dict
            Результат работы метода: ключи — названия статистик,
            плюс 'imputed_data' (опционально) и 'ci_<stat>' (опционально).
        mpc : float, optional
            Предельно допустимая концентрация.

        Returns
        -------
        dict :
            Словарь со всеми метриками.
        """
        metrics: Dict[str, Any] = {}

        # (a-e) Статистики распределения: смещение, направление
        for stat in self.STAT_KEYS:
            true_val = true_stats.get(stat, np.nan)
            est_val = method_result.get(stat, np.nan)

            bias_info = self.compute_bias(true_val, est_val)
            metrics[f"{stat}_true"] = true_val
            metrics[f"{stat}_estimated"] = est_val
            metrics[f"{stat}_abs_bias"] = bias_info["absolute_bias"]
            metrics[f"{stat}_rel_bias_pct"] = bias_info["relative_bias_percent"]
            metrics[f"{stat}_abs_rel_bias"] = bias_info["absolute_relative_bias"]

            # Направление смещения
            if not np.isnan(bias_info["absolute_bias"]):
                if bias_info["absolute_bias"] > 0:
                    metrics[f"{stat}_direction"] = "overestimate"
                elif bias_info["absolute_bias"] < 0:
                    metrics[f"{stat}_direction"] = "underestimate"
                else:
                    metrics[f"{stat}_direction"] = "exact"
            else:
                metrics[f"{stat}_direction"] = "n/a"

        # (f) Вероятность превышения ПДК
        if mpc is not None:
            true_exceed = true_stats.get("exceedance_prob", None)
            if true_exceed is None and "data" in true_stats:
                true_exceed = self._compute_exceedance_probability(
                    np.asarray(true_stats["data"]), mpc
                )

            est_exceed = method_result.get("exceedance_prob", None)
            if est_exceed is None and "imputed_data" in method_result:
                est_exceed = self._compute_exceedance_probability(
                    np.asarray(method_result["imputed_data"]), mpc
                )

            metrics["exceedance_true"] = true_exceed
            metrics["exceedance_estimated"] = est_exceed

            if true_exceed is not None and est_exceed is not None \
                    and not (np.isnan(true_exceed) or np.isnan(est_exceed)):
                abs_err = abs(est_exceed - true_exceed)
                rel_err_pp = (est_exceed - true_exceed) * 100.0  # в процентных пунктах
                metrics["absolute_error_exceedance"] = abs_err
                metrics["relative_error_exceedance_pp"] = rel_err_pp

                if est_exceed > true_exceed:
                    metrics["exceedance_direction"] = "conservative"  # завышает риск
                elif est_exceed < true_exceed:
                    metrics["exceedance_direction"] = "anti-conservative"  # занижает риск
                else:
                    metrics["exceedance_direction"] = "exact"
            else:
                metrics["absolute_error_exceedance"] = np.nan
                metrics["relative_error_exceedance_pp"] = np.nan
                metrics["exceedance_direction"] = "n/a"

        # (g) Восстановление распределения
        true_data = true_stats.get("data", None)
        imputed_data = method_result.get("imputed_data", None)

        if true_data is not None and imputed_data is not None:
            try:
                true_arr = np.asarray(true_data, dtype=float)
                imp_arr = np.asarray(imputed_data, dtype=float)
                true_arr = true_arr[~np.isnan(true_arr)]
                imp_arr = imp_arr[~np.isnan(imp_arr)]

                if len(true_arr) > 1 and len(imp_arr) > 1:
                    ks_stat, ks_p = stats.ks_2samp(true_arr, imp_arr)
                    metrics["ks_statistic"] = float(ks_stat)
                    metrics["ks_p_value"] = float(ks_p)

                    # Крамер-фон Мизес
                    try:
                        cvm_res = stats.cramervonmises_2samp(true_arr, imp_arr)
                        metrics["cramer_von_mises"] = float(cvm_res.statistic)
                    except Exception:
                        metrics["cramer_von_mises"] = np.nan
                else:
                    metrics["ks_statistic"] = np.nan
                    metrics["ks_p_value"] = np.nan
                    metrics["cramer_von_mises"] = np.nan
            except Exception as e:
                self.logger.debug(f"Distribution comparison failed: {e}")
                metrics["ks_statistic"] = np.nan
                metrics["ks_p_value"] = np.nan
                metrics["cramer_von_mises"] = np.nan
        else:
            metrics["ks_statistic"] = np.nan
            metrics["ks_p_value"] = np.nan
            metrics["cramer_von_mises"] = np.nan

        # (h) Доверительные интервалы (если доступны)
        for stat in self.STAT_KEYS:
            ci_key = f"ci_{stat}"
            if ci_key in method_result:
                ci = method_result[ci_key]
                if ci is not None and len(ci) == 2:
                    lo, hi = ci
                    true_val = true_stats.get(stat, np.nan)
                    if lo is not None and hi is not None and not np.isnan(true_val):
                        metrics[f"{stat}_ci_contains_true"] = bool(lo <= true_val <= hi)
                        metrics[f"{stat}_ci_width"] = float(hi - lo)
                    else:
                        metrics[f"{stat}_ci_contains_true"] = None
                        metrics[f"{stat}_ci_width"] = np.nan

        # Признак успешности метода
        metrics["converged"] = bool(method_result.get("converged", True))

        return metrics

    # ------------------------------------------------------------------
    # 6. Сравнение всех методов
    # ------------------------------------------------------------------
    def evaluate_all_methods(
        self,
        true_stats: Dict[str, Any],
        methods_results: Dict[str, Dict[str, Any]],
        mpc: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Оценивает все методы и создаёт сравнительную таблицу с ранжированием.
        """
        rows = []
        for method_name, result in methods_results.items():
            if result is None:
                self.logger.warning(f"Method '{method_name}' returned None, skipping")
                continue
            metrics = self.evaluate_single_method(true_stats, result, mpc=mpc)
            metrics["method"] = method_name
            rows.append(metrics)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("method")

        # Ранжирование по ключевым метрикам (меньше = лучше)
        rank_metrics = []
        for stat in self.STAT_KEYS:
            col = f"{stat}_abs_rel_bias"
            if col in df.columns:
                rank_metrics.append(col)
        if "absolute_error_exceedance" in df.columns:
            rank_metrics.append("absolute_error_exceedance")
        if "ks_statistic" in df.columns:
            rank_metrics.append("ks_statistic")

        for col in rank_metrics:
            df[f"{col}_rank"] = df[col].rank(method="min", na_option="bottom")

        rank_cols = [f"{c}_rank" for c in rank_metrics]
        if rank_cols:
            df["overall_rank_score"] = df[rank_cols].mean(axis=1)
            df = df.sort_values("overall_rank_score")
            df["overall_rank"] = df["overall_rank_score"].rank(method="min")

        return df

    # ------------------------------------------------------------------
    # 7. Полное сравнение
    # ------------------------------------------------------------------
    def run_full_comparison(
        self,
        data: Dict[str, Any],
        censoring_rates: List[float],
        methods: List[str],
        n_simulations: int = 100,
        mpc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        ПОЛНОЕ СРАВНЕНИЕ — запускает все эксперименты.

        Parameters
        ----------
        data : dict
            Должен содержать:
              - 'values' : np.ndarray — истинные полные данные
              - 'metal'  : str (опционально) — название металла
        censoring_rates : list of float
            Доли цензурирования, напр. [0.1, 0.2, ..., 0.8]
        methods : list of str
            Названия методов для сравнения.
        n_simulations : int
            Число Монте-Карло повторений.
        mpc : float, optional
            Предельно допустимая концентрация.

        Returns
        -------
        dict со следующими полями:
            summary_by_censoring_rate, summary_by_method,
            overall_ranking, best_method_by_metric_and_rate,
            raw_results, metadata
        """
        true_values = np.asarray(data["values"], dtype=float)
        true_values = true_values[~np.isnan(true_values)]
        metal_name = data.get("metal", "unknown")

        if mpc is None and metal_name in self.mpc_dict:
            mpc = self.mpc_dict[metal_name]

        self.logger.info(
            f"Starting full comparison: metal={metal_name}, n={len(true_values)}, "
            f"methods={methods}, censoring_rates={censoring_rates}, "
            f"n_simulations={n_simulations}, mpc={mpc}"
        )

        # Истинные статистики
        true_stats = self._compute_true_stats(true_values, mpc=mpc)

        # Хранилище для сырых результатов
        raw_results: Dict[float, Dict[str, List[Dict[str, Any]]]] = {
            rate: {m: [] for m in methods} for rate in censoring_rates
        }

        rng = np.random.default_rng(self.random_seed)

        # Основной цикл
        total_iters = len(censoring_rates) * n_simulations
        with tqdm(total=total_iters, desc="Simulations") as pbar:
            for rate in censoring_rates:
                for sim_idx in range(n_simulations):
                    sim_seed = int(rng.integers(0, 2**31 - 1))
                    try:
                        censored = self._apply_censoring(true_values, rate, seed=sim_seed)
                    except Exception as e:
                        self.logger.error(f"Censoring failed at rate={rate}: {e}")
                        pbar.update(1)
                        continue

                    for method_name in methods:
                        try:
                            method_res = self._apply_method(
                                method_name, censored, mpc=mpc
                            )
                        except Exception as e:
                            self.logger.debug(
                                f"Method '{method_name}' failed at rate={rate}, "
                                f"sim={sim_idx}: {e}"
                            )
                            method_res = self._nan_result()

                        metrics = self.evaluate_single_method(
                            true_stats, method_res, mpc=mpc
                        )
                        metrics["simulation"] = sim_idx
                        metrics["censoring_rate"] = rate
                        raw_results[rate][method_name].append(metrics)

                    pbar.update(1)

        # Агрегация: summary_by_censoring_rate
        summary_by_rate: Dict[float, pd.DataFrame] = {}
        for rate in censoring_rates:
            agg_rows = []
            for method_name in methods:
                sims = raw_results[rate][method_name]
                if not sims:
                    continue
                df_sims = pd.DataFrame(sims)
                agg = self._aggregate_simulations(df_sims, true_stats, method_name)
                agg_rows.append(agg)
            if agg_rows:
                df_rate = pd.DataFrame(agg_rows).set_index("method")
                # Общий ранг
                df_rate = self._add_overall_rank(df_rate)
                summary_by_rate[rate] = df_rate

        # summary_by_method: поведение каждого метода при разных rates
        summary_by_method: Dict[str, pd.DataFrame] = {}
        for method_name in methods:
            rows = []
            for rate in censoring_rates:
                sims = raw_results[rate][method_name]
                if not sims:
                    continue
                df_sims = pd.DataFrame(sims)
                agg = self._aggregate_simulations(df_sims, true_stats, method_name)
                agg["censoring_rate"] = rate
                rows.append(agg)
            if rows:
                df_m = pd.DataFrame(rows).set_index("censoring_rate")
                summary_by_method[method_name] = df_m

        # Общий ранг по всему эксперименту
        overall_ranking = self._compute_overall_ranking(summary_by_rate)

        # Лучший метод для каждой метрики × ставки
        best_method = self._compute_best_method_table(summary_by_rate)

        results = {
            "summary_by_censoring_rate": summary_by_rate,
            "summary_by_method": summary_by_method,
            "overall_ranking": overall_ranking,
            "best_method_by_metric_and_rate": best_method,
            "raw_results": raw_results,
            "metadata": {
                "n_simulations": n_simulations,
                "censoring_rates": list(censoring_rates),
                "methods": list(methods),
                "sample_size": int(len(true_values)),
                "metal": metal_name,
                "mpc": mpc,
                "random_seed": self.random_seed,
            },
        }

        self.logger.info("Full comparison completed")
        return results

    # ------------------------------------------------------------------
    # 8. Создание Markdown-отчёта
    # ------------------------------------------------------------------
    def create_comparison_report(self, full_results: Dict[str, Any]) -> str:
        """Создаёт текстовый отчёт в Markdown."""
        md: List[str] = []
        meta = full_results.get("metadata", {})

        md.append("# Отчёт о сравнении методов обработки цензурированных данных\n")
        md.append("## Описание эксперимента\n")
        md.append(f"- **Металл:** {meta.get('metal', 'n/a')}")
        md.append(f"- **Размер выборки:** {meta.get('sample_size', 'n/a')}")
        md.append(f"- **Число симуляций:** {meta.get('n_simulations', 'n/a')}")
        md.append(f"- **Доли цензурирования:** {meta.get('censoring_rates', [])}")
        md.append(f"- **Исследуемые методы:** {meta.get('methods', [])}")
        md.append(f"- **ПДК:** {meta.get('mpc', 'не задано')}")
        md.append(f"- **Random seed:** {meta.get('random_seed', 'n/a')}\n")

        # Лучшие методы
        md.append("## Лучший метод для каждой метрики и доли цензурирования\n")
        best = full_results.get("best_method_by_metric_and_rate")
        if best is not None and not best.empty:
            md.append(best.to_markdown())
            md.append("")

        # Общий рейтинг
        md.append("## Общий рейтинг методов\n")
        overall = full_results.get("overall_ranking")
        if overall is not None and not overall.empty:
            md.append(overall.to_markdown())
            md.append("")

        # Таблицы по каждой доле цензурирования
        md.append("## Детальные результаты по долям цензурирования\n")
        for rate, df in full_results.get("summary_by_censoring_rate", {}).items():
            md.append(f"### Доля цензурирования = {rate:.0%}\n")
            cols_to_show = [c for c in df.columns if "rank" not in c][:15]
            md.append(df[cols_to_show].round(4).to_markdown())
            md.append("")

        # Рекомендации
        md.append("## Рекомендации\n")
        md.append(self._generate_recommendations(full_results))

        # Выводы
        md.append("\n## Выводы\n")
        if overall is not None and not overall.empty:
            best_overall = overall.index[0]
            md.append(
                f"- **Лучший метод в среднем:** `{best_overall}` "
                f"(средний ранг = {overall.iloc[0, 0]:.2f})"
            )
            worst_overall = overall.index[-1]
            md.append(f"- **Худший метод в среднем:** `{worst_overall}`")

        md.append(
            "- Для высоких уровней цензурирования (>50%) рекомендуется использовать "
            "MLE или ROS методы вместо простой подстановки."
        )
        md.append(
            "- Простая подстановка (zero/half/lod) быстра, но даёт значительное смещение."
        )

        return "\n".join(md)

    # ------------------------------------------------------------------
    # 9. Вспомогательные методы
    # ------------------------------------------------------------------
    def _apply_method(
        self,
        method_name: str,
        censored_data: Dict[str, Any],
        mpc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Маршрутизация: по имени метода вызывает соответствующий модуль.
        При отсутствии модуля — пытается работать корректно с заглушкой.
        """
        values = np.asarray(censored_data["values"], dtype=float)
        is_censored = np.asarray(censored_data["is_censored"], dtype=bool)
        detection_limits = np.asarray(
            censored_data.get("detection_limits", np.full_like(values, np.nan)),
            dtype=float,
        )

        try:
            if method_name.startswith("substitution"):
                return self._dispatch_substitution(method_name, values, is_censored, detection_limits, mpc)
            elif method_name.startswith("mle"):
                return self._dispatch_mle(method_name, values, is_censored, detection_limits, mpc)
            elif method_name == "km":
                return self._dispatch_km(values, is_censored, detection_limits, mpc)
            elif method_name == "ros":
                return self._dispatch_ros(values, is_censored, detection_limits, mpc)
            elif method_name == "multiple_imputation":
                return self._dispatch_mi(values, is_censored, detection_limits, mpc)
            elif method_name == "bayesian":
                return self._dispatch_bayesian(values, is_censored, detection_limits, mpc)
            else:
                raise ValueError(f"Unknown method: {method_name}")
        except Exception as e:
            self.logger.debug(f"Method {method_name} dispatch failed: {e}")
            return self._nan_result()

    # ---- диспетчеры к внешним модулям (с graceful fallback) ----
    def _dispatch_substitution(self, method_name, values, is_censored, dl, mpc):
        try:
            from module_3_substitution import SubstitutionMethods  # type: ignore
            sub = SubstitutionMethods()
            kind = method_name.replace("substitution_", "")
            return sub.apply(values, is_censored, dl, method=kind, mpc=mpc)
        except Exception:
            # Fallback: простая реализация
            imputed = values.copy()
            if method_name == "substitution_zero":
                imputed[is_censored] = 0.0
            elif method_name == "substitution_half":
                imputed[is_censored] = dl[is_censored] / 2.0
            elif method_name == "substitution_lod":
                imputed[is_censored] = dl[is_censored]
            return self._stats_from_data(imputed, mpc)

    def _dispatch_mle(self, method_name, values, is_censored, dl, mpc):
        try:
            if method_name == "mle_normal":
                from module_4_mle import MLENormal  # type: ignore
                return MLENormal().fit(values, is_censored, dl, mpc=mpc)
            else:
                from module_4_mle import MLELogNormal  # type: ignore
                return MLELogNormal().fit(values, is_censored, dl, mpc=mpc)
        except Exception:
            return self._nan_result()

    def _dispatch_km(self, values, is_censored, dl, mpc):
        try:
            from module_5_km import KaplanMeier  # type: ignore
            return KaplanMeier().fit(values, is_censored, dl, mpc=mpc)
        except Exception:
            return self._nan_result()

    def _dispatch_ros(self, values, is_censored, dl, mpc):
        try:
            from module_6_ros import ROS  # type: ignore
            return ROS().fit(values, is_censored, dl, mpc=mpc)
        except Exception:
            return self._nan_result()

    def _dispatch_mi(self, values, is_censored, dl, mpc):
        try:
            from module_7_mi import MultipleImputation  # type: ignore
            return MultipleImputation().fit(values, is_censored, dl, mpc=mpc)
        except Exception:
            return self._nan_result()

    def _dispatch_bayesian(self, values, is_censored, dl, mpc):
        try:
            from module_8_bayesian import BayesianEstimator  # type: ignore
            return BayesianEstimator().fit(values, is_censored, dl, mpc=mpc)
        except Exception:
            return self._nan_result()

    # ---- Цензурирование (с graceful fallback) ----
    def _apply_censoring(
        self,
        true_values: np.ndarray,
        rate: float,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Создаёт цензурированный датасет (через модуль 2 или простой fallback)."""
        try:
            from module_2_censoring import Censoring  # type: ignore
            censor = Censoring()
            return censor.apply_left_censoring(true_values, rate=rate, seed=seed)
        except Exception:
            # Fallback: простое левое цензурирование по квантилю
            rng = np.random.default_rng(seed)
            lod = float(np.quantile(true_values, rate))
            is_censored = true_values < lod
            values = true_values.copy()
            values[is_censored] = lod  # обычно присваивают LOD
            detection_limits = np.full_like(true_values, lod)
            return {
                "values": values,
                "is_censored": is_censored,
                "detection_limits": detection_limits,
                "lod": lod,
                "rate_actual": float(is_censored.mean()),
            }

    def _compute_exceedance_probability(
        self, values: np.ndarray, threshold: float
    ) -> float:
        """Вычисляет долю значений, превышающих порог."""
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0 or threshold is None:
            return np.nan
        return float(np.mean(values > threshold))

    def _compute_true_stats(
        self, true_values: np.ndarray, mpc: Optional[float] = None
    ) -> Dict[str, Any]:
        """Вычисляет истинные статистики по полным данным."""
        stats_dict = {
            "mean": float(np.mean(true_values)),
            "median": float(np.median(true_values)),
            "std": float(np.std(true_values, ddof=1)),
            "variance": float(np.var(true_values, ddof=1)),
            "p90": float(np.percentile(true_values, 90)),
            "p95": float(np.percentile(true_values, 95)),
            "data": true_values,
        }
        if mpc is not None:
            stats_dict["exceedance_prob"] = self._compute_exceedance_probability(
                true_values, mpc
            )
        return stats_dict

    def _stats_from_data(
        self, data: np.ndarray, mpc: Optional[float] = None
    ) -> Dict[str, Any]:
        """Создаёт result-dict со статистиками из массива данных."""
        data = np.asarray(data, dtype=float)
        clean = data[~np.isnan(data)]
        if len(clean) == 0:
            return self._nan_result()
        res = {
            "mean": float(np.mean(clean)),
            "median": float(np.median(clean)),
            "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else np.nan,
            "variance": float(np.var(clean, ddof=1)) if len(clean) > 1 else np.nan,
            "p90": float(np.percentile(clean, 90)),
            "p95": float(np.percentile(clean, 95)),
            "imputed_data": data,
            "converged": True,
        }
        if mpc is not None:
            res["exceedance_prob"] = self._compute_exceedance_probability(data, mpc)
        return res

    def _nan_result(self) -> Dict[str, Any]:
        """Возвращает результат-заглушку с NaN для несошедшегося метода."""
        return {k: np.nan for k in self.STAT_KEYS} | {"converged": False}

    def _aggregate_simulations(
        self,
        df_sims: pd.DataFrame,
        true_stats: Dict[str, Any],
        method_name: str,
    ) -> Dict[str, Any]:
        """Агрегирует метрики по симуляциям."""
        agg: Dict[str, Any] = {"method": method_name}

        # По статистикам: среднее абс. отн. смещения, RMSE, покрытие ДИ
        for stat in self.STAT_KEYS:
            est_col = f"{stat}_estimated"
            abs_rel_col = f"{stat}_abs_rel_bias"
            if est_col in df_sims.columns:
                estimates = df_sims[est_col].values
                true_val = true_stats.get(stat, np.nan)
                agg[f"{stat}_rmse"] = self.compute_rmse(true_val, estimates)
                agg[f"{stat}_mean_abs_rel_bias"] = float(
                    np.nanmean(df_sims[abs_rel_col].values)
                )
                agg[f"{stat}_mean_estimate"] = float(np.nanmean(estimates))

                # Coverage probability
                ci_contains_col = f"{stat}_ci_contains_true"
                if ci_contains_col in df_sims.columns:
                    vals = df_sims[ci_contains_col].dropna()
                    if len(vals) > 0:
                        agg[f"{stat}_coverage"] = float(vals.mean())

        # Exceedance
        if "absolute_error_exceedance" in df_sims.columns:
            agg["mean_abs_error_exceedance"] = float(
                np.nanmean(df_sims["absolute_error_exceedance"].values)
            )

        # KS
        if "ks_statistic" in df_sims.columns:
            agg["mean_ks_statistic"] = float(np.nanmean(df_sims["ks_statistic"].values))

        # Convergence rate
        if "converged" in df_sims.columns:
            agg["convergence_rate"] = float(df_sims["converged"].mean())

        return agg

    def _add_overall_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет общий ранг на основе ключевых метрик."""
        rank_cols = [
            c for c in df.columns
            if c.endswith("_mean_abs_rel_bias") or c == "mean_abs_error_exceedance"
            or c == "mean_ks_statistic"
        ]
        if not rank_cols:
            return df

        rank_df = df[rank_cols].rank(method="min", na_option="bottom")
        df = df.copy()
        df["overall_rank_score"] = rank_df.mean(axis=1)
        df["overall_rank"] = df["overall_rank_score"].rank(method="min")
        df = df.sort_values("overall_rank_score")
        return df

    def _compute_overall_ranking(
        self, summary_by_rate: Dict[float, pd.DataFrame]
    ) -> pd.DataFrame:
        """Вычисляет общий рейтинг методов по всем долям цензурирования."""
        if not summary_by_rate:
            return pd.DataFrame()

        all_scores = []
        for rate, df in summary_by_rate.items():
            if "overall_rank_score" in df.columns:
                all_scores.append(df["overall_rank_score"].rename(rate))

        if not all_scores:
            return pd.DataFrame()

        combined = pd.concat(all_scores, axis=1)
        result = pd.DataFrame({
            "mean_rank_score": combined.mean(axis=1),
            "worst_rank_score": combined.max(axis=1),
            "best_rank_score": combined.min(axis=1),
        }).sort_values("mean_rank_score")
        return result

    def _compute_best_method_table(
        self, summary_by_rate: Dict[float, pd.DataFrame]
    ) -> pd.DataFrame:
        """Для каждой метрики и каждой доли — определяет лучший метод."""
        if not summary_by_rate:
            return pd.DataFrame()

        rows = []
        for rate, df in summary_by_rate.items():
            metric_cols = [
                c for c in df.columns
                if c.endswith("_mean_abs_rel_bias")
                or c == "mean_abs_error_exceedance"
                or c == "mean_ks_statistic"
            ]
            for col in metric_cols:
                if df[col].notna().any():
                    best = df[col].idxmin()
                    best_val = df[col].min()
                    rows.append({
                        "censoring_rate": rate,
                        "metric": col,
                        "best_method": best,
                        "best_value": best_val,
                    })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).pivot(
            index="metric", columns="censoring_rate", values="best_method"
        )

    def _generate_recommendations(self, full_results: Dict[str, Any]) -> str:
        """Генерирует текстовые рекомендации."""
        recs = []
        meta = full_results.get("metadata", {})
        rates = meta.get("censoring_rates", [])

        if not rates:
            return "- Недостаточно данных для рекомендаций.\n"

        low_rates = [r for r in rates if r <= 0.3]
        high_rates = [r for r in rates if r > 0.5]

        summary = full_results.get("summary_by_censoring_rate", {})

        if low_rates:
            best_low = None
            for r in low_rates:
                if r in summary and "overall_rank_score" in summary[r].columns:
                    best_low = summary[r].index[0]
                    break
            if best_low:
                recs.append(
                    f"- **При низкой цензуре (≤30%):** рекомендуется `{best_low}`"
                )

        if high_rates:
            best_high = None
            for r in high_rates:
                if r in summary and "overall_rank_score" in summary[r].columns:
                    best_high = summary[r].index[0]
                    break
            if best_high:
                recs.append(
                    f"- **При высокой цензуре (>50%):** рекомендуется `{best_high}`"
                )

        recs.append(
            "- Избегайте substitution_zero при расчёте вероятности превышения ПДК "
            "(сильно занижает риск)."
        )
        recs.append(
            "- Если нужны доверительные интервалы — используйте MLE или Bayesian методы."
        )

        return "\n".join(recs) + "\n"

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------
    def save_results(self, results: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Сохраняет результаты. Формат определяется по расширению:
          .pkl / .pickle -> pickle
          .xlsx          -> Excel (сохраняются только DataFrame)
          .csv           -> CSV (сохраняется overall_ranking)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix in {".pkl", ".pickle"}:
            with open(filepath, "wb") as f:
                pickle.dump(results, f)
            self.logger.info(f"Results saved (pickle) to {filepath}")

        elif filepath.suffix == ".xlsx":
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                if "overall_ranking" in results and isinstance(
                    results["overall_ranking"], pd.DataFrame
                ):
                    results["overall_ranking"].to_excel(writer, sheet_name="Overall")
                for rate, df in results.get("summary_by_censoring_rate", {}).items():
                    sheet = f"Rate_{int(rate*100)}"
                    df.to_excel(writer, sheet_name=sheet[:31])
                if "best_method_by_metric_and_rate" in results:
                    best = results["best_method_by_metric_and_rate"]
                    if isinstance(best, pd.DataFrame) and not best.empty:
                        best.to_excel(writer, sheet_name="BestMethods")
            self.logger.info(f"Results saved (Excel) to {filepath}")

        elif filepath.suffix == ".csv":
            if "overall_ranking" in results and isinstance(
                results["overall_ranking"], pd.DataFrame
            ):
                results["overall_ranking"].to_csv(filepath)
                self.logger.info(f"Overall ranking saved (CSV) to {filepath}")

        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")

    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Загружает ранее сохранённые результаты (pickle)."""
        filepath = Path(filepath)
        if filepath.suffix in {".pkl", ".pickle"}:
            with open(filepath, "rb") as f:
                results = pickle.load(f)
            self.logger.info(f"Results loaded from {filepath}")
            return results
        else:
            raise ValueError(f"Loading supported only for .pkl, got {filepath.suffix}")

    def export_to_latex(
        self, df: pd.DataFrame, filepath: Union[str, Path], caption: str = ""
    ) -> None:
        """Экспортирует DataFrame в LaTeX."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        latex = df.to_latex(float_format="%.4f", caption=caption, longtable=False)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(latex)
        self.logger.info(f"LaTeX table saved to {filepath}")


# ============================================================================
# Демонстрация
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ: module_9_evaluation.MethodEvaluator")
    print("=" * 70)

    # Конфигурация
    config = {
        "mpc": {"Pb": 0.03, "Cd": 0.005, "Zn": 1.0},
        "random_seed": 2024,
        "log_level": logging.INFO,
    }

    evaluator = MethodEvaluator(config=config)

    # Генерируем синтетические "истинные" данные (логнормальное распределение)
    rng = np.random.default_rng(2024)
    n = 200
    true_data = rng.lognormal(mean=np.log(0.02), sigma=0.6, size=n)

    data = {
        "values": true_data,
        "metal": "Pb",
    }

    print(f"\nСгенерирована выборка: n={n}")
    print(f"  mean = {true_data.mean():.4f}")
    print(f"  median = {np.median(true_data):.4f}")
    print(f"  std = {true_data.std():.4f}")

    # --- Демонстрация 1: compute_bias ---
    print("\n--- 1) compute_bias ---")
    bias = evaluator.compute_bias(true_value=0.02, estimated_value=0.023)
    for k, v in bias.items():
        print(f"  {k}: {v:.4f}")

    # --- Демонстрация 2: compute_rmse ---
    print("\n--- 2) compute_rmse ---")
    estimates = np.array([0.018, 0.022, 0.025, 0.019, 0.021])
    rmse = evaluator.compute_rmse(true_value=0.02, estimates=estimates)
    print(f"  RMSE = {rmse:.6f}")

    # --- Демонстрация 3: coverage_probability ---
    print("\n--- 3) compute_coverage_probability ---")
    cis = [(0.018, 0.022), (0.019, 0.023), (0.021, 0.025), (0.017, 0.021)]
    cov = evaluator.compute_coverage_probability(true_value=0.02, confidence_intervals=cis)
    print(f"  Coverage = {cov:.2%} (ожидается ~ покрытие для true=0.02)")

    # --- Демонстрация 4: evaluate_single_method ---
    print("\n--- 4) evaluate_single_method ---")
    true_stats = evaluator._compute_true_stats(true_data, mpc=0.03)
    fake_result = evaluator._stats_from_data(true_data * 1.1, mpc=0.03)  # +10% смещение
    metrics = evaluator.evaluate_single_method(true_stats, fake_result, mpc=0.03)
    print("  Ключевые метрики (искусственное +10% смещение):")
    for key in ["mean_rel_bias_pct", "median_rel_bias_pct",
                "absolute_error_exceedance", "ks_statistic"]:
        if key in metrics:
            val = metrics[key]
            print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

    # --- Демонстрация 5: run_full_comparison ---
    print("\n--- 5) run_full_comparison (упрощённая) ---")
    print("    (используются fallback-реализации для substitution)")

    results = evaluator.run_full_comparison(
        data=data,
        censoring_rates=[0.1, 0.3, 0.5],
        methods=["substitution_zero", "substitution_half", "substitution_lod"],
        n_simulations=20,
        mpc=0.03,
    )

    print("\n  Общий рейтинг методов:")
    print(results["overall_ranking"].round(3))

    print("\n  Таблица для rate=0.5 (первые 5 колонок):")
    df50 = results["summary_by_censoring_rate"][0.5]
    print(df50.iloc[:, :5].round(4))

    print("\n  Лучший метод по метрике и доле цензурирования:")
    print(results["best_method_by_metric_and_rate"])

    # --- Демонстрация 6: отчёт ---
    print("\n--- 6) create_comparison_report (фрагмент) ---")
    report = evaluator.create_comparison_report(results)
    print(report[:1500])
    print("... [отчёт обрезан]")

    # --- Демонстрация 7: сохранение ---
    print("\n--- 7) save_results ---")
    out_dir = Path("./_eval_demo_output")
    out_dir.mkdir(exist_ok=True)
    evaluator.save_results(results, out_dir / "results.pkl")
    try:
        evaluator.save_results(results, out_dir / "results.xlsx")
    except ImportError:
        print("  (openpyxl не установлен — пропускаем Excel)")
    evaluator.export_to_latex(
        results["overall_ranking"],
        out_dir / "overall_ranking.tex",
        caption="Общий рейтинг методов",
    )

    print(f"\nФайлы сохранены в: {out_dir.absolute()}")
    print("\n" + "=" * 70)
    print("Демонстрация завершена.")
    print("=" * 70)