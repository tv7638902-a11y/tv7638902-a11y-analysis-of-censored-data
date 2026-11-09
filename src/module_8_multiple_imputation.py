"""
Модуль множественной импутации (Multiple Imputation, MI) для цензурированных данных.

Реализует метод множественной импутации с объединением результатов по правилам Рубина
(Rubin's rules) для корректного учёта неопределённости, связанной с импутацией
значений ниже предела обнаружения (LOD).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MultipleImputation:
    """
    Множественная импутация для цензурированных данных.

    Для каждого цензурированного наблюдения генерируется M значений из
    усечённого распределения с параметрами, оценёнными с учётом неопределённости
    (параметрический бутстрап или апостериорное распределение). Затем результаты
    объединяются по правилам Рубина.
    """

    def __init__(
        self,
        n_imputations: int = 20,
        distribution: str = "lognormal",
        imputation_method: str = "parametric_bootstrap",
    ):
        """
        Параметры
        ---------
        n_imputations : int
            Количество импутаций M (рекомендуется 20-50).
        distribution : str
            Предполагаемое распределение данных: 'lognormal' или 'normal'.
        imputation_method : str
            Метод импутации:
            - 'parametric_bootstrap' — бутстрап + MLE + усечённое распределение,
            - 'bayesian' — семплирование из апостериорного распределения,
            - 'pmm' — Predictive Mean Matching.
        """
        self.n_imputations = int(n_imputations)
        self.distribution = distribution
        self.imputation_method = imputation_method
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"MultipleImputation initialized: M={self.n_imputations}, "
            f"dist={self.distribution}, method={self.imputation_method}"
        )

    # ------------------------------------------------------------------ #
    # Вспомогательные методы для работы с распределениями                 #
    # ------------------------------------------------------------------ #

    def _fit_mle(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Подгонка MLE параметров распределения по наблюдаемым данным.

        Для 'lognormal' возвращает (mu, sigma) логарифмированных данных.
        Для 'normal' возвращает (mu, sigma).
        """
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if self.distribution == "lognormal":
            data = data[data > 0]
            logs = np.log(data)
            mu = float(np.mean(logs))
            sigma = float(np.std(logs, ddof=1)) if len(logs) > 1 else 0.1
            sigma = max(sigma, 1e-6)
            return mu, sigma
        elif self.distribution == "normal":
            mu = float(np.mean(data))
            sigma = float(np.std(data, ddof=1)) if len(data) > 1 else 0.1
            sigma = max(sigma, 1e-6)
            return mu, sigma
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _sample_truncated(
        self, lod: float, mu: float, sigma: float, rng: np.random.Generator
    ) -> float:
        """
        Семплирование одного значения из усечённого распределения X < LOD.

        Использует метод обратной CDF:
            u ~ Uniform(0, F(LOD|θ))
            x = F⁻¹(u|θ)
        """
        if self.distribution == "lognormal":
            # Для lognormal параметризация: F(x) = Φ((log x - mu)/sigma)
            log_lod = np.log(lod)
            upper = stats.norm.cdf(log_lod, loc=mu, scale=sigma)
            upper = min(max(upper, 1e-12), 1 - 1e-12)
            u = rng.uniform(1e-12, upper)
            log_x = stats.norm.ppf(u, loc=mu, scale=sigma)
            x = float(np.exp(log_x))
            # Защита: должно быть 0 < x < LOD
            if not (0 < x < lod):
                x = lod * 0.5
            return x
        elif self.distribution == "normal":
            upper = stats.norm.cdf(lod, loc=mu, scale=sigma)
            upper = min(max(upper, 1e-12), 1 - 1e-12)
            u = rng.uniform(1e-12, upper)
            x = float(stats.norm.ppf(u, loc=mu, scale=sigma))
            if x >= lod:
                x = lod - 1e-9
            return x
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    # ------------------------------------------------------------------ #
    # Генерация одной импутации                                          #
    # ------------------------------------------------------------------ #

    def _generate_single_imputation_parametric(
        self, censored_data: dict, iteration: int
    ) -> np.ndarray:
        """
        Генерация одного заполненного датасета методом параметрического бутстрапа.

        Параметры
        ---------
        censored_data : dict
            Должен содержать ключи:
            - 'observed' : np.ndarray наблюдаемых (нецензурированных) значений,
            - 'censored_lods' : np.ndarray LOD-значений для каждого цензурированного,
            - (опционально) 'n_censored' : int.
        iteration : int
            Номер итерации (используется для seed).

        Возвращает
        ----------
        np.ndarray длины n_observed + n_censored с заполненными значениями.
        """
        rng = np.random.default_rng(seed=iteration * 1000 + 7)

        observed = np.asarray(censored_data["observed"], dtype=float)
        lods = np.asarray(censored_data["censored_lods"], dtype=float)
        n_censored = len(lods)

        if len(observed) < 2:
            raise ValueError("Недостаточно наблюдаемых значений для бутстрапа.")

        # a) Бутстрап-выборка из наблюдаемых данных
        boot_idx = rng.integers(0, len(observed), size=len(observed))
        boot_sample = observed[boot_idx]

        # b) MLE на бутстрап-выборке
        mu_boot, sigma_boot = self._fit_mle(boot_sample)

        # c) Генерация импутированных значений из усечённого распределения
        imputed = np.empty(n_censored)
        for i, lod_i in enumerate(lods):
            imputed[i] = self._sample_truncated(lod_i, mu_boot, sigma_boot, rng)

        # d) Объединение
        return np.concatenate([observed, imputed])

    def _generate_single_imputation_bayesian(
        self, censored_data: dict, posterior_params: np.ndarray
    ) -> np.ndarray:
        """
        Генерация импутации на основе апостериорного распределения параметров.

        Параметры
        ---------
        censored_data : dict
            См. _generate_single_imputation_parametric.
        posterior_params : np.ndarray формы (n_samples, 2)
            Семплы (mu, sigma) из апостериорного распределения.
        """
        rng = np.random.default_rng()
        idx = rng.integers(0, len(posterior_params))
        mu, sigma = posterior_params[idx]

        observed = np.asarray(censored_data["observed"], dtype=float)
        lods = np.asarray(censored_data["censored_lods"], dtype=float)

        imputed = np.empty(len(lods))
        for i, lod_i in enumerate(lods):
            imputed[i] = self._sample_truncated(lod_i, float(mu), float(sigma), rng)

        return np.concatenate([observed, imputed])

    def _generate_single_imputation_pmm(
        self, censored_data: dict, iteration: int, k: int = 5
    ) -> np.ndarray:
        """
        Predictive Mean Matching: для каждого цензурированного наблюдения
        берётся одно из k ближайших по значению наблюдаемых значений <= LOD
        (или k наименьших, если <= LOD нет).
        """
        rng = np.random.default_rng(seed=iteration * 1000 + 13)
        observed = np.asarray(censored_data["observed"], dtype=float)
        lods = np.asarray(censored_data["censored_lods"], dtype=float)

        imputed = np.empty(len(lods))
        for i, lod_i in enumerate(lods):
            candidates = observed[observed < lod_i]
            if len(candidates) == 0:
                # fallback: берём k наименьших наблюдаемых
                sorted_obs = np.sort(observed)
                candidates = sorted_obs[: max(k, 1)]
            # k ближайших к LOD снизу
            dists = np.abs(candidates - lod_i)
            k_eff = min(k, len(candidates))
            nearest_idx = np.argsort(dists)[:k_eff]
            donor = candidates[rng.choice(nearest_idx)]
            # гарантируем < LOD
            if donor >= lod_i:
                donor = lod_i * 0.99
            imputed[i] = donor

        return np.concatenate([observed, imputed])

    # ------------------------------------------------------------------ #
    # Правила Рубина                                                     #
    # ------------------------------------------------------------------ #

    def _rubins_rules(
        self, estimates: List[float], variances: List[float]
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Объединение M оценок по правилам Рубина.

        Параметры
        ---------
        estimates : list of float
            Точечные оценки из каждой импутации.
        variances : list of float
            Соответствующие оценки дисперсии (квадрат SE).

        Возвращает
        ----------
        dict с ключами:
            combined_estimate, within_variance, between_variance,
            total_variance, degrees_of_freedom,
            confidence_interval_95, fraction_missing_info,
            relative_efficiency.
        """
        estimates = np.asarray(estimates, dtype=float)
        variances = np.asarray(variances, dtype=float)
        M = len(estimates)

        if M < 2:
            raise ValueError("Для правил Рубина нужно минимум 2 импутации.")

        theta_bar = float(np.mean(estimates))
        W = float(np.mean(variances))
        B = float(np.var(estimates, ddof=1))
        T = W + (1.0 + 1.0 / M) * B

        # Степени свободы (Rubin 1987)
        if B > 0 and W > 0:
            df = (M - 1) * (1.0 + W / ((1.0 + 1.0 / M) * B)) ** 2
        else:
            df = float("inf")

        # Доверительный интервал
        if np.isfinite(df) and df > 0:
            t_crit = stats.t.ppf(0.975, df)
        else:
            t_crit = 1.96
        se = np.sqrt(max(T, 0.0))
        ci = (theta_bar - t_crit * se, theta_bar + t_crit * se)

        # Fraction of missing information
        if T > 0:
            gamma = ((1.0 + 1.0 / M) * B) / T
        else:
            gamma = 0.0
        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Относительная эффективность
        rel_eff = 1.0 / (1.0 + gamma / M)

        return {
            "combined_estimate": theta_bar,
            "within_variance": W,
            "between_variance": B,
            "total_variance": T,
            "degrees_of_freedom": float(df),
            "confidence_interval_95": ci,
            "fraction_missing_info": gamma,
            "relative_efficiency": float(rel_eff),
        }

    # ------------------------------------------------------------------ #
    # Статистики для одного датасета                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_statistics(
        data: np.ndarray, lod_threshold: Optional[float] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Вычисление статистик и их стандартных ошибок для одного датасета.

        Возвращает словарь: имя -> (оценка, дисперсия оценки).
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        results: Dict[str, Tuple[float, float]] = {}

        # Среднее и его дисперсия
        mean = float(np.mean(data))
        var_data = float(np.var(data, ddof=1)) if n > 1 else 0.0
        var_mean = var_data / n
        results["mean"] = (mean, var_mean)

        # Дисперсия и её дисперсия (приближение для нормального распределения)
        var_of_var = 2.0 * var_data ** 2 / max(n - 1, 1)
        results["variance"] = (var_data, var_of_var)

        # Перцентили — SE через асимптотическую формулу:
        # Var(q_p) ≈ p(1-p) / (n * f(q_p)^2)
        # Для простоты и устойчивости используем бутстрап внутри датасета
        rng = np.random.default_rng(seed=0)
        n_boot = 200
        for p in [50, 75, 90, 95, 99]:
            q = float(np.percentile(data, p))
            boot_qs = np.empty(n_boot)
            for b in range(n_boot):
                sample = data[rng.integers(0, n, size=n)]
                boot_qs[b] = np.percentile(sample, p)
            var_q = float(np.var(boot_qs, ddof=1))
            results[f"p{p}"] = (q, var_q)

        # Вероятность превышения LOD (или заданного порога)
        if lod_threshold is not None:
            p_hat = float(np.mean(data > lod_threshold))
            var_p = p_hat * (1 - p_hat) / n
            results["exceedance_prob"] = (p_hat, max(var_p, 1e-12))

        return results

    # ------------------------------------------------------------------ #
    # Основной метод fit                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, censored_data: dict) -> dict:
        """
        Основной метод: выполняет M импутаций и объединяет результаты по Рубину.

        Параметры
        ---------
        censored_data : dict
            {
              'observed': np.ndarray,         # наблюдаемые значения
              'censored_lods': np.ndarray,    # LOD для каждого цензурированного
              'exceedance_threshold': float,  # (опц.) порог для P(X > threshold)
              'posterior_params': np.ndarray  # (опц.) для bayesian метода
            }
        """
        observed = np.asarray(censored_data["observed"], dtype=float)
        lods = np.asarray(censored_data["censored_lods"], dtype=float)
        exceedance_threshold = censored_data.get(
            "exceedance_threshold",
            float(np.max(lods)) if len(lods) > 0 else float(np.max(observed)),
        )
        posterior_params = censored_data.get("posterior_params", None)

        self.logger.info(
            f"Начало импутации: n_observed={len(observed)}, "
            f"n_censored={len(lods)}, M={self.n_imputations}"
        )

        imputed_datasets: List[np.ndarray] = []
        individual_rows: List[Dict[str, float]] = []
        per_stat_estimates: Dict[str, List[float]] = {}
        per_stat_variances: Dict[str, List[float]] = {}

        for m in range(self.n_imputations):
            if self.imputation_method == "parametric_bootstrap":
                imputed = self._generate_single_imputation_parametric(
                    censored_data, iteration=m
                )
            elif self.imputation_method == "bayesian":
                if posterior_params is None:
                    raise ValueError(
                        "Для метода 'bayesian' необходим 'posterior_params'."
                    )
                imputed = self._generate_single_imputation_bayesian(
                    censored_data, posterior_params
                )
            elif self.imputation_method == "pmm":
                imputed = self._generate_single_imputation_pmm(
                    censored_data, iteration=m
                )
            else:
                raise ValueError(
                    f"Неизвестный imputation_method: {self.imputation_method}"
                )

            imputed_datasets.append(imputed)

            stats_m = self._compute_statistics(
                imputed, lod_threshold=exceedance_threshold
            )

            row = {"iteration": m}
            for stat_name, (est, var) in stats_m.items():
                row[stat_name] = est
                row[f"{stat_name}_var"] = var
                per_stat_estimates.setdefault(stat_name, []).append(est)
                per_stat_variances.setdefault(stat_name, []).append(var)
            individual_rows.append(row)

        individual_df = pd.DataFrame(individual_rows)

        # Применяем правила Рубина к каждой статистике
        combined_results: Dict[str, Dict] = {}
        fraction_missing_info: Dict[str, float] = {}

        for stat_name in per_stat_estimates.keys():
            rubin = self._rubins_rules(
                per_stat_estimates[stat_name],
                per_stat_variances[stat_name],
            )
            combined_results[stat_name] = {
                "estimate": rubin["combined_estimate"],
                "se": float(np.sqrt(max(rubin["total_variance"], 0.0))),
                "ci_95": rubin["confidence_interval_95"],
                "df": rubin["degrees_of_freedom"],
                "within_var": rubin["within_variance"],
                "between_var": rubin["between_variance"],
                "fmi": rubin["fraction_missing_info"],
                "relative_efficiency": rubin["relative_efficiency"],
            }
            fraction_missing_info[stat_name] = rubin["fraction_missing_info"]

        # Диагностика сходимости: максимальная FMI × (1/sqrt(M))
        max_fmi = max(fraction_missing_info.values()) if fraction_missing_info else 0.0
        convergence_diag = float(max_fmi / np.sqrt(self.n_imputations))

        percentiles_dict = {
            p: combined_results[f"p{p}"]["estimate"] for p in [75, 90, 95, 99]
        }

        result = {
            "method_name": "multiple_imputation",
            "n_imputations": self.n_imputations,
            "distribution": self.distribution,
            "imputation_method": self.imputation_method,
            "imputed_datasets": imputed_datasets,
            "individual_estimates": individual_df,
            "combined_results": combined_results,
            "estimated_mean": combined_results["mean"]["estimate"],
            "estimated_variance": combined_results["variance"]["estimate"],
            "estimated_std": float(np.sqrt(max(combined_results["variance"]["estimate"], 0.0))),
            "estimated_median": combined_results["p50"]["estimate"],
            "estimated_percentiles": percentiles_dict,
            "estimated_exceedance_prob": combined_results.get(
                "exceedance_prob", {"estimate": np.nan}
            )["estimate"],
            "fraction_missing_info": fraction_missing_info,
            "convergence_diagnostic": convergence_diag,
        }

        self.logger.info(
            f"Импутация завершена. mean={result['estimated_mean']:.4f}, "
            f"max FMI={max_fmi:.3f}, conv_diag={convergence_diag:.4f}"
        )

        return result

    # ------------------------------------------------------------------ #
    # Проверка достаточности M                                           #
    # ------------------------------------------------------------------ #

    def check_n_imputations(
        self, censored_data: dict, max_m: int = 100, step: int = 5
    ) -> dict:
        """
        Исследование сходимости оценок при увеличении M.

        Для каждого M в [step, 2*step, ..., max_m] запускает fit и отслеживает
        стабильность оценки среднего и его SE. Рекомендованное M — минимальное,
        при котором относительные изменения становятся малыми.
        """
        original_m = self.n_imputations
        ms = list(range(step, max_m + 1, step))
        means, ses, fmis = [], [], []

        for M in ms:
            self.n_imputations = M
            res = self.fit(censored_data)
            means.append(res["combined_results"]["mean"]["estimate"])
            ses.append(res["combined_results"]["mean"]["se"])
            fmis.append(max(res["fraction_missing_info"].values()))

        self.n_imputations = original_m

        # Рекомендация: относительное изменение SE между соседними < 2%
        recommended_m = ms[-1]
        for i in range(1, len(ses)):
            rel_change = abs(ses[i] - ses[i - 1]) / max(abs(ses[i - 1]), 1e-12)
            if rel_change < 0.02:
                recommended_m = ms[i]
                break

        # График сходимости
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ms, means, "o-")
        axes[0].set_xlabel("M")
        axes[0].set_ylabel("Среднее")
        axes[0].set_title("Сходимость оценки среднего")
        axes[0].axvline(recommended_m, color="red", linestyle="--", label=f"M*={recommended_m}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(ms, ses, "s-", color="orange")
        axes[1].set_xlabel("M")
        axes[1].set_ylabel("SE(среднего)")
        axes[1].set_title("Сходимость SE")
        axes[1].axvline(recommended_m, color="red", linestyle="--")
        axes[1].grid(alpha=0.3)

        axes[2].plot(ms, fmis, "^-", color="green")
        axes[2].set_xlabel("M")
        axes[2].set_ylabel("max FMI")
        axes[2].set_title("Доля missing information")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        return {
            "ms": ms,
            "means": means,
            "ses": ses,
            "fmis": fmis,
            "recommended_m": recommended_m,
            "figure": fig,
        }

    # ------------------------------------------------------------------ #
    # Диагностические графики                                            #
    # ------------------------------------------------------------------ #

    def plot_imputation_diagnostics(self, result: dict) -> Figure:
        """
        Построение диагностических графиков.
        """
        imputed_datasets = result["imputed_datasets"]
        individual = result["individual_estimates"]
        fmi = result["fraction_missing_info"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1) Гистограммы заполненных датасетов
        ax = axes[0, 0]
        n_show = min(5, len(imputed_datasets))
        for i in range(n_show):
            ax.hist(
                imputed_datasets[i],
                bins=30,
                alpha=0.35,
                density=True,
                label=f"Imp #{i+1}",
            )
        ax.set_title(f"Распределения {n_show} импутированных датасетов")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Плотность")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 2) Разброс оценок среднего между импутациями
        ax = axes[0, 1]
        ax.plot(individual["iteration"], individual["mean"], "o-", color="steelblue")
        combined_mean = result["combined_results"]["mean"]["estimate"]
        ci = result["combined_results"]["mean"]["ci_95"]
        ax.axhline(combined_mean, color="red", linestyle="--", label=f"Объединённое={combined_mean:.3f}")
        ax.fill_between(
            individual["iteration"],
            ci[0], ci[1],
            alpha=0.2, color="red", label=f"95% ДИ"
        )
        ax.set_title("Оценки среднего по импутациям")
        ax.set_xlabel("Номер импутации")
        ax.set_ylabel("Среднее")
        ax.legend()
        ax.grid(alpha=0.3)

        # 3) Разброс оценок перцентилей
        ax = axes[1, 0]
        for p in ["p75", "p90", "p95"]:
            if p in individual.columns:
                ax.plot(individual["iteration"], individual[p], "o-", label=p, alpha=0.7)
        ax.set_title("Оценки перцентилей по импутациям")
        ax.set_xlabel("Номер импутации")
        ax.set_ylabel("Значение перцентиля")
        ax.legend()
        ax.grid(alpha=0.3)

        # 4) Fraction of Missing Information
        ax = axes[1, 1]
        names = list(fmi.keys())
        values = [fmi[n] for n in names]
        colors = ["crimson" if v > 0.5 else "orange" if v > 0.2 else "seagreen" for v in values]
        ax.barh(names, values, color=colors)
        ax.axvline(0.2, color="gray", linestyle=":", label="FMI=0.2")
        ax.axvline(0.5, color="black", linestyle=":", label="FMI=0.5")
        ax.set_title("Fraction of Missing Information")
        ax.set_xlabel("FMI")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        return fig


# ====================================================================== #
# ДЕМО-БЛОК                                                              #
# ====================================================================== #

if __name__ == "__main__":
    np.random.seed(42)

    # Генерация синтетических данных: логнормальное распределение
    true_mu, true_sigma = 1.0, 0.6
    n_total = 150
    full_data = np.random.lognormal(mean=true_mu, sigma=true_sigma, size=n_total)

    # Цензурирование: значения ниже LOD = 1.5 заменяются на LOD
    LOD = 1.5
    observed_mask = full_data > LOD
    observed_values = full_data[observed_mask]
    n_censored = int(np.sum(~observed_mask))
    censored_lods = np.full(n_censored, LOD)

    print("=" * 65)
    print("ДЕМОНСТРАЦИЯ MultipleImputation")
    print("=" * 65)
    print(f"Всего наблюдений: {n_total}")
    print(f"Наблюдаемых (> LOD): {len(observed_values)}")
    print(f"Цензурированных (< LOD): {n_censored}")
    print(f"Истинное среднее (lognormal): {np.exp(true_mu + true_sigma**2/2):.3f}")
    print(f"Среднее по полным данным:    {np.mean(full_data):.3f}")

    censored_data = {
        "observed": observed_values,
        "censored_lods": censored_lods,
        "exceedance_threshold": 5.0,
    }

    # Запуск множественной импутации
    mi = MultipleImputation(
        n_imputations=25,
        distribution="lognormal",
        imputation_method="parametric_bootstrap",
    )
    result = mi.fit(censored_data)

    print("\n--- Объединённые оценки (правила Рубина) ---")
    for stat, info in result["combined_results"].items():
        ci = info["ci_95"]
        print(
            f"{stat:>18}: {info['estimate']:>8.4f}  "
            f"SE={info['se']:>7.4f}  "
            f"95% CI=[{ci[0]:>7.4f}, {ci[1]:>7.4f}]  "
            f"df={info['df']:>7.1f}  FMI={info['fmi']:.3f}"
        )

    print(f"\nДиагностика сходимости: {result['convergence_diagnostic']:.5f}")
    print(f"Оценка P(X > 5.0): {result['estimated_exceedance_prob']:.4f}")

    # Диагностические графики
    fig = mi.plot_imputation_diagnostics(result)
    fig.savefig("mi_diagnostics.png", dpi=100, bbox_inches="tight")
    print("\nДиагностические графики сохранены в 'mi_diagnostics.png'.")

    # Проверка достаточности M
    print("\n--- Проверка достаточности M (это займёт немного времени) ---")
    conv = mi.check_n_imputations(censored_data, max_m=40, step=5)
    print(f"Рекомендованное M: {conv['recommended_m']}")
    conv["figure"].savefig("mi_convergence.png", dpi=100, bbox_inches="tight")
    print("График сходимости сохранён в 'mi_convergence.png'.")

    plt.close("all")
    print("\nГотово.")