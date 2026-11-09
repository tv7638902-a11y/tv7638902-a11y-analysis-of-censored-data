"""
src/module_4_mle.py

Метод максимального правдоподобия (MLE) для оценки параметров
распределения по левоцензурированным данным.

Теоретическая основа
--------------------
Для левоцензурированных данных с одним порогом детекции LOD функция
правдоподобия имеет вид:

    L(θ) = ∏_{uncensored} f(xᵢ; θ) × ∏_{censored} F(LOD; θ)

где f(·; θ) — плотность, F(·; θ) — функция распределения.

Логарифм правдоподобия:

    ℓ(θ) = Σ_{uncensored} ln f(xᵢ; θ) + Σ_{censored} ln F(LOD; θ)

MLE оценивает θ путём максимизации ℓ(θ) (минимизации -ℓ(θ)).
Это более эффективный подход по сравнению с методами подстановки,
так как он использует всю доступную информацию, включая факт
нахождения цензурированных значений ниже LOD.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy import optimize, stats, special

try:
    import numdifftools as nd
    _HAS_NDT = True
except ImportError:
    _HAS_NDT = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

_EPS = 1e-12


class MLECensored:
    """
    MLE для левоцензурированных данных.

    Параметры
    ---------
    distribution : str
        "normal" | "lognormal" | "gamma" | "weibull"
    """

    SUPPORTED = ("normal", "lognormal", "gamma", "weibull")

    def __init__(self, distribution: str = "lognormal"):
        if distribution not in self.SUPPORTED:
            raise ValueError(
                f"Неподдерживаемое распределение: {distribution}. "
                f"Доступно: {self.SUPPORTED}"
            )
        self.distribution = distribution
        self.logger = logger
        self.logger.debug("MLECensored инициализирован: %s", distribution)

    # ------------------------------------------------------------------
    # Функции отрицательного логарифмического правдоподобия
    # ------------------------------------------------------------------

    def _log_likelihood_normal(
        self,
        params: tuple,
        uncensored: np.ndarray,
        lod,
        n_censored: int,
    ) -> float:
        """
        Отрицательный лог-правдоподобие для нормального распределения.

        ℓ(μ,σ) = Σ ln φ((xᵢ-μ)/σ) - n_u·ln(σ) + n_c · ln Φ((LOD-μ)/σ)
        """
        mu, sigma = params
        if sigma <= 0:
            return 1e12

        ll = 0.0
        if uncensored.size > 0:
            ll += np.sum(stats.norm.logpdf(uncensored, loc=mu, scale=sigma))

        if n_censored > 0:
            lod_arr = np.atleast_1d(lod)
            if lod_arr.size == 1:
                cdf = stats.norm.cdf(lod_arr[0], loc=mu, scale=sigma)
                cdf = max(cdf, _EPS)
                ll += n_censored * np.log(cdf)
            else:
                cdf = stats.norm.cdf(lod_arr, loc=mu, scale=sigma)
                cdf = np.clip(cdf, _EPS, 1.0)
                ll += np.sum(np.log(cdf))

        if not np.isfinite(ll):
            return 1e12
        return -ll

    def _log_likelihood_lognormal(
        self,
        params: tuple,
        uncensored: np.ndarray,
        lod,
        n_censored: int,
    ) -> float:
        """
        Отрицательный лог-правдоподобие для логнормального распределения.

        Параметры: (μ_log, σ_log) — среднее и СКО в лог-пространстве.

        ℓ = Σ ln f_LN(xᵢ) + n_c · ln Φ((ln LOD - μ_log)/σ_log)
        """
        mu_log, sigma_log = params
        if sigma_log <= 0:
            return 1e12

        ll = 0.0
        if uncensored.size > 0:
            pos = uncensored[uncensored > 0]
            if pos.size != uncensored.size:
                return 1e12
            ll += np.sum(stats.lognorm.logpdf(
                pos, s=sigma_log, scale=np.exp(mu_log)
            ))

        if n_censored > 0:
            lod_arr = np.atleast_1d(lod).astype(float)
            if np.any(lod_arr <= 0):
                return 1e12
            z = (np.log(lod_arr) - mu_log) / sigma_log
            cdf = stats.norm.cdf(z)
            cdf = np.clip(cdf, _EPS, 1.0)
            if lod_arr.size == 1:
                ll += n_censored * np.log(cdf[0])
            else:
                ll += np.sum(np.log(cdf))

        if not np.isfinite(ll):
            return 1e12
        return -ll

    def _log_likelihood_gamma(
        self,
        params: tuple,
        uncensored: np.ndarray,
        lod,
        n_censored: int,
    ) -> float:
        """
        Отрицательный лог-правдоподобие для гамма-распределения.

        Параметры: (shape k, scale θ).
        """
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return 1e12

        ll = 0.0
        if uncensored.size > 0:
            if np.any(uncensored <= 0):
                return 1e12
            ll += np.sum(stats.gamma.logpdf(uncensored, a=shape, scale=scale))

        if n_censored > 0:
            lod_arr = np.atleast_1d(lod).astype(float)
            if np.any(lod_arr <= 0):
                return 1e12
            cdf = stats.gamma.cdf(lod_arr, a=shape, scale=scale)
            cdf = np.clip(cdf, _EPS, 1.0)
            if lod_arr.size == 1:
                ll += n_censored * np.log(cdf[0])
            else:
                ll += np.sum(np.log(cdf))

        if not np.isfinite(ll):
            return 1e12
        return -ll

    def _log_likelihood_weibull(
        self,
        params: tuple,
        uncensored: np.ndarray,
        lod,
        n_censored: int,
    ) -> float:
        """
        Отрицательный лог-правдоподобие для распределения Вейбулла.

        Параметры: (shape c, scale λ).
        """
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return 1e12

        ll = 0.0
        if uncensored.size > 0:
            if np.any(uncensored <= 0):
                return 1e12
            ll += np.sum(stats.weibull_min.logpdf(
                uncensored, c=shape, scale=scale
            ))

        if n_censored > 0:
            lod_arr = np.atleast_1d(lod).astype(float)
            if np.any(lod_arr <= 0):
                return 1e12
            cdf = stats.weibull_min.cdf(lod_arr, c=shape, scale=scale)
            cdf = np.clip(cdf, _EPS, 1.0)
            if lod_arr.size == 1:
                ll += n_censored * np.log(cdf[0])
            else:
                ll += np.sum(np.log(cdf))

        if not np.isfinite(ll):
            return 1e12
        return -ll

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    def _extract_data(self, censored_data: dict):
        """Извлекает массивы наблюдений, индикаторов и LOD."""
        values = np.asarray(censored_data.get("values"), dtype=float)
        if "censored" in censored_data:
            censored_flags = np.asarray(censored_data["censored"], dtype=bool)
        elif "is_censored" in censored_data:
            censored_flags = np.asarray(censored_data["is_censored"], dtype=bool)
        else:
            raise KeyError("censored_data должен содержать 'censored' или 'is_censored'")

        if "lod" in censored_data:
            lod = censored_data["lod"]
        elif "LOD" in censored_data:
            lod = censored_data["LOD"]
        elif "detection_limit" in censored_data:
            lod = censored_data["detection_limit"]
        else:
            # пробуем взять как максимум цензурированных
            if censored_flags.any():
                lod = float(np.max(values[censored_flags]))
            else:
                raise KeyError("Не указан LOD")

        uncensored = values[~censored_flags]
        return values, censored_flags, uncensored, lod

    def _initial_guess(self, uncensored: np.ndarray, lod_scalar: float) -> tuple:
        """Начальные приближения параметров."""
        if uncensored.size < 2:
            # грубая оценка
            u = np.array([lod_scalar / 2.0, lod_scalar])
        else:
            u = uncensored

        if self.distribution == "normal":
            return (float(np.mean(u)), float(np.std(u, ddof=1) + 1e-3))
        if self.distribution == "lognormal":
            pos = u[u > 0]
            if pos.size < 2:
                pos = np.array([lod_scalar / 2.0, lod_scalar])
            lp = np.log(pos)
            return (float(np.mean(lp)), float(np.std(lp, ddof=1) + 1e-3))
        if self.distribution == "gamma":
            m = float(np.mean(u))
            v = float(np.var(u, ddof=1) + 1e-6)
            scale = v / m
            shape = m / scale
            return (max(shape, 0.5), max(scale, 1e-3))
        if self.distribution == "weibull":
            m = float(np.mean(u))
            return (1.2, max(m, 1e-3))
        raise ValueError(self.distribution)

    def _get_nll(self):
        return {
            "normal": self._log_likelihood_normal,
            "lognormal": self._log_likelihood_lognormal,
            "gamma": self._log_likelihood_gamma,
            "weibull": self._log_likelihood_weibull,
        }[self.distribution]

    def _param_bounds(self):
        if self.distribution == "normal":
            return [(None, None), (1e-6, None)]
        if self.distribution == "lognormal":
            return [(None, None), (1e-6, None)]
        if self.distribution == "gamma":
            return [(1e-4, None), (1e-6, None)]
        if self.distribution == "weibull":
            return [(1e-4, None), (1e-6, None)]
        raise ValueError

    def _param_names(self):
        return {
            "normal": ("mu", "sigma"),
            "lognormal": ("mu_log", "sigma_log"),
            "gamma": ("shape", "scale"),
            "weibull": ("shape", "scale"),
        }[self.distribution]

    # ------------------------------------------------------------------
    # Оптимизация
    # ------------------------------------------------------------------

    def _optimize(self, nll_fn, x0, args):
        """Многопопыточная оптимизация с разными методами."""
        bounds = self._param_bounds()
        best = None

        # серия начальных точек
        starts = [x0]
        rng = np.random.default_rng(42)
        for _ in range(4):
            jitter = x0 * np.exp(rng.normal(0, 0.3, size=len(x0)))
            jitter = tuple(float(max(1e-4, j)) if b[0] is not None and b[0] > 0 else float(j)
                           for j, b in zip(jitter, bounds))
            starts.append(jitter)

        for method in ("L-BFGS-B", "Nelder-Mead"):
            for s0 in starts:
                try:
                    if method == "L-BFGS-B":
                        res = optimize.minimize(
                            nll_fn, x0=s0, args=args,
                            method=method, bounds=bounds,
                        )
                    else:
                        res = optimize.minimize(
                            nll_fn, x0=s0, args=args,
                            method=method,
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
                        )
                    if res.success and np.isfinite(res.fun):
                        if best is None or res.fun < best.fun:
                            best = res
                except Exception as e:
                    self.logger.debug("Опт. ошибка (%s): %s", method, e)

        return best

    # ------------------------------------------------------------------
    # Распределение-свойства
    # ------------------------------------------------------------------

    def _distribution_object(self, params_dict: dict):
        """Возвращает scipy frozen distribution."""
        if self.distribution == "normal":
            return stats.norm(loc=params_dict["mu"], scale=params_dict["sigma"])
        if self.distribution == "lognormal":
            return stats.lognorm(s=params_dict["sigma_log"],
                                 scale=np.exp(params_dict["mu_log"]))
        if self.distribution == "gamma":
            return stats.gamma(a=params_dict["shape"], scale=params_dict["scale"])
        if self.distribution == "weibull":
            return stats.weibull_min(c=params_dict["shape"], scale=params_dict["scale"])
        raise ValueError

    # ------------------------------------------------------------------
    # E[X | X < LOD]
    # ------------------------------------------------------------------

    def compute_conditional_mean_below_lod(
        self, params: dict, lod: float
    ) -> float:
        """
        Вычисляет E[X | X < LOD] для подогнанного распределения.

        - Нормальное:
            E[X|X<L] = μ - σ · φ(z) / Φ(z),  z = (L-μ)/σ
        - Логнормальное:
            E[X|X<L] = exp(μ+σ²/2) · Φ((ln L - μ - σ²)/σ) / Φ((ln L - μ)/σ)
        - Для гамма и Вейбулла — числовое интегрирование.
        """
        lod = float(lod)
        if self.distribution == "normal":
            mu, sigma = params["mu"], params["sigma"]
            z = (lod - mu) / sigma
            Phi = max(stats.norm.cdf(z), _EPS)
            phi = stats.norm.pdf(z)
            return float(mu - sigma * phi / Phi)

        if self.distribution == "lognormal":
            mu, sigma = params["mu_log"], params["sigma_log"]
            if lod <= 0:
                return lod / 2.0
            ln_l = np.log(lod)
            z = (ln_l - mu) / sigma
            Phi = max(stats.norm.cdf(z), _EPS)
            Phi_shift = stats.norm.cdf(z - sigma)
            return float(np.exp(mu + sigma**2 / 2.0) * Phi_shift / Phi)

        dist = self._distribution_object(params)
        F_lod = max(dist.cdf(lod), _EPS)
        # численное интегрирование ∫₀^LOD x f(x) dx / F(LOD)
        try:
            from scipy.integrate import quad
            lo = 0.0 if self.distribution in ("gamma", "weibull") else -np.inf
            val, _ = quad(lambda x: x * dist.pdf(x), lo, lod, limit=200)
            return float(val / F_lod)
        except Exception as e:
            self.logger.warning("Числ. интегрирование не удалось: %s", e)
            return float(lod / 2.0)

    # ------------------------------------------------------------------
    # Основной метод fit
    # ------------------------------------------------------------------

    def fit(self, censored_data: dict) -> dict:
        """
        Подгоняет параметры распределения методом MLE.

        Параметры
        ---------
        censored_data : dict
            Должен содержать:
              - 'values': массив всех наблюдений
              - 'censored' (или 'is_censored'): булев массив
              - 'lod' (или 'LOD'): порог детекции (скаляр)
        """
        values, flags, uncensored, lod = self._extract_data(censored_data)
        n = values.size
        n_censored = int(flags.sum())
        n_uncensored = n - n_censored

        cens_frac = n_censored / max(n, 1)
        if cens_frac > 0.8:
            self.logger.warning(
                "Доля цензурирования %.1f%% > 80%%: MLE может быть ненадёжен",
                cens_frac * 100,
            )
        if n_uncensored < 3:
            self.logger.warning(
                "Слишком мало нецензурированных наблюдений (%d)", n_uncensored
            )

        if np.ndim(lod) > 0:
            lod_scalar = float(np.mean(lod))
        else:
            lod_scalar = float(lod)

        x0 = self._initial_guess(uncensored, lod_scalar)
        nll_fn = self._get_nll()
        args = (uncensored, lod_scalar, n_censored)

        self.logger.info(
            "MLE fit: dist=%s, n=%d, n_c=%d, LOD=%.4g, x0=%s",
            self.distribution, n, n_censored, lod_scalar, x0,
        )

        res = self._optimize(nll_fn, np.array(x0, dtype=float), args)

        if res is None:
            self.logger.error("Оптимизация MLE не сошлась")
            return self._failed_result(x0)

        params_arr = res.x
        param_names = self._param_names()
        fitted_params = {name: float(v) for name, v in zip(param_names, params_arr)}
        log_lik = -float(res.fun)
        k = len(params_arr)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(max(n, 1)) - 2 * log_lik

        # стандартные ошибки через Гессиан
        std_errors = self._compute_std_errors(nll_fn, params_arr, args, param_names)

        # статистики распределения
        dist = self._distribution_object(fitted_params)
        mean_val = float(dist.mean())
        var_val = float(dist.var())
        std_val = float(np.sqrt(var_val))
        median_val = float(dist.median())
        percentiles = {q: float(dist.ppf(q / 100.0)) for q in (75, 90, 95, 99)}
        exceed_prob = float(1.0 - dist.cdf(lod_scalar))

        # доверительные интервалы (дельта/бутстрап по параметрам)
        ci = self._approx_confidence_intervals(fitted_params, std_errors)

        # импутация
        imputed = values.copy()
        if n_censored > 0:
            cond_mean = self.compute_conditional_mean_below_lod(fitted_params, lod_scalar)
            imputed[flags] = cond_mean

        result = {
            "method_name": f"mle_{self.distribution}",
            "distribution": self.distribution,
            "fitted_params": fitted_params,
            "param_std_errors": std_errors,
            "log_likelihood": log_lik,
            "aic": float(aic),
            "bic": float(bic),
            "convergence": True,
            "estimated_mean": mean_val,
            "estimated_variance": var_val,
            "estimated_std": std_val,
            "estimated_median": median_val,
            "estimated_percentiles": percentiles,
            "estimated_exceedance_prob": exceed_prob,
            "confidence_intervals": ci,
            "imputed_values": imputed,
        }
        self.logger.info(
            "MLE готово: params=%s, logL=%.3f, AIC=%.2f",
            fitted_params, log_lik, aic,
        )
        return result

    def _failed_result(self, x0) -> dict:
        param_names = self._param_names()
        return {
            "method_name": f"mle_{self.distribution}",
            "distribution": self.distribution,
            "fitted_params": {n: float(v) for n, v in zip(param_names, x0)},
            "param_std_errors": {n: float("nan") for n in param_names},
            "log_likelihood": float("nan"),
            "aic": float("nan"),
            "bic": float("nan"),
            "convergence": False,
            "estimated_mean": float("nan"),
            "estimated_variance": float("nan"),
            "estimated_std": float("nan"),
            "estimated_median": float("nan"),
            "estimated_percentiles": {q: float("nan") for q in (75, 90, 95, 99)},
            "estimated_exceedance_prob": float("nan"),
            "confidence_intervals": {
                "mean_95ci": (float("nan"), float("nan")),
                "p95_95ci": (float("nan"), float("nan")),
            },
            "imputed_values": np.array([]),
        }

    def _compute_std_errors(self, nll_fn, params_arr, args, param_names):
        """Стандартные ошибки через обратную матрицу Гессиана."""
        try:
            if _HAS_NDT:
                H = nd.Hessian(lambda p: nll_fn(tuple(p), *args))(params_arr)
            else:
                H = self._numeric_hessian(
                    lambda p: nll_fn(tuple(p), *args), params_arr
                )
            cov = np.linalg.inv(H)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            return {name: float(s) for name, s in zip(param_names, se)}
        except Exception as e:
            self.logger.warning("Не удалось оценить SE параметров: %s", e)
            return {name: float("nan") for name in param_names}

    @staticmethod
    def _numeric_hessian(fn, x, eps: float = 1e-4):
        """Простая центральная разностная оценка Гессиана."""
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n))
        f0 = fn(x)
        for i in range(n):
            for j in range(i, n):
                xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                H[i, j] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * eps * eps)
                H[j, i] = H[i, j]
        return H

    def _approx_confidence_intervals(self, params: dict, se: dict) -> dict:
        """
        Приближённые 95% ДИ для среднего и 95-перцентиля через
        параметрическую симуляцию по SE параметров.
        """
        names = list(params.keys())
        try:
            mu_vec = np.array([params[n] for n in names])
            sd_vec = np.array([se.get(n, 0.0) for n in names])
            if np.any(~np.isfinite(sd_vec)) or np.all(sd_vec == 0):
                raise ValueError("нет SE")
            rng = np.random.default_rng(123)
            n_sim = 500
            means, p95s = [], []
            for _ in range(n_sim):
                draw = rng.normal(mu_vec, sd_vec)
                p = dict(zip(names, draw))
                try:
                    if self.distribution in ("gamma", "weibull") and (
                        p.get("shape", 1) <= 0 or p.get("scale", 1) <= 0
                    ):
                        continue
                    if self.distribution == "normal" and p["sigma"] <= 0:
                        continue
                    if self.distribution == "lognormal" and p["sigma_log"] <= 0:
                        continue
                    d = self._distribution_object(p)
                    means.append(d.mean())
                    p95s.append(d.ppf(0.95))
                except Exception:
                    continue
            if len(means) < 10:
                raise ValueError
            return {
                "mean_95ci": (
                    float(np.percentile(means, 2.5)),
                    float(np.percentile(means, 97.5)),
                ),
                "p95_95ci": (
                    float(np.percentile(p95s, 2.5)),
                    float(np.percentile(p95s, 97.5)),
                ),
            }
        except Exception as e:
            self.logger.debug("ДИ не вычислены: %s", e)
            return {
                "mean_95ci": (float("nan"), float("nan")),
                "p95_95ci": (float("nan"), float("nan")),
            }

    # ------------------------------------------------------------------
    # Variable LOD
    # ------------------------------------------------------------------

    def fit_with_variable_lod(self, censored_data: dict) -> dict:
        """
        MLE для данных с несколькими порогами LOD.

        censored_data должен содержать:
          - 'values', 'censored'
          - 'lods': массив LOD той же длины, что и values
                    (для нецензурированных значения не используются)
        """
        values = np.asarray(censored_data["values"], dtype=float)
        flags = np.asarray(
            censored_data.get("censored", censored_data.get("is_censored")),
            dtype=bool,
        )
        lods = np.asarray(censored_data["lods"], dtype=float)
        if lods.shape != values.shape:
            raise ValueError("lods должен иметь ту же форму, что и values")

        uncensored = values[~flags]
        lods_censored = lods[flags]
        n = values.size
        n_censored = int(flags.sum())
        n_uncensored = n - n_censored

        if n_censored / max(n, 1) > 0.8:
            self.logger.warning("Доля цензурирования > 80%% — MLE может быть ненадёжен")

        lod_scalar = float(np.mean(lods_censored)) if n_censored else float(np.mean(lods))
        x0 = self._initial_guess(uncensored, lod_scalar)
        nll_fn = self._get_nll()
        args = (uncensored, lods_censored, n_censored)

        res = self._optimize(nll_fn, np.array(x0, dtype=float), args)
        if res is None:
            return self._failed_result(x0)

        params_arr = res.x
        param_names = self._param_names()
        fitted_params = {name: float(v) for name, v in zip(param_names, params_arr)}
        log_lik = -float(res.fun)
        k = len(params_arr)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(max(n, 1)) - 2 * log_lik

        std_errors = self._compute_std_errors(nll_fn, params_arr, args, param_names)
        dist = self._distribution_object(fitted_params)

        imputed = values.copy()
        if n_censored > 0:
            # отдельная условная среднее для каждого LOD
            unique_lods = np.unique(lods_censored)
            cond_map = {
                float(l): self.compute_conditional_mean_below_lod(fitted_params, float(l))
                for l in unique_lods
            }
            imputed[flags] = np.array([cond_map[float(l)] for l in lods_censored])

        return {
            "method_name": f"mle_{self.distribution}_variable_lod",
            "distribution": self.distribution,
            "fitted_params": fitted_params,
            "param_std_errors": std_errors,
            "log_likelihood": log_lik,
            "aic": float(aic),
            "bic": float(bic),
            "convergence": True,
            "estimated_mean": float(dist.mean()),
            "estimated_variance": float(dist.var()),
            "estimated_std": float(np.sqrt(dist.var())),
            "estimated_median": float(dist.median()),
            "estimated_percentiles": {
                q: float(dist.ppf(q / 100.0)) for q in (75, 90, 95, 99)
            },
            "estimated_exceedance_prob": float(1.0 - dist.cdf(lod_scalar)),
            "confidence_intervals": self._approx_confidence_intervals(
                fitted_params, std_errors
            ),
            "imputed_values": imputed,
        }

    # ------------------------------------------------------------------
    # Выбор лучшего распределения
    # ------------------------------------------------------------------

    def select_best_distribution(
        self,
        censored_data: dict,
        distributions: Optional[list] = None,
    ) -> dict:
        """
        Подгоняет данные ко всем распределениям и выбирает лучшее по AIC.
        """
        if distributions is None:
            distributions = list(self.SUPPORTED)

        results = {}
        for dist_name in distributions:
            try:
                mle = MLECensored(distribution=dist_name)
                res = mle.fit(censored_data)
                results[dist_name] = res
            except Exception as e:
                self.logger.warning("Ошибка для %s: %s", dist_name, e)

        valid = {k: v for k, v in results.items()
                 if v.get("convergence") and np.isfinite(v.get("aic", np.nan))}
        if not valid:
            return {"all_results": results, "best": None, "best_distribution": None}

        best_name = min(valid.keys(), key=lambda k: valid[k]["aic"])
        self.logger.info("Лучшее распределение по AIC: %s", best_name)
        return {
            "all_results": results,
            "best": valid[best_name],
            "best_distribution": best_name,
            "aic_comparison": {k: v["aic"] for k, v in valid.items()},
            "bic_comparison": {k: v["bic"] for k, v in valid.items()},
        }

    # ------------------------------------------------------------------
    # Параметрический бутстрап
    # ------------------------------------------------------------------

    def bootstrap_confidence_intervals(
        self,
        censored_data: dict,
        n_bootstrap: int = 500,
    ) -> dict:
        """
        Параметрические бутстрап-ДИ для параметров и производных статистик.

        Алгоритм:
          1) Подогнать распределение к данным (MLE).
          2) Повторять B раз:
             - сгенерировать выборку размера n из подогнанного распределения,
             - применить то же цензурирование по LOD,
             - пересчитать MLE,
             - сохранить параметры и статистики.
          3) Вычислить перцентильные 95% ДИ.
        """
        values, flags, uncensored, lod = self._extract_data(censored_data)
        n = values.size
        base = self.fit(censored_data)
        if not base["convergence"]:
            return {"error": "Базовая подгонка не сошлась"}

        lod_scalar = float(np.mean(lod)) if np.ndim(lod) > 0 else float(lod)
        dist = self._distribution_object(base["fitted_params"])
        rng = np.random.default_rng(2024)

        param_names = self._param_names()
        param_samples = {name: [] for name in param_names}
        mean_samples, p95_samples = [], []

        successes = 0
        for b in range(n_bootstrap):
            try:
                sim = dist.rvs(size=n, random_state=rng)
                sim_flags = sim < lod_scalar
                if sim_flags.sum() == n:  # все цензурированы
                    continue
                sim_data = {
                    "values": np.where(sim_flags, lod_scalar, sim),
                    "censored": sim_flags,
                    "lod": lod_scalar,
                }
                sub = MLECensored(distribution=self.distribution)
                # выключим лишние сообщения в бутстрапе
                sub.logger.setLevel(logging.ERROR)
                r = sub.fit(sim_data)
                if not r["convergence"]:
                    continue
                for name in param_names:
                    param_samples[name].append(r["fitted_params"][name])
                mean_samples.append(r["estimated_mean"])
                p95_samples.append(r["estimated_percentiles"][95])
                successes += 1
            except Exception as e:
                self.logger.debug("Бутстрап %d сбой: %s", b, e)

        self.logger.info("Bootstrap: %d/%d успешных итераций", successes, n_bootstrap)

        def pct(a):
            a = np.asarray(a, dtype=float)
            if a.size < 10:
                return (float("nan"), float("nan"))
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

        return {
            "n_bootstrap": n_bootstrap,
            "n_successful": successes,
            "param_ci": {name: pct(param_samples[name]) for name in param_names},
            "mean_95ci": pct(mean_samples),
            "p95_95ci": pct(p95_samples),
            "param_samples": {name: np.array(param_samples[name]) for name in param_names},
            "base_fit": base,
        }


# ----------------------------------------------------------------------
# Пример использования
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    rng = np.random.default_rng(0)
    # Истинное логнормальное распределение
    true_mu, true_sigma = 1.0, 0.6
    n_total = 200
    true_vals = rng.lognormal(mean=true_mu, sigma=true_sigma, size=n_total)
    LOD = 1.5
    censored_flags = true_vals < LOD
    obs_vals = np.where(censored_flags, LOD, true_vals)

    data = {
        "values": obs_vals,
        "censored": censored_flags,
        "lod": LOD,
    }
    print(f"Истинные параметры: mu={true_mu}, sigma={true_sigma}")
    print(f"Доля цензурирования: {censored_flags.mean():.1%}")

    print("\n=== MLE (логнормальное) ===")
    mle = MLECensored(distribution="lognormal")
    res = mle.fit(data)
    print(f"Параметры: {res['fitted_params']}")
    print(f"Станд. ошибки: {res['param_std_errors']}")
    print(f"log-L={res['log_likelihood']:.3f}, AIC={res['aic']:.2f}")
    print(f"Среднее: {res['estimated_mean']:.3f} (истин. {np.exp(true_mu+true_sigma**2/2):.3f})")
    print(f"P95: {res['estimated_percentiles'][95]:.3f}")
    print(f"ДИ среднего: {res['confidence_intervals']['mean_95ci']}")

    print("\n=== Выбор лучшего распределения ===")
    best = mle.select_best_distribution(data)
    print(f"Лучшее: {best['best_distribution']}")
    print(f"AIC сравнение: {best['aic_comparison']}")

    print("\n=== Переменные LOD ===")
    lods_var = rng.choice([1.0, 1.5, 2.0], size=n_total)
    flags_var = true_vals < lods_var
    obs_var = np.where(flags_var, lods_var, true_vals)
    data_var = {"values": obs_var, "censored": flags_var, "lods": lods_var}
    mle2 = MLECensored("lognormal")
    res_var = mle2.fit_with_variable_lod(data_var)
    print(f"Параметры (var LOD): {res_var['fitted_params']}")

    print("\n=== Параметрический бутстрап (n=100) ===")
    boot = mle.bootstrap_confidence_intervals(data, n_bootstrap=100)
    print(f"Успешных итераций: {boot['n_successful']}")
    print(f"ДИ параметров: {boot['param_ci']}")
    print(f"ДИ среднего: {boot['mean_95ci']}")
    print(f"ДИ P95: {boot['p95_95ci']}")