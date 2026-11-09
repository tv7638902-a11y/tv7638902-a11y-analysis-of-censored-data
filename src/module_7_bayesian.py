"""
Модуль 7: Байесовская оценка параметров распределения по цензурированным данным.

Реализует байесовский подход к оценке параметров распределений (логнормальное,
нормальное, гамма) для данных с цензурированием слева (значения ниже предела
обнаружения LOD).

Основные возможности:
- Полное апостериорное распределение параметров через MCMC
- Лапласовская аппроксимация для быстрого скрининга
- Data augmentation для цензурированных значений
- Диагностика сходимости (R-hat, ESS, acceptance rate)
- Апостериорное предиктивное распределение
- Информационные критерии (DIC, WAIC)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Dict, List, Tuple, Callable, Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


# ------------------------------------------------------------------
# Литературные априорные значения для некоторых металлов (в воде, мг/л)
# ------------------------------------------------------------------
METAL_PRIOR_INFO: Dict[str, Dict[str, float]] = {
    "Pb":  {"mu0": np.log(0.005),  "sigma0": 2.0},
    "Cd":  {"mu0": np.log(0.001),  "sigma0": 2.0},
    "Hg":  {"mu0": np.log(0.0005), "sigma0": 2.0},
    "As":  {"mu0": np.log(0.005),  "sigma0": 2.0},
    "Cr":  {"mu0": np.log(0.01),   "sigma0": 2.0},
    "Ni":  {"mu0": np.log(0.01),   "sigma0": 2.0},
    "Cu":  {"mu0": np.log(0.05),   "sigma0": 2.0},
    "Zn":  {"mu0": np.log(0.1),    "sigma0": 2.0},
    "Fe":  {"mu0": np.log(0.1),    "sigma0": 2.0},
    "default": {"mu0": np.log(0.01), "sigma0": 2.0},
}


class BayesianCensored:
    """Байесовская оценка параметров по цензурированным данным."""

    def __init__(
        self,
        distribution: str = "lognormal",
        prior_type: str = "weakly_informative",
    ):
        """
        Parameters
        ----------
        distribution : {"lognormal", "normal", "gamma"}
            Семейство распределений.
        prior_type : {"non_informative", "weakly_informative",
                      "informative", "custom"}
            Тип априорного распределения.
        """
        if distribution not in ("lognormal", "normal", "gamma"):
            raise ValueError(f"Unknown distribution: {distribution}")
        if prior_type not in (
            "non_informative", "weakly_informative", "informative", "custom"
        ):
            raise ValueError(f"Unknown prior_type: {prior_type}")

        self.distribution = distribution
        self.prior_type = prior_type

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
                )
            )
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"BayesianCensored initialized: distribution={distribution}, "
            f"prior_type={prior_type}"
        )

    # --------------------------------------------------------------
    # Priors
    # --------------------------------------------------------------
    def _set_priors(
        self,
        prior_type: str,
        metal: Optional[str] = None,
        custom_priors: Optional[dict] = None,
    ) -> dict:
        """
        Задаёт априорные распределения.

        Returns
        -------
        dict : с полями
            - "log_prior": Callable(params) -> float
            - "sample": Callable(rng) -> np.ndarray   (для инициализации)
            - "names": List[str]
            - "description": str
        """
        if self.distribution == "lognormal":
            names = ["mu_log", "sigma_log"]
        elif self.distribution == "normal":
            names = ["mu", "sigma"]
        elif self.distribution == "gamma":
            names = ["shape", "scale"]
        else:
            raise ValueError(self.distribution)

        if prior_type == "custom":
            if custom_priors is None:
                raise ValueError("custom_priors must be provided for 'custom'")
            return {
                "log_prior": custom_priors["log_prior"],
                "sample": custom_priors["sample"],
                "names": custom_priors.get("names", names),
                "description": "custom",
            }

        # --- Логнормальная модель ---
        if self.distribution == "lognormal":
            if prior_type == "non_informative":
                def log_prior(params):
                    mu, sigma = params
                    if not (-10 < mu < 10):
                        return -np.inf
                    if not (0.01 < sigma < 5):
                        return -np.inf
                    return 0.0  # Uniform

                def sample(rng):
                    return np.array([rng.uniform(-10, 10),
                                     rng.uniform(0.01, 5)])

                return {"log_prior": log_prior, "sample": sample,
                        "names": names, "description": "non-informative"}

            if prior_type == "weakly_informative":
                info = METAL_PRIOR_INFO.get(metal, METAL_PRIOR_INFO["default"])
                mu0, s0 = info["mu0"], 2.0

                def log_prior(params):
                    mu, sigma = params
                    if sigma <= 0:
                        return -np.inf
                    # mu ~ N(mu0, s0^2)
                    lp_mu = -0.5 * ((mu - mu0) / s0) ** 2 \
                            - 0.5 * np.log(2 * np.pi * s0 ** 2)
                    # sigma ~ Half-Cauchy(0, 1)
                    lp_sig = np.log(2 / (np.pi * (1 + sigma ** 2)))
                    return lp_mu + lp_sig

                def sample(rng):
                    return np.array([rng.normal(mu0, s0),
                                     abs(rng.standard_cauchy()) + 1e-3])

                return {"log_prior": log_prior, "sample": sample,
                        "names": names,
                        "description": f"weakly_informative(mu0={mu0:.2f})"}

            if prior_type == "informative":
                info = METAL_PRIOR_INFO.get(metal, METAL_PRIOR_INFO["default"])
                mu0, s0 = info["mu0"], 0.5  # узкое

                def log_prior(params):
                    mu, sigma = params
                    if sigma <= 0:
                        return -np.inf
                    lp_mu = -0.5 * ((mu - mu0) / s0) ** 2
                    # sigma ~ Half-Normal(0, 0.5)
                    lp_sig = -0.5 * (sigma / 0.5) ** 2
                    return lp_mu + lp_sig

                def sample(rng):
                    return np.array([rng.normal(mu0, s0),
                                     abs(rng.normal(0, 0.5)) + 1e-3])

                return {"log_prior": log_prior, "sample": sample,
                        "names": names,
                        "description": f"informative(mu0={mu0:.2f})"}

        # --- Нормальная модель ---
        if self.distribution == "normal":
            if prior_type == "non_informative":
                def log_prior(params):
                    mu, sigma = params
                    if sigma <= 0 or not (-1e6 < mu < 1e6):
                        return -np.inf
                    return -np.log(sigma)  # Jeffreys

                def sample(rng):
                    return np.array([rng.normal(0, 1), abs(rng.normal(0, 1)) + 1e-3])
            else:
                def log_prior(params):
                    mu, sigma = params
                    if sigma <= 0:
                        return -np.inf
                    lp_mu = -0.5 * (mu / 10.0) ** 2
                    lp_sig = np.log(2 / (np.pi * (1 + sigma ** 2)))
                    return lp_mu + lp_sig

                def sample(rng):
                    return np.array([rng.normal(0, 10),
                                     abs(rng.standard_cauchy()) + 1e-3])

            return {"log_prior": log_prior, "sample": sample,
                    "names": names, "description": prior_type}

        # --- Гамма модель ---
        if self.distribution == "gamma":
            def log_prior(params):
                shape, scale = params
                if shape <= 0 or scale <= 0:
                    return -np.inf
                # Exp(1) для обоих — слабо информативное
                return -shape - scale

            def sample(rng):
                return np.array([rng.exponential(1.0) + 0.1,
                                 rng.exponential(1.0) + 0.1])

            return {"log_prior": log_prior, "sample": sample,
                    "names": names, "description": prior_type}

        raise RuntimeError("unreachable")

    # --------------------------------------------------------------
    # Likelihood
    # --------------------------------------------------------------
    def _log_likelihood(
        self,
        params: np.ndarray,
        uncensored: np.ndarray,
        lod: float,
        n_censored: int,
    ) -> float:
        """Лог-правдоподобие для цензурированных данных."""
        if self.distribution == "lognormal":
            mu, sigma = params
            if sigma <= 0:
                return -np.inf
            # Для нецензурированных: f(x) = (1/(x*sigma*sqrt(2pi))) * exp(-(log x - mu)^2 / (2 sigma^2))
            if len(uncensored) > 0:
                logx = np.log(uncensored)
                ll_uncens = np.sum(
                    -np.log(uncensored) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
                    - 0.5 * ((logx - mu) / sigma) ** 2
                )
            else:
                ll_uncens = 0.0
            # Цензурированные: F(LOD) = Phi((log LOD - mu)/sigma)
            if n_censored > 0 and lod > 0:
                z = (np.log(lod) - mu) / sigma
                logF = stats.norm.logcdf(z)
                ll_cens = n_censored * logF
            else:
                ll_cens = 0.0
            return ll_uncens + ll_cens

        if self.distribution == "normal":
            mu, sigma = params
            if sigma <= 0:
                return -np.inf
            if len(uncensored) > 0:
                ll_uncens = np.sum(stats.norm.logpdf(uncensored, mu, sigma))
            else:
                ll_uncens = 0.0
            if n_censored > 0:
                ll_cens = n_censored * stats.norm.logcdf(lod, mu, sigma)
            else:
                ll_cens = 0.0
            return ll_uncens + ll_cens

        if self.distribution == "gamma":
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return -np.inf
            if len(uncensored) > 0:
                ll_uncens = np.sum(
                    stats.gamma.logpdf(uncensored, a=shape, scale=scale)
                )
            else:
                ll_uncens = 0.0
            if n_censored > 0 and lod > 0:
                ll_cens = n_censored * stats.gamma.logcdf(
                    lod, a=shape, scale=scale
                )
            else:
                ll_cens = 0.0
            return ll_uncens + ll_cens

        return -np.inf

    def _log_posterior(
        self,
        params: np.ndarray,
        uncensored: np.ndarray,
        lod: float,
        n_censored: int,
        priors: dict,
    ) -> float:
        """Лог-апостериорное (пропорциональное)."""
        lp = priors["log_prior"](params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self._log_likelihood(params, uncensored, lod, n_censored)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # --------------------------------------------------------------
    # Helpers: parse censored_data
    # --------------------------------------------------------------
    @staticmethod
    def _parse_data(censored_data: dict) -> Tuple[np.ndarray, float, int, Optional[str]]:
        """Извлекает данные из словаря censored_data."""
        if "uncensored_values" in censored_data:
            uncens = np.asarray(censored_data["uncensored_values"], dtype=float)
        elif "values" in censored_data and "censored" in censored_data:
            vals = np.asarray(censored_data["values"], dtype=float)
            cens = np.asarray(censored_data["censored"], dtype=bool)
            uncens = vals[~cens]
        else:
            raise ValueError("censored_data must contain 'uncensored_values' or 'values'+'censored'")

        lod = float(censored_data.get("lod", censored_data.get("LOD", np.nan)))
        if "n_censored" in censored_data:
            n_cens = int(censored_data["n_censored"])
        elif "censored" in censored_data:
            n_cens = int(np.sum(censored_data["censored"]))
        else:
            n_cens = 0

        metal = censored_data.get("metal", None)
        return uncens, lod, n_cens, metal

    # --------------------------------------------------------------
    # Initial values via MLE / moments
    # --------------------------------------------------------------
    def _initial_values(
        self,
        uncensored: np.ndarray,
        lod: float,
        n_censored: int,
    ) -> np.ndarray:
        """Начальное приближение (простая MLE/моменты)."""
        if self.distribution == "lognormal":
            if len(uncensored) > 1:
                logs = np.log(np.clip(uncensored, 1e-10, None))
                mu0 = float(np.mean(logs))
                sigma0 = float(np.std(logs, ddof=1)) or 0.5
            else:
                mu0 = np.log(lod) if lod > 0 else 0.0
                sigma0 = 1.0
            return np.array([mu0, max(sigma0, 0.1)])

        if self.distribution == "normal":
            if len(uncensored) > 1:
                return np.array([float(np.mean(uncensored)),
                                 float(np.std(uncensored, ddof=1)) or 1.0])
            return np.array([lod / 2 if lod > 0 else 0.0, 1.0])

        if self.distribution == "gamma":
            if len(uncensored) > 1:
                m = float(np.mean(uncensored))
                v = float(np.var(uncensored, ddof=1)) or 1.0
                scale0 = v / m
                shape0 = m / scale0
                return np.array([max(shape0, 0.1), max(scale0, 1e-6)])
            return np.array([1.0, 1.0])

        return np.array([0.0, 1.0])

    # --------------------------------------------------------------
    # MCMC: Adaptive Metropolis-Hastings with data augmentation
    # --------------------------------------------------------------
    def _sample_augmented(
        self,
        params: np.ndarray,
        lod: float,
        n_censored: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Семплирует 'истинные' значения для цензурированных наблюдений."""
        if n_censored == 0:
            return np.array([])
        if self.distribution == "lognormal":
            mu, sigma = params
            # усечённое снизу lognormal на (0, lod)
            # эквивалентно: usechen Normal(mu, sigma) на (-inf, log(lod))
            upper = (np.log(lod) - mu) / sigma
            # inverse CDF method
            u = rng.uniform(0, stats.norm.cdf(upper), size=n_censored)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = stats.norm.ppf(u)
            return np.exp(mu + sigma * z)
        if self.distribution == "normal":
            mu, sigma = params
            upper = (lod - mu) / sigma
            u = rng.uniform(0, stats.norm.cdf(upper), size=n_censored)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = stats.norm.ppf(u)
            return mu + sigma * z
        if self.distribution == "gamma":
            shape, scale = params
            Fup = stats.gamma.cdf(lod, a=shape, scale=scale)
            u = rng.uniform(0, max(Fup, 1e-12), size=n_censored)
            return stats.gamma.ppf(u, a=shape, scale=scale)
        return np.array([])

    def _run_mh_chain(
        self,
        uncensored: np.ndarray,
        lod: float,
        n_censored: int,
        priors: dict,
        n_iterations: int,
        rng: np.random.Generator,
        init: np.ndarray,
        adapt_every: int = 100,
        target_accept: float = 0.234,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Одна цепь адаптивного Metropolis-Hastings.

        Returns
        -------
        samples : (n_iterations, n_params)
        imputed_samples : (n_iterations, n_censored)
        acceptance_rate : float
        """
        n_params = len(init)
        samples = np.zeros((n_iterations, n_params))
        imputed = np.zeros((n_iterations, max(n_censored, 1)))

        current = init.copy()
        current_lp = self._log_posterior(current, uncensored, lod, n_censored, priors)
        # Если старт плохой — попробуем ещё
        tries = 0
        while not np.isfinite(current_lp) and tries < 50:
            current = priors["sample"](rng)
            current_lp = self._log_posterior(
                current, uncensored, lod, n_censored, priors
            )
            tries += 1

        # Начальная ковариация proposal
        prop_scale = 0.1 * np.abs(current) + 0.01
        prop_cov = np.diag(prop_scale ** 2)

        n_accept = 0
        recent_accept = 0

        for i in range(n_iterations):
            # Данные-аугментация (используем imputed для улучшения смесей, но для LL используем формулу через F(LOD))
            if n_censored > 0:
                imp = self._sample_augmented(current, lod, n_censored, rng)
                imputed[i, :n_censored] = imp

            # Предложение
            try:
                proposal = rng.multivariate_normal(current, prop_cov)
            except Exception:
                proposal = current + rng.normal(0, prop_scale)

            prop_lp = self._log_posterior(
                proposal, uncensored, lod, n_censored, priors
            )

            if np.isfinite(prop_lp):
                log_alpha = prop_lp - current_lp
                if np.log(rng.uniform()) < log_alpha:
                    current = proposal
                    current_lp = prop_lp
                    n_accept += 1
                    recent_accept += 1

            samples[i] = current

            # Адаптация
            if (i + 1) % adapt_every == 0 and i < n_iterations // 2:
                rate = recent_accept / adapt_every
                recent_accept = 0
                # масштабный коэффициент
                if rate < target_accept:
                    prop_cov *= 0.9
                else:
                    prop_cov *= 1.1
                # Эмпирическая ковариация (если накопилось достаточно)
                if i > 200:
                    emp_cov = np.cov(samples[max(0, i - 500):i + 1].T)
                    if emp_cov.ndim == 0:
                        emp_cov = np.array([[float(emp_cov)]])
                    # регуляризация
                    emp_cov = emp_cov + 1e-8 * np.eye(n_params)
                    prop_cov = 0.5 * prop_cov + 0.5 * (2.38 ** 2 / n_params) * emp_cov

        if n_censored == 0:
            imputed = np.zeros((n_iterations, 0))
        else:
            imputed = imputed[:, :n_censored]

        return samples, imputed, n_accept / n_iterations

    # --------------------------------------------------------------
    # Convergence diagnostics
    # --------------------------------------------------------------
    @staticmethod
    def _gelman_rubin(chains: np.ndarray) -> np.ndarray:
        """R-hat по цепям. chains: (n_chains, n_iter, n_params)."""
        n_chains, n_iter, n_params = chains.shape
        if n_chains < 2:
            return np.full(n_params, np.nan)
        chain_means = chains.mean(axis=1)              # (chains, params)
        chain_vars = chains.var(axis=1, ddof=1)        # (chains, params)
        overall_mean = chain_means.mean(axis=0)
        B = n_iter * chain_means.var(axis=0, ddof=1)   # between
        W = chain_vars.mean(axis=0)                    # within
        var_hat = (1 - 1 / n_iter) * W + B / n_iter
        with np.errstate(invalid="ignore", divide="ignore"):
            r_hat = np.sqrt(var_hat / W)
        return r_hat

    @staticmethod
    def _effective_sample_size(x: np.ndarray) -> float:
        """ESS для одномерной цепи."""
        x = np.asarray(x)
        n = len(x)
        if n < 10:
            return float(n)
        x = x - x.mean()
        # автокорреляция через FFT
        f = np.fft.fft(x, n=2 * n)
        acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
        acf /= acf[0] if acf[0] > 0 else 1.0
        # Geyer's initial positive sequence
        tau = 1.0
        for k in range(1, n - 1):
            if acf[k] < 0.05:
                break
            tau += 2 * acf[k]
        tau = max(tau, 1.0)
        return float(n / tau)

    # --------------------------------------------------------------
    # Posterior summaries
    # --------------------------------------------------------------
    @staticmethod
    def _hdi(samples: np.ndarray, prob: float = 0.95) -> Tuple[float, float]:
        """Highest density interval."""
        s = np.sort(samples)
        n = len(s)
        k = int(np.floor(prob * n))
        if k < 1 or k >= n:
            return float(s[0]), float(s[-1])
        widths = s[k:] - s[:n - k]
        i = int(np.argmin(widths))
        return float(s[i]), float(s[i + k])

    def _summarize(self, samples: np.ndarray, names: List[str]) -> dict:
        """Сводка по апостериорным samples."""
        out = {}
        for i, name in enumerate(names):
            col = samples[:, i]
            lo, hi = self._hdi(col, 0.95)
            out[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col, ddof=1)),
                "median": float(np.median(col)),
                "hdi_95": (lo, hi),
            }
        return out

    # --------------------------------------------------------------
    # Derived quantities from posterior
    # --------------------------------------------------------------
    def _derived_quantities(
        self,
        posterior: np.ndarray,
        threshold: Optional[float] = None,
    ) -> dict:
        """Производные: среднее, перцентили, вероятность превышения."""
        means, variances, medians = [], [], []
        p75, p90, p95, p99 = [], [], [], []
        excs = []

        for params in posterior:
            if self.distribution == "lognormal":
                mu, sigma = params
                if sigma <= 0:
                    continue
                m = np.exp(mu + sigma ** 2 / 2)
                v = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
                med = np.exp(mu)
                p75.append(stats.lognorm.ppf(0.75, s=sigma, scale=np.exp(mu)))
                p90.append(stats.lognorm.ppf(0.90, s=sigma, scale=np.exp(mu)))
                p95.append(stats.lognorm.ppf(0.95, s=sigma, scale=np.exp(mu)))
                p99.append(stats.lognorm.ppf(0.99, s=sigma, scale=np.exp(mu)))
                if threshold is not None and threshold > 0:
                    excs.append(1 - stats.lognorm.cdf(
                        threshold, s=sigma, scale=np.exp(mu)))
            elif self.distribution == "normal":
                mu, sigma = params
                if sigma <= 0:
                    continue
                m, v, med = mu, sigma ** 2, mu
                p75.append(stats.norm.ppf(0.75, mu, sigma))
                p90.append(stats.norm.ppf(0.90, mu, sigma))
                p95.append(stats.norm.ppf(0.95, mu, sigma))
                p99.append(stats.norm.ppf(0.99, mu, sigma))
                if threshold is not None:
                    excs.append(1 - stats.norm.cdf(threshold, mu, sigma))
            elif self.distribution == "gamma":
                shape, scale = params
                if shape <= 0 or scale <= 0:
                    continue
                m = shape * scale
                v = shape * scale ** 2
                med = stats.gamma.median(a=shape, scale=scale)
                p75.append(stats.gamma.ppf(0.75, a=shape, scale=scale))
                p90.append(stats.gamma.ppf(0.90, a=shape, scale=scale))
                p95.append(stats.gamma.ppf(0.95, a=shape, scale=scale))
                p99.append(stats.gamma.ppf(0.99, a=shape, scale=scale))
                if threshold is not None:
                    excs.append(1 - stats.gamma.cdf(
                        threshold, a=shape, scale=scale))
            else:
                continue

            means.append(m)
            variances.append(v)
            medians.append(med)

        means = np.asarray(means)
        variances = np.asarray(variances)
        medians = np.asarray(medians)
        p75 = np.asarray(p75); p90 = np.asarray(p90)
        p95 = np.asarray(p95); p99 = np.asarray(p99)
        excs = np.asarray(excs) if len(excs) else np.array([np.nan])

        est_mean = float(np.mean(means))
        est_var = float(np.mean(variances))
        est_std = float(np.sqrt(est_var))
        est_median = float(np.mean(medians))

        return {
            "estimated_mean": est_mean,
            "estimated_variance": est_var,
            "estimated_std": est_std,
            "estimated_median": est_median,
            "estimated_percentiles": {
                75: float(np.mean(p75)),
                90: float(np.mean(p90)),
                95: float(np.mean(p95)),
                99: float(np.mean(p99)),
            },
            "estimated_exceedance_prob": float(np.mean(excs))
                if threshold is not None else float("nan"),
            "credible_intervals": {
                "mean_95hdi": self._hdi(means, 0.95),
                "p95_95hdi": self._hdi(p95, 0.95),
                "exceedance_95hdi": self._hdi(excs, 0.95)
                    if threshold is not None else (float("nan"), float("nan")),
            },
        }

    # --------------------------------------------------------------
    # MAIN: fit_mcmc
    # --------------------------------------------------------------
    def fit_mcmc(
        self,
        censored_data: dict,
        n_iterations: int = 5000,
        n_burn: int = 1000,
        n_chains: int = 4,
        thin: int = 2,
        seed: Optional[int] = None,
        threshold: Optional[float] = None,
        custom_priors: Optional[dict] = None,
    ) -> dict:
        """
        MCMC оценка апостериорного распределения.

        Parameters
        ----------
        censored_data : dict
            Данные с ключами: 'uncensored_values' | ('values', 'censored'),
            'lod' | 'LOD', 'n_censored' (опц.), 'metal' (опц.).
        n_iterations, n_burn, n_chains, thin : параметры MCMC
        seed : int, optional
        threshold : float, optional
            Порог для оценки вероятности превышения.
        custom_priors : dict, optional
            Для prior_type="custom".

        Returns
        -------
        dict (см. docstring модуля)
        """
        uncensored, lod, n_cens, metal = self._parse_data(censored_data)
        if threshold is None:
            threshold = censored_data.get("threshold", None)

        priors = self._set_priors(self.prior_type, metal=metal,
                                  custom_priors=custom_priors)

        self.logger.info(
            f"MCMC: n_uncens={len(uncensored)}, n_cens={n_cens}, lod={lod}, "
            f"prior={priors['description']}"
        )

        # --- Попытка использовать PyMC ---
        if HAS_PYMC and self.distribution == "lognormal" and self.prior_type != "custom":
            try:
                return self._fit_pymc_lognormal(
                    uncensored, lod, n_cens, metal,
                    n_iterations, n_burn, n_chains, thin,
                    seed, threshold, priors,
                )
            except Exception as e:
                self.logger.warning(f"PyMC failed, falling back to MH: {e}")

        # --- Собственная реализация MH ---
        base_seed = seed if seed is not None else 42
        all_samples = []
        all_imputed = []
        acc_rates = []

        for c in range(n_chains):
            rng = np.random.default_rng(base_seed + c)
            init = self._initial_values(uncensored, lod, n_cens)
            # варьируем начальные значения
            init = init + rng.normal(0, 0.1 * np.abs(init) + 0.01)
            samp, imp, acc = self._run_mh_chain(
                uncensored, lod, n_cens, priors,
                n_iterations, rng, init,
            )
            all_samples.append(samp)
            all_imputed.append(imp)
            acc_rates.append(acc)
            self.logger.info(f"  Chain {c + 1}/{n_chains}: acceptance={acc:.3f}")

        chains_full = np.stack(all_samples, axis=0)  # (n_chains, n_iter, n_params)

        # R-hat на всех итерациях (до burn-in): корректнее — на пост-burn-in
        post_burn = chains_full[:, n_burn:, :]
        # thinning
        post_burn_thin = post_burn[:, ::thin, :]

        # Flatten
        posterior = post_burn_thin.reshape(-1, post_burn_thin.shape[-1])

        # imputed: берём средние по пост-burn-in по каждой цензурированной точке
        if n_cens > 0:
            imp_all = np.concatenate(
                [im[n_burn::thin] for im in all_imputed], axis=0
            )
            imputed_values = imp_all.mean(axis=0)
        else:
            imputed_values = np.array([])

        # Diagnostics
        r_hat_arr = self._gelman_rubin(post_burn_thin)
        ess_arr = np.array([
            self._effective_sample_size(posterior[:, i])
            for i in range(posterior.shape[1])
        ])

        r_hat = {name: float(r_hat_arr[i]) for i, name in enumerate(priors["names"])}
        ess = {name: float(ess_arr[i]) for i, name in enumerate(priors["names"])}

        for name in priors["names"]:
            if r_hat[name] > 1.1:
                warnings.warn(
                    f"R-hat for {name} = {r_hat[name]:.3f} > 1.1 — "
                    "chains may not have converged. Consider more iterations."
                )
            if ess[name] < 100:
                warnings.warn(
                    f"ESS for {name} = {ess[name]:.1f} < 100 — "
                    "effective sample size is low."
                )

        summary = self._summarize(posterior, priors["names"])
        derived = self._derived_quantities(posterior, threshold=threshold)

        # DIC / WAIC
        dic = self._compute_dic(posterior, uncensored, lod, n_cens)
        waic = self._compute_waic(posterior, uncensored, lod, n_cens)

        result = {
            "method_name": f"bayesian_{self.distribution}",
            "posterior_samples": posterior,
            "param_names": priors["names"],
            "posterior_summary": summary,
            "convergence_diagnostics": {
                "r_hat": r_hat,
                "ess": ess,
                "acceptance_rate": float(np.mean(acc_rates)),
            },
            "imputed_values": imputed_values,
            "dic": dic,
            "waic": waic,
            # Для plot_diagnostics
            "_chains": post_burn_thin,
            "_full_chains": chains_full,
            "_n_burn": n_burn,
            "_thin": thin,
        }
        result.update(derived)

        self.logger.info(
            f"MCMC done. n_posterior_samples={posterior.shape[0]}, "
            f"mean R-hat={np.mean(list(r_hat.values())):.3f}"
        )
        return result

    # --------------------------------------------------------------
    # PyMC implementation (lognormal)
    # --------------------------------------------------------------
    def _fit_pymc_lognormal(
        self, uncensored, lod, n_cens, metal,
        n_iterations, n_burn, n_chains, thin,
        seed, threshold, priors,
    ) -> dict:
        """Реализация через PyMC."""
        self.logger.info("Using PyMC (NUTS) sampler")

        info = METAL_PRIOR_INFO.get(metal, METAL_PRIOR_INFO["default"])
        mu0 = info["mu0"]

        with pm.Model() as model:
            if self.prior_type == "non_informative":
                mu = pm.Uniform("mu_log", lower=-10, upper=10)
                sigma = pm.Uniform("sigma_log", lower=0.01, upper=5)
            elif self.prior_type == "informative":
                mu = pm.Normal("mu_log", mu=mu0, sigma=0.5)
                sigma = pm.HalfNormal("sigma_log", sigma=0.5)
            else:  # weakly_informative
                mu = pm.Normal("mu_log", mu=mu0, sigma=2.0)
                sigma = pm.HalfCauchy("sigma_log", beta=1.0)

            if len(uncensored) > 0:
                pm.Lognormal("obs", mu=mu, sigma=sigma, observed=uncensored)
            if n_cens > 0 and lod > 0:
                # Цензурирование: добавляем вклад n_cens * log F(lod)
                import pytensor.tensor as pt
                z = (pt.log(lod) - mu) / sigma
                logF = pm.logcdf(pm.Normal.dist(0, 1), z)
                pm.Potential("censored", n_cens * logF)

            idata = pm.sample(
                draws=n_iterations - n_burn,
                tune=n_burn,
                chains=n_chains,
                cores=1,
                random_seed=seed,
                progressbar=False,
                return_inferencedata=True,
            )

        post = idata.posterior
        mu_samples = post["mu_log"].values.reshape(-1)[::thin]
        sigma_samples = post["sigma_log"].values.reshape(-1)[::thin]
        posterior = np.column_stack([mu_samples, sigma_samples])

        # Диагностика
        summary_az = az.summary(idata, var_names=["mu_log", "sigma_log"])
        r_hat = {
            "mu_log": float(summary_az.loc["mu_log", "r_hat"]),
            "sigma_log": float(summary_az.loc["sigma_log", "r_hat"]),
        }
        ess = {
            "mu_log": float(summary_az.loc["mu_log", "ess_bulk"]),
            "sigma_log": float(summary_az.loc["sigma_log", "ess_bulk"]),
        }

        # Imputed values
        if n_cens > 0 and lod > 0:
            rng = np.random.default_rng(seed)
            imp = []
            for mu_s, sig_s in posterior:
                vals = self._sample_augmented(
                    np.array([mu_s, sig_s]), lod, n_cens, rng
                )
                imp.append(vals)
            imputed_values = np.mean(np.array(imp), axis=0)
        else:
            imputed_values = np.array([])

        summary = self._summarize(posterior, ["mu_log", "sigma_log"])
        derived = self._derived_quantities(posterior, threshold=threshold)

        chains_full = np.stack(
            [post["mu_log"].values, post["sigma_log"].values], axis=-1
        )  # (chains, iter, params)

        dic = self._compute_dic(posterior, uncensored, lod, n_cens)
        waic = self._compute_waic(posterior, uncensored, lod, n_cens)

        result = {
            "method_name": f"bayesian_{self.distribution}_pymc",
            "posterior_samples": posterior,
            "param_names": ["mu_log", "sigma_log"],
            "posterior_summary": summary,
            "convergence_diagnostics": {
                "r_hat": r_hat,
                "ess": ess,
                "acceptance_rate": float("nan"),
            },
            "imputed_values": imputed_values,
            "dic": dic,
            "waic": waic,
            "_chains": chains_full,
            "_full_chains": chains_full,
            "_n_burn": 0,
            "_thin": thin,
        }
        result.update(derived)
        return result

    # --------------------------------------------------------------
    # Информационные критерии
    # --------------------------------------------------------------
    def _compute_dic(
        self, posterior: np.ndarray,
        uncensored: np.ndarray, lod: float, n_cens: int,
    ) -> float:
        """DIC = D_bar + p_D, где p_D = D_bar - D(theta_bar)."""
        devs = np.array([
            -2 * self._log_likelihood(p, uncensored, lod, n_cens)
            for p in posterior
        ])
        devs = devs[np.isfinite(devs)]
        if len(devs) == 0:
            return float("nan")
        D_bar = float(np.mean(devs))
        theta_bar = posterior.mean(axis=0)
        D_theta_bar = -2 * self._log_likelihood(
            theta_bar, uncensored, lod, n_cens
        )
        if not np.isfinite(D_theta_bar):
            return float("nan")
        pD = D_bar - D_theta_bar
        return D_bar + pD

    def _compute_waic(
        self, posterior: np.ndarray,
        uncensored: np.ndarray, lod: float, n_cens: int,
    ) -> float:
        """
        Упрощённый WAIC: для каждого наблюдения считаем log-pointwise-predictive
        density, затем lppd - p_waic.
        """
        if len(uncensored) == 0:
            # для цензурированных считаем как группу
            return float("nan")

        # (n_posterior, n_obs) матрица log p(y_i | theta_s)
        n_s = posterior.shape[0]
        n_obs = len(uncensored)
        log_p = np.full((n_s, n_obs), -np.inf)
        for s, p in enumerate(posterior):
            if self.distribution == "lognormal":
                mu, sigma = p
                if sigma > 0:
                    log_p[s] = stats.lognorm.logpdf(
                        uncensored, s=sigma, scale=np.exp(mu)
                    )
            elif self.distribution == "normal":
                mu, sigma = p
                if sigma > 0:
                    log_p[s] = stats.norm.logpdf(uncensored, mu, sigma)
            elif self.distribution == "gamma":
                shape, scale = p
                if shape > 0 and scale > 0:
                    log_p[s] = stats.gamma.logpdf(
                        uncensored, a=shape, scale=scale
                    )

        # lppd
        with np.errstate(invalid="ignore"):
            lppd = np.sum(
                np.log(np.mean(np.exp(log_p - log_p.max(axis=0)), axis=0))
                + log_p.max(axis=0)
            )
            p_waic = np.sum(np.var(log_p, axis=0, ddof=1))
        return float(-2 * (lppd - p_waic))

    # --------------------------------------------------------------
    # Approximate (Laplace)
    # --------------------------------------------------------------
    def fit_approximate(
        self,
        censored_data: dict,
        n_samples: int = 2000,
        seed: Optional[int] = None,
        threshold: Optional[float] = None,
        custom_priors: Optional[dict] = None,
    ) -> dict:
        """Лапласовская аппроксимация апостериорного."""
        uncensored, lod, n_cens, metal = self._parse_data(censored_data)
        if threshold is None:
            threshold = censored_data.get("threshold", None)
        priors = self._set_priors(self.prior_type, metal=metal,
                                  custom_priors=custom_priors)

        init = self._initial_values(uncensored, lod, n_cens)

        def neg_log_post(p):
            v = self._log_posterior(p, uncensored, lod, n_cens, priors)
            return -v if np.isfinite(v) else 1e12

        res = minimize(neg_log_post, init, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
        map_est = res.x

        # Гессиан — численно
        eps = 1e-5
        n = len(map_est)
        H = np.zeros((n, n))
        f0 = neg_log_post(map_est)
        for i in range(n):
            for j in range(n):
                p_pp = map_est.copy(); p_pp[i] += eps; p_pp[j] += eps
                p_pm = map_est.copy(); p_pm[i] += eps; p_pm[j] -= eps
                p_mp = map_est.copy(); p_mp[i] -= eps; p_mp[j] += eps
                p_mm = map_est.copy(); p_mm[i] -= eps; p_mm[j] -= eps
                H[i, j] = (neg_log_post(p_pp) - neg_log_post(p_pm)
                           - neg_log_post(p_mp) + neg_log_post(p_mm)) / (4 * eps ** 2)
        H = 0.5 * (H + H.T) + 1e-6 * np.eye(n)

        try:
            cov = np.linalg.inv(H)
            # регуляризация, если отрицательные собственные значения
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 1e-10)
            cov = evecs @ np.diag(evals) @ evecs.T
        except np.linalg.LinAlgError:
            cov = np.eye(n) * 0.01

        rng = np.random.default_rng(seed)
        posterior = rng.multivariate_normal(map_est, cov, size=n_samples)

        summary = self._summarize(posterior, priors["names"])
        derived = self._derived_quantities(posterior, threshold=threshold)

        result = {
            "method_name": f"bayesian_{self.distribution}_laplace",
            "posterior_samples": posterior,
            "param_names": priors["names"],
            "posterior_summary": summary,
            "map_estimate": {priors["names"][i]: float(map_est[i])
                             for i in range(n)},
            "posterior_cov": cov,
            "convergence_diagnostics": {
                "r_hat": {name: float("nan") for name in priors["names"]},
                "ess": {name: float(n_samples) for name in priors["names"]},
                "acceptance_rate": float("nan"),
            },
            "imputed_values": np.array([]),
            "dic": float("nan"),
            "waic": float("nan"),
        }
        result.update(derived)
        return result

    # --------------------------------------------------------------
    # Posterior predictive
    # --------------------------------------------------------------
    def compute_posterior_predictive(
        self,
        posterior_samples: np.ndarray,
        n_predictions: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Апостериорное предиктивное распределение."""
        rng = np.random.default_rng(seed)
        n_post = len(posterior_samples)
        # выбираем n_predictions индексов (с возвращением если n_post < n_predictions)
        idx = rng.integers(0, n_post, size=n_predictions)
        preds = np.empty(n_predictions)
        for k, i in enumerate(idx):
            p = posterior_samples[i]
            if self.distribution == "lognormal":
                mu, sigma = p
                preds[k] = rng.lognormal(mu, max(sigma, 1e-6))
            elif self.distribution == "normal":
                mu, sigma = p
                preds[k] = rng.normal(mu, max(sigma, 1e-6))
            elif self.distribution == "gamma":
                shape, scale = p
                preds[k] = rng.gamma(max(shape, 1e-6), max(scale, 1e-6))
        return preds

    # --------------------------------------------------------------
    # Diagnostics plots
    # --------------------------------------------------------------
    def plot_diagnostics(self, result: dict):
        """Диагностические графики MCMC."""
        if not HAS_MPL:
            raise ImportError("matplotlib is required for plotting")

        chains = result.get("_chains")
        if chains is None:
            chains = result["posterior_samples"][np.newaxis, :, :]

        names = result["param_names"]
        n_params = len(names)

        fig, axes = plt.subplots(n_params, 4, figsize=(16, 3.5 * n_params),
                                 squeeze=False)

        for i, name in enumerate(names):
            # Traceplot
            ax = axes[i, 0]
            for c in range(chains.shape[0]):
                ax.plot(chains[c, :, i], alpha=0.7, lw=0.6,
                        label=f"chain {c + 1}")
            ax.set_title(f"Traceplot: {name}")
            ax.set_xlabel("iteration")
            ax.set_ylabel(name)
            if chains.shape[0] <= 4:
                ax.legend(fontsize=7, loc="best")

            # Histogram
            ax = axes[i, 1]
            flat = chains[:, :, i].reshape(-1)
            ax.hist(flat, bins=40, density=True, alpha=0.7, edgecolor="k")
            ax.axvline(np.mean(flat), color="red", ls="--", label="mean")
            ax.set_title(f"Posterior: {name}")
            ax.legend(fontsize=8)

            # Autocorrelation
            ax = axes[i, 2]
            x = chains[0, :, i] - chains[0, :, i].mean()
            n = len(x)
            maxlag = min(100, n // 4)
            acf = np.array([
                np.correlate(x[:n - k], x[k:])[0] / np.correlate(x, x)[0]
                for k in range(maxlag)
            ])
            ax.bar(range(maxlag), acf, width=1.0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"ACF: {name}")
            ax.set_xlabel("lag")

            # Running mean
            ax = axes[i, 3]
            for c in range(chains.shape[0]):
                rm = np.cumsum(chains[c, :, i]) / (np.arange(chains.shape[1]) + 1)
                ax.plot(rm, alpha=0.7, lw=0.8)
            ax.set_title(f"Running mean: {name}")
            ax.set_xlabel("iteration")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(2024)
    # Истинное логнормальное распределение: mu = -5, sigma = 0.8
    true_mu, true_sigma = -5.0, 0.8
    n = 100
    all_vals = rng.lognormal(true_mu, true_sigma, size=n)
    lod = 0.005
    censored_mask = all_vals < lod
    uncens_vals = all_vals[~censored_mask]
    n_cens = int(censored_mask.sum())

    print(f"\n--- Данные ---")
    print(f"Всего: {n}, цензурировано: {n_cens} ({100 * n_cens / n:.1f}%), "
          f"LOD = {lod}")

    data = {
        "uncensored_values": uncens_vals,
        "lod": lod,
        "n_censored": n_cens,
        "metal": "Pb",
        "threshold": 0.01,
    }

    # Байесовская модель
    bc = BayesianCensored(distribution="lognormal",
                          prior_type="weakly_informative")

    # 1. Быстрая аппроксимация
    print("\n--- Лапласовская аппроксимация ---")
    approx = bc.fit_approximate(data, n_samples=2000, seed=42,
                                threshold=0.01)
    print(f"MAP: {approx['map_estimate']}")
    print(f"Estimated mean: {approx['estimated_mean']:.5f}")
    print(f"Estimated P95:  {approx['estimated_percentiles'][95]:.5f}")
    print(f"P(X > 0.01):    {approx['estimated_exceedance_prob']:.4f}")

    # 2. Полный MCMC
    print("\n--- MCMC (Metropolis-Hastings) ---")
    res = bc.fit_mcmc(
        data,
        n_iterations=3000,
        n_burn=1000,
        n_chains=3,
        thin=2,
        seed=42,
        threshold=0.01,
    )

    print(f"\nPosterior summary:")
    for name, stats_d in res["posterior_summary"].items():
        print(f"  {name}: mean={stats_d['mean']:.4f}, "
              f"std={stats_d['std']:.4f}, "
              f"95% HDI = [{stats_d['hdi_95'][0]:.4f}, "
              f"{stats_d['hdi_95'][1]:.4f}]")
    print(f"\nTrue values: mu_log={true_mu}, sigma_log={true_sigma}")

    print(f"\nConvergence:")
    print(f"  R-hat: {res['convergence_diagnostics']['r_hat']}")
    print(f"  ESS:   {res['convergence_diagnostics']['ess']}")
    print(f"  Mean acceptance rate: "
          f"{res['convergence_diagnostics']['acceptance_rate']:.3f}")

    print(f"\nDerived quantities:")
    print(f"  E[X]      = {res['estimated_mean']:.5f}")
    print(f"  SD[X]     = {res['estimated_std']:.5f}")
    print(f"  Median    = {res['estimated_median']:.5f}")
    print(f"  P95       = {res['estimated_percentiles'][95]:.5f}")
    print(f"  P(X>0.01) = {res['estimated_exceedance_prob']:.4f}")
    print(f"  95% HDI for mean: {res['credible_intervals']['mean_95hdi']}")
    print(f"\nDIC  = {res['dic']:.2f}")
    print(f"WAIC = {res['waic']:.2f}")

    # 3. Posterior predictive
    pp = bc.compute_posterior_predictive(
        res["posterior_samples"], n_predictions=500, seed=1
    )
    print(f"\nPosterior predictive: mean={pp.mean():.5f}, "
          f"P95={np.percentile(pp, 95):.5f}")

    # 4. Диагностические графики
    if HAS_MPL:
        try:
            fig = bc.plot_diagnostics(res)
            fig.savefig("mcmc_diagnostics.png", dpi=90)
            print("\nДиагностические графики сохранены в mcmc_diagnostics.png")
        except Exception as e:
            print(f"Не удалось построить графики: {e}")