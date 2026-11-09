"""
Модуль визуализации результатов сравнения методов обработки цензурированных данных.

Создаёт все графики для проекта: от обзора исходных данных до итогового дашборда.
Поддерживает сохранение в PNG/PDF и интерактивную визуализацию через plotly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Переводы подписей (RU / EN)
# ---------------------------------------------------------------------------
LABELS: Dict[str, Dict[str, str]] = {
    "ru": {
        "concentration": "Концентрация",
        "frequency": "Частота",
        "density": "Плотность",
        "censoring_rate": "Доля цензурирования",
        "relative_bias": "Относительное смещение, %",
        "rmse": "RMSE",
        "method": "Метод",
        "mean": "Среднее",
        "variance": "Дисперсия",
        "median": "Медиана",
        "p90": "P90",
        "p95": "P95",
        "exceed": "P(X > ПДК), п.п.",
        "ks_stat": "KS-статистика",
        "coverage": "Вероятность покрытия",
        "ci_width": "Ширина ДИ",
        "true": "Истинное",
        "observed": "Наблюдаемое",
        "imputed": "Восстановленное",
        "lost": "Потерянные данные",
        "lod": "LOD",
        "mpc": "ПДК",
        "censored": "Цензурировано",
        "rank": "Ранг",
        "avg_rank": "Средний ранг",
        "overview_title": "Обзор исходных данных",
        "censoring_effect_title": "Эффект цензурирования на наблюдаемое распределение",
        "imputed_title": "Восстановленные распределения",
        "bias_title": "Сравнение смещений оценок",
        "bias_vs_censoring_title": "Зависимость смещения от доли цензурирования",
        "rmse_title": "RMSE методов",
        "ranking_title": "Рейтинги методов",
        "exceedance_title": "Оценка вероятности превышения ПДК",
        "ci_title": "Доверительные интервалы",
        "qq_title": "Q-Q plots: сравнение с логнормальным",
        "dashboard_title": "Итоговый дашборд: сравнение методов обработки цензурированных данных",
    },
    "en": {
        "concentration": "Concentration",
        "frequency": "Frequency",
        "density": "Density",
        "censoring_rate": "Censoring rate",
        "relative_bias": "Relative bias, %",
        "rmse": "RMSE",
        "method": "Method",
        "mean": "Mean",
        "variance": "Variance",
        "median": "Median",
        "p90": "P90",
        "p95": "P95",
        "exceed": "P(X > MPC), p.p.",
        "ks_stat": "KS statistic",
        "coverage": "Coverage probability",
        "ci_width": "CI width",
        "true": "True",
        "observed": "Observed",
        "imputed": "Imputed",
        "lost": "Lost data",
        "lod": "LOD",
        "mpc": "MPC",
        "censored": "Censored",
        "rank": "Rank",
        "avg_rank": "Avg. rank",
        "overview_title": "Original data overview",
        "censoring_effect_title": "Censoring effect on observed distribution",
        "imputed_title": "Imputed distributions",
        "bias_title": "Bias comparison",
        "bias_vs_censoring_title": "Bias vs. censoring rate",
        "rmse_title": "Methods RMSE",
        "ranking_title": "Methods ranking",
        "exceedance_title": "MPC exceedance probability estimation",
        "ci_title": "Confidence intervals",
        "qq_title": "Q-Q plots: comparison with lognormal",
        "dashboard_title": "Summary dashboard: comparison of censored data methods",
    },
}


DEFAULT_PALETTE: Dict[str, str] = {
    "deletion": "#808080",
    "zero_substitution": "#1f1f1f",
    "half_lod": "#d62728",
    "lod_substitution": "#8c564b",
    "lod_sqrt2": "#e377c2",
    "uniform_imputation": "#ff7f0e",
    "mean_imputation": "#bcbd22",
    "kaplan_meier": "#2ca02c",
    "robust_roS": "#17becf",
    "mle_lognormal": "#1f77b4",
    "mle_normal": "#9467bd",
    "regression_on_order": "#17a085",
    "multiple_imputation": "#4b0082",
}


class Visualizer:
    """Класс для создания всех визуализаций проекта."""

    def __init__(
        self,
        config: Optional[dict] = None,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Параметры
        ---------
        config : dict
            Конфигурация: цветовая палитра, язык, DPI, output_dir, metals, lods, mpc.
        style : str
            Стиль matplotlib.
        """
        self.config = config or {}
        try:
            plt.style.use(style)
        except OSError:
            logger.warning("Стиль %s недоступен, используется default", style)
            plt.style.use("default")

        self.palette: Dict[str, str] = {
            **DEFAULT_PALETTE,
            **self.config.get("palette", {}),
        }
        self.language: str = self.config.get("language", "ru")
        self.labels = LABELS[self.language]
        self.dpi: int = int(self.config.get("dpi", 150))
        self.save_dpi: int = int(self.config.get("save_dpi", 300))
        self.output_dir = Path(self.config.get("output_dir", "reports/figures"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "font.size": self.config.get("font_size", 11),
                "font.family": self.config.get("font_family", "DejaVu Sans"),
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "figure.dpi": self.dpi,
            }
        )

        logger.info(
            "Visualizer инициализирован (язык=%s, DPI=%d, output=%s)",
            self.language,
            self.dpi,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _color(self, method: str) -> str:
        return self.palette.get(method, "#333333")

    def _bias_color(self, bias_pct: float) -> str:
        a = abs(bias_pct)
        if a < 10:
            return "#2ca02c"
        if a < 20:
            return "#ff7f0e"
        return "#d62728"

    # ------------------------------------------------------------------
    # 1. Обзор исходных данных
    # ------------------------------------------------------------------
    def plot_original_data_overview(self, data: dict) -> Figure:
        """
        Обзор исходных данных по металлам.

        Параметры
        ---------
        data : dict
            {metal_name: {"values": np.array, "mpc": float}}
        """
        metals = list(data.keys())
        n_metals = len(metals)
        n_cols = min(5, n_metals)
        n_rows = int(np.ceil(n_metals / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for i, metal in enumerate(metals):
            ax = axes_flat[i]
            info = data[metal]
            values = np.asarray(info["values"]) if isinstance(info, dict) else np.asarray(info)
            values = values[~np.isnan(values)]
            mpc = info.get("mpc") if isinstance(info, dict) else None

            ax.hist(values, bins=40, density=True, alpha=0.6, color="#4C72B0", edgecolor="white")
            if len(values) > 1:
                try:
                    kde = stats.gaussian_kde(values)
                    xs = np.linspace(values.min(), values.max(), 200)
                    ax.plot(xs, kde(xs), color="#1f1f1f", lw=1.5)
                except Exception:
                    pass

            if mpc is not None:
                ax.axvline(mpc, color="red", linestyle="--", lw=1.2, label=self.labels["mpc"])

            ax.set_title(
                f"{metal}\nn={len(values)}, mean={values.mean():.3g}, med={np.median(values):.3g}",
                fontsize=10,
            )
            ax.set_xlabel(self.labels["concentration"])
            ax.set_ylabel(self.labels["density"])
            if mpc is not None:
                ax.legend(fontsize=8)

        for j in range(n_metals, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(self.labels["overview_title"], fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 2. Эффект цензурирования
    # ------------------------------------------------------------------
    def plot_censoring_effect(
        self,
        true_data: np.ndarray,
        censored_datasets: Dict[float, Dict[str, Any]],
    ) -> Figure:
        """
        Как цензурирование искажает видимую картину.

        Параметры
        ---------
        censored_datasets : dict
            {rate: {"observed": np.array, "lod": float}}
        """
        true_data = np.asarray(true_data)
        true_data = true_data[~np.isnan(true_data)]

        rates = sorted(censored_datasets.keys())
        n = len(rates)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        bins = np.linspace(true_data.min(), true_data.max(), 40)

        for i, rate in enumerate(rates):
            ax = axes_flat[i]
            ds = censored_datasets[rate]
            observed = np.asarray(ds["observed"])
            observed = observed[~np.isnan(observed)]
            lod = float(ds["lod"])

            ax.hist(true_data, bins=bins, color="lightgray", alpha=0.7,
                    label=self.labels["true"], edgecolor="white")
            ax.hist(observed, bins=bins, color="#4C72B0", alpha=0.75,
                    label=self.labels["observed"], edgecolor="white")

            below = true_data[true_data < lod]
            if len(below) > 0:
                ax.hist(below, bins=bins, color="red", alpha=0.25,
                        hatch="///", label=self.labels["lost"], edgecolor="red")

            ax.axvline(lod, color="black", linestyle="--", lw=1.3,
                       label=f"{self.labels['lod']}={lod:.3g}")
            ax.set_title(f"{self.labels['censored']}: {rate*100:.0f}%", fontsize=11)
            ax.set_xlabel(self.labels["concentration"])
            ax.set_ylabel(self.labels["frequency"])
            ax.legend(fontsize=7, loc="upper right")

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(self.labels["censoring_effect_title"], fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 3. Восстановленные распределения
    # ------------------------------------------------------------------
    def plot_imputed_distributions(
        self,
        true_data: np.ndarray,
        methods_results: Dict[str, Dict[str, Any]],
        censoring_rate: float,
    ) -> Figure:
        """
        Сравнение восстановленных распределений.

        Параметры
        ---------
        methods_results : dict
            {method_name: {"imputed": np.array, "ks_stat": float}}
        """
        true_data = np.asarray(true_data)
        true_data = true_data[~np.isnan(true_data)]

        methods = list(methods_results.keys())
        n = len(methods)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        bins = np.linspace(true_data.min(), true_data.max(), 40)

        for i, method in enumerate(methods):
            ax = axes_flat[i]
            res = methods_results[method]
            imputed = np.asarray(res.get("imputed", []))
            imputed = imputed[~np.isnan(imputed)]
            ks = res.get("ks_stat", None)

            ax.hist(true_data, bins=bins, density=True, color="lightgray",
                    alpha=0.6, label=self.labels["true"])
            color = self._color(method)
            if len(imputed) > 0:
                ax.hist(imputed, bins=bins, density=True, color=color,
                        alpha=0.55, label=self.labels["imputed"])

                try:
                    xs = np.linspace(bins.min(), bins.max(), 200)
                    ax.plot(xs, stats.gaussian_kde(true_data)(xs),
                            color="black", lw=1.2, alpha=0.7)
                    ax.plot(xs, stats.gaussian_kde(imputed)(xs),
                            color=color, lw=1.4)
                except Exception:
                    pass

            title = method
            if ks is not None:
                title += f"\nKS={ks:.3f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(self.labels["concentration"])
            ax.set_ylabel(self.labels["density"])
            ax.legend(fontsize=7)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"{self.labels['imputed_title']} ({censoring_rate*100:.0f}% {self.labels['censored'].lower()})",
            fontsize=15, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 4. Barplot смещений
    # ------------------------------------------------------------------
    def plot_bias_comparison_barplot(self, evaluation_results: pd.DataFrame) -> Figure:
        """
        Barplot смещений для 6 статистик.

        Параметры
        ---------
        evaluation_results : pd.DataFrame
            Колонки: method, bias_mean_pct, bias_var_pct, bias_median_pct,
                     bias_p90_pct, bias_p95_pct, error_exceed_pp
        """
        df = evaluation_results.copy()

        metrics = [
            ("bias_mean_pct", f"{self.labels['mean']} (%)"),
            ("bias_var_pct", f"{self.labels['variance']} (%)"),
            ("bias_median_pct", f"{self.labels['median']} (%)"),
            ("bias_p90_pct", f"{self.labels['p90']} (%)"),
            ("bias_p95_pct", f"{self.labels['p95']} (%)"),
            ("error_exceed_pp", self.labels["exceed"]),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()

        for ax, (col, title) in zip(axes_flat, metrics):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            sub = df[["method", col]].dropna().copy()
            sub["abs"] = sub[col].abs()
            sub = sub.sort_values("abs")
            colors = [self._bias_color(v) for v in sub[col].values]

            ax.barh(sub["method"], sub[col], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(10, color="orange", linestyle="--", lw=0.8, alpha=0.7)
            ax.axvline(-10, color="orange", linestyle="--", lw=0.8, alpha=0.7)
            ax.axvline(20, color="red", linestyle="--", lw=0.8, alpha=0.7)
            ax.axvline(-20, color="red", linestyle="--", lw=0.8, alpha=0.7)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel(self.labels["relative_bias"])

        legend_elems = [
            Patch(facecolor="#2ca02c", label="|bias| < 10%"),
            Patch(facecolor="#ff7f0e", label="10% ≤ |bias| < 20%"),
            Patch(facecolor="#d62728", label="|bias| ≥ 20%"),
        ]
        fig.legend(handles=legend_elems, loc="lower center",
                   ncol=3, bbox_to_anchor=(0.5, -0.01))

        fig.suptitle(self.labels["bias_title"], fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 5. Смещение vs доля цензурирования
    # ------------------------------------------------------------------
    def _extract_metric_series(
        self, full_results: dict, metric: str
    ) -> Dict[str, Dict[float, float]]:
        """
        Приводит full_results к структуре {method: {rate: value}} по метрике.

        full_results предполагается в формате:
          {rate: {method: {metric: value, ...}}}
        либо
          {method: {rate: {metric: value, ...}}}
        """
        out: Dict[str, Dict[float, float]] = {}
        if not full_results:
            return out

        # Определяем формат
        first_key = next(iter(full_results))
        first_val = full_results[first_key]
        if isinstance(first_key, (float, int)):
            # {rate: {method: {metric: val}}}
            for rate, methods in full_results.items():
                for method, metrics in methods.items():
                    if metric in metrics and metrics[metric] is not None:
                        out.setdefault(method, {})[float(rate)] = metrics[metric]
        else:
            # {method: {rate: {metric: val}}}
            for method, rates in full_results.items():
                for rate, metrics in rates.items():
                    if isinstance(metrics, dict) and metric in metrics and metrics[metric] is not None:
                        out.setdefault(method, {})[float(rate)] = metrics[metric]
        return out

    def plot_bias_vs_censoring_rate(self, full_results: dict) -> Figure:
        """
        КЛЮЧЕВОЙ ГРАФИК: смещение vs доля цензурирования.
        """
        metrics = [
            ("bias_mean_pct", f"{self.labels['mean']} (%)"),
            ("bias_var_pct", f"{self.labels['variance']} (%)"),
            ("bias_p90_pct", f"{self.labels['p90']} (%)"),
            ("bias_p95_pct", f"{self.labels['p95']} (%)"),
            ("error_exceed_pp", self.labels["exceed"]),
            ("ks_stat", self.labels["ks_stat"]),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()

        for ax, (metric, title) in zip(axes_flat, metrics):
            data = self._extract_metric_series(full_results, metric)
            if not data:
                ax.set_visible(False)
                continue

            for method, series in data.items():
                rates = sorted(series.keys())
                values = [series[r] for r in rates]
                ax.plot([r * 100 for r in rates], values,
                        marker="o", lw=1.6, color=self._color(method),
                        label=method, markersize=4)

            ax.axhline(0, color="black", lw=0.8)
            if metric != "ks_stat":
                ax.axhspan(-10, 10, color="green", alpha=0.08)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(7, len(labels)),
                       bbox_to_anchor=(0.5, -0.02), fontsize=9)

        fig.suptitle(self.labels["bias_vs_censoring_title"],
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 6. RMSE comparison
    # ------------------------------------------------------------------
    def plot_rmse_comparison(self, full_results: dict) -> Figure:
        """
        RMSE методов при разных долях цензурирования + heatmap.
        """
        rmse_data = self._extract_metric_series(full_results, "rmse_mean")
        if not rmse_data:
            rmse_data = self._extract_metric_series(full_results, "rmse")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        ax1 = axes[0]
        for method, series in rmse_data.items():
            rates = sorted(series.keys())
            values = [series[r] for r in rates]
            ax1.plot([r * 100 for r in rates], values,
                     marker="o", lw=1.6, color=self._color(method),
                     label=method, markersize=4)
        ax1.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax1.set_ylabel(self.labels["rmse"])
        ax1.set_title(f"{self.labels['rmse']} vs {self.labels['censoring_rate']}")
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Heatmap
        ax2 = axes[1]
        if rmse_data:
            methods = list(rmse_data.keys())
            all_rates = sorted({r for s in rmse_data.values() for r in s.keys()})
            matrix = np.full((len(methods), len(all_rates)), np.nan)
            for i, m in enumerate(methods):
                for j, r in enumerate(all_rates):
                    if r in rmse_data[m]:
                        matrix[i, j] = rmse_data[m][r]

            sns.heatmap(
                matrix, annot=True, fmt=".3g",
                xticklabels=[f"{r*100:.0f}%" for r in all_rates],
                yticklabels=methods, cmap="RdYlGn_r",
                cbar_kws={"label": self.labels["rmse"]}, ax=ax2,
            )
            ax2.set_xlabel(self.labels["censoring_rate"])
            ax2.set_ylabel(self.labels["method"])
            ax2.set_title(f"{self.labels['rmse']} heatmap")

        fig.suptitle(self.labels["rmse_title"], fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 7. Heatmap рейтингов
    # ------------------------------------------------------------------
    def plot_heatmap_rankings(self, full_results: dict) -> Figure:
        """
        Тепловая карта рейтингов методов по комбинациям (метрика × доля).
        """
        metrics = ["bias_mean_pct", "bias_var_pct",
                   "bias_p90_pct", "bias_p95_pct",
                   "error_exceed_pp", "ks_stat"]

        series_by_metric = {m: self._extract_metric_series(full_results, m) for m in metrics}

        all_methods = sorted({m for d in series_by_metric.values() for m in d.keys()})
        all_rates = sorted({r for d in series_by_metric.values()
                            for s in d.values() for r in s.keys()})
        if not all_methods or not all_rates:
            logger.warning("Нет данных для heatmap рейтингов")
            return plt.figure()

        col_tuples = []
        rank_matrix = np.full((len(all_methods), len(metrics) * len(all_rates)), np.nan)

        for mi, metric in enumerate(metrics):
            data = series_by_metric[metric]
            for ri, rate in enumerate(all_rates):
                col = mi * len(all_rates) + ri
                col_tuples.append(f"{metric}\n{rate*100:.0f}%")
                vals = {}
                for method in all_methods:
                    if method in data and rate in data[method]:
                        vals[method] = abs(data[method][rate])
                if not vals:
                    continue
                sorted_methods = sorted(vals, key=lambda x: vals[x])
                for rank, method in enumerate(sorted_methods, 1):
                    rank_matrix[all_methods.index(method), col] = rank

        avg_rank = np.nanmean(rank_matrix, axis=1).reshape(-1, 1)
        full_matrix = np.hstack([rank_matrix, avg_rank])
        col_labels = col_tuples + [self.labels["avg_rank"]]

        fig, ax = plt.subplots(figsize=(max(14, 0.4 * len(col_labels)),
                                        max(6, 0.4 * len(all_methods))))
        sns.heatmap(
            full_matrix, annot=True, fmt=".1f",
            xticklabels=col_labels, yticklabels=all_methods,
            cmap="RdYlGn_r", cbar_kws={"label": self.labels["rank"]},
            ax=ax, linewidths=0.4,
        )
        ax.set_title(self.labels["ranking_title"], fontsize=14, fontweight="bold")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 8. Exceedance probability
    # ------------------------------------------------------------------
    def plot_exceedance_probability_comparison(self, full_results: dict) -> Figure:
        """
        Оценки P(X > MPC) для методов при разных долях цензурирования.
        """
        data = self._extract_metric_series(full_results, "exceed_prob")
        true_prob = None
        # Пытаемся достать истинное значение
        if isinstance(next(iter(full_results.values()), None), dict):
            for v in full_results.values():
                if isinstance(v, dict) and "true_exceed_prob" in v:
                    true_prob = v["true_exceed_prob"]
                    break
                for m_v in v.values() if isinstance(v, dict) else []:
                    if isinstance(m_v, dict) and "true_exceed_prob" in m_v:
                        true_prob = m_v["true_exceed_prob"]
                        break

        fig, ax = plt.subplots(figsize=(12, 7))

        for method, series in data.items():
            rates = sorted(series.keys())
            values = [series[r] for r in rates]
            ax.plot([r * 100 for r in rates], values,
                    marker="o", lw=1.6, color=self._color(method),
                    label=method, markersize=5)

        if true_prob is not None:
            ax.axhline(true_prob, color="black", linestyle="--",
                       lw=1.4, label=f"{self.labels['true']}={true_prob:.3f}")
            ax.axhspan(true_prob * 0.8, true_prob * 1.2,
                       color="green", alpha=0.1, label="±20%")

        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel("P(X > MPC)")
        ax.set_title(self.labels["exceedance_title"], fontsize=14, fontweight="bold")
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 9. Confidence intervals
    # ------------------------------------------------------------------
    def plot_confidence_interval_comparison(
        self, full_results: dict, statistic: str = "mean"
    ) -> Figure:
        """
        Coverage и ширина ДИ для разных методов.

        full_results должен содержать поля ci_coverage_{stat} и ci_width_{stat}.
        """
        cov_key = f"ci_coverage_{statistic}"
        width_key = f"ci_width_{statistic}"

        cov_data = self._extract_metric_series(full_results, cov_key)
        width_data = self._extract_metric_series(full_results, width_key)

        # Усредняем по rate
        methods = sorted(set(cov_data.keys()) | set(width_data.keys()))
        cov_mean = [np.mean(list(cov_data.get(m, {0: np.nan}).values())) for m in methods]
        width_mean = [np.mean(list(width_data.get(m, {0: np.nan}).values())) for m in methods]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1, ax2 = axes

        colors = [self._color(m) for m in methods]
        ax1.bar(methods, cov_mean, color=colors, edgecolor="black", linewidth=0.5)
        ax1.axhline(0.95, color="red", linestyle="--", label="nominal 0.95")
        ax1.set_ylabel(self.labels["coverage"])
        ax1.set_title(f"Coverage probability ({statistic})")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(methods, width_mean, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_ylabel(self.labels["ci_width"])
        ax2.set_title(f"CI width ({statistic})")
        ax2.tick_params(axis="x", rotation=45)

        fig.suptitle(self.labels["ci_title"], fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # ------------------------------------------------------------------
    # 10. Q-Q plots
    # ------------------------------------------------------------------
    def plot_probability_plots_comparison(
        self,
        true_data: np.ndarray,
        methods_results: Dict[str, Dict[str, Any]],
    ) -> Figure:
        """
        Q-Q plots для каждого метода против теоретического логнормального.
        """
        methods = list(methods_results.keys())
        n = len(methods)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for i, method in enumerate(methods):
            ax = axes_flat[i]
            imputed = np.asarray(methods_results[method].get("imputed", []))
            imputed = imputed[(~np.isnan(imputed)) & (imputed > 0)]
            if len(imputed) < 5:
                ax.set_visible(False)
                continue

            try:
                shape, loc, scale = stats.lognorm.fit(imputed, floc=0)
                stats.probplot(
                    imputed, dist=stats.lognorm, sparams=(shape, loc, scale), plot=ax
                )
                ax.get_lines()[0].set_markerfacecolor(self._color(method))
                ax.get_lines()[0].set_markeredgecolor(self._color(method))
                ax.get_lines()[0].set_markersize(3)
                ax.get_lines()[1].set_color("black")
            except Exception as e:
                logger.warning("Q-Q plot ошибка для %s: %s", method, e)

            ax.set_title(method, fontsize=10)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(self.labels["qq_title"], fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # 11. Итоговый дашборд
    # ------------------------------------------------------------------
    def plot_summary_dashboard(self, full_results: dict) -> Figure:
        """
        6 ключевых визуализаций на одном рисунке.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # 1. Bias mean vs censoring rate
        ax = fig.add_subplot(gs[0, 0])
        data = self._extract_metric_series(full_results, "bias_mean_pct")
        for m, s in data.items():
            r = sorted(s.keys())
            ax.plot([x*100 for x in r], [s[x] for x in r],
                    marker="o", color=self._color(m), label=m, lw=1.4, markersize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-10, 10, color="green", alpha=0.08)
        ax.set_title(f"{self.labels['mean']} - {self.labels['relative_bias']}", fontweight="bold")
        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel(self.labels["relative_bias"])
        ax.legend(fontsize=7, ncol=2)

        # 2. Bias P95
        ax = fig.add_subplot(gs[0, 1])
        data = self._extract_metric_series(full_results, "bias_p95_pct")
        for m, s in data.items():
            r = sorted(s.keys())
            ax.plot([x*100 for x in r], [s[x] for x in r],
                    marker="o", color=self._color(m), label=m, lw=1.4, markersize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-10, 10, color="green", alpha=0.08)
        ax.set_title(f"{self.labels['p95']} - {self.labels['relative_bias']}", fontweight="bold")
        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel(self.labels["relative_bias"])

        # 3. RMSE
        ax = fig.add_subplot(gs[1, 0])
        data = self._extract_metric_series(full_results, "rmse_mean")
        if not data:
            data = self._extract_metric_series(full_results, "rmse")
        for m, s in data.items():
            r = sorted(s.keys())
            ax.plot([x*100 for x in r], [s[x] for x in r],
                    marker="o", color=self._color(m), label=m, lw=1.4, markersize=3)
        ax.set_title(self.labels["rmse_title"], fontweight="bold")
        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel(self.labels["rmse"])

        # 4. KS stat
        ax = fig.add_subplot(gs[1, 1])
        data = self._extract_metric_series(full_results, "ks_stat")
        for m, s in data.items():
            r = sorted(s.keys())
            ax.plot([x*100 for x in r], [s[x] for x in r],
                    marker="o", color=self._color(m), label=m, lw=1.4, markersize=3)
        ax.set_title(self.labels["ks_stat"], fontweight="bold")
        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel(self.labels["ks_stat"])

        # 5. Exceedance
        ax = fig.add_subplot(gs[2, 0])
        data = self._extract_metric_series(full_results, "exceed_prob")
        for m, s in data.items():
            r = sorted(s.keys())
            ax.plot([x*100 for x in r], [s[x] for x in r],
                    marker="o", color=self._color(m), label=m, lw=1.4, markersize=3)
        ax.set_title(self.labels["exceedance_title"], fontweight="bold")
        ax.set_xlabel(f"{self.labels['censoring_rate']}, %")
        ax.set_ylabel("P(X > MPC)")

        # 6. Avg bias (|bias| усреднённый)
        ax = fig.add_subplot(gs[2, 1])
        bias_mean = self._extract_metric_series(full_results, "bias_mean_pct")
        avg_abs_bias = {m: np.mean([abs(v) for v in s.values()])
                        for m, s in bias_mean.items()}
        sorted_methods = sorted(avg_abs_bias.keys(), key=lambda x: avg_abs_bias[x])
        values = [avg_abs_bias[m] for m in sorted_methods]
        colors = [self._bias_color(v) for v in values]
        ax.barh(sorted_methods, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Avg |bias mean|, %", fontweight="bold")
        ax.set_xlabel(self.labels["relative_bias"])

        fig.suptitle(self.labels["dashboard_title"],
                     fontsize=17, fontweight="bold", y=0.995)
        return fig

    # ------------------------------------------------------------------
    # 12. Сохранение всех графиков
    # ------------------------------------------------------------------
    def save_all_plots(
        self,
        full_results: dict,
        output_dir: Optional[str] = None,
        original_data: Optional[dict] = None,
        true_data: Optional[np.ndarray] = None,
        censored_datasets: Optional[dict] = None,
        methods_results: Optional[dict] = None,
        evaluation_results: Optional[pd.DataFrame] = None,
        example_censoring_rate: float = 0.3,
    ) -> List[str]:
        """
        Генерирует и сохраняет все доступные графики.

        Параметры
        ---------
        full_results : dict
            Полные результаты по всем методам и долям цензурирования.
        output_dir : str, optional
            Папка для сохранения. По умолчанию self.output_dir.
        original_data : dict, optional
            Для plot_original_data_overview.
        true_data : np.ndarray, optional
            Для censoring/imputed/qq plots.
        censored_datasets : dict, optional
            Для plot_censoring_effect.
        methods_results : dict, optional
            Для plot_imputed_distributions и Q-Q plots.
        evaluation_results : pd.DataFrame, optional
            Для plot_bias_comparison_barplot.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []

        def _save(fig: Figure, name: str):
            if fig is None:
                return
            png = out / f"{name}.png"
            pdf = out / f"{name}.pdf"
            fig.savefig(png, dpi=self.save_dpi, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(png))
            saved.append(str(pdf))
            logger.info("Сохранён график: %s", name)

        try:
            if original_data is not None:
                _save(self.plot_original_data_overview(original_data), "01_original_data_overview")
        except Exception as e:
            logger.exception("Ошибка 01: %s", e)

        try:
            if true_data is not None and censored_datasets is not None:
                _save(self.plot_censoring_effect(true_data, censored_datasets),
                      "02_censoring_effect")
        except Exception as e:
            logger.exception("Ошибка 02: %s", e)

        try:
            if true_data is not None and methods_results is not None:
                _save(self.plot_imputed_distributions(true_data, methods_results,
                                                     example_censoring_rate),
                      "03_imputed_distributions")
        except Exception as e:
            logger.exception("Ошибка 03: %s", e)

        try:
            if evaluation_results is not None:
                _save(self.plot_bias_comparison_barplot(evaluation_results),
                      "04_bias_barplot")
        except Exception as e:
            logger.exception("Ошибка 04: %s", e)

        try:
            _save(self.plot_bias_vs_censoring_rate(full_results),
                  "05_bias_vs_censoring_rate")
        except Exception as e:
            logger.exception("Ошибка 05: %s", e)

        try:
            _save(self.plot_rmse_comparison(full_results), "06_rmse_comparison")
        except Exception as e:
            logger.exception("Ошибка 06: %s", e)

        try:
            _save(self.plot_heatmap_rankings(full_results), "07_heatmap_rankings")
        except Exception as e:
            logger.exception("Ошибка 07: %s", e)

        try:
            _save(self.plot_exceedance_probability_comparison(full_results),
                  "08_exceedance_probability")
        except Exception as e:
            logger.exception("Ошибка 08: %s", e)

        try:
            _save(self.plot_confidence_interval_comparison(full_results, "mean"),
                  "09_confidence_intervals")
        except Exception as e:
            logger.exception("Ошибка 09: %s", e)

        try:
            if true_data is not None and methods_results is not None:
                _save(self.plot_probability_plots_comparison(true_data, methods_results),
                      "10_qq_plots")
        except Exception as e:
            logger.exception("Ошибка 10: %s", e)

        try:
            _save(self.plot_summary_dashboard(full_results), "11_summary_dashboard")
        except Exception as e:
            logger.exception("Ошибка 11: %s", e)

        logger.info("Всего сохранено файлов: %d", len(saved))
        return saved

    # ------------------------------------------------------------------
    # Интерактивная визуализация (plotly)
    # ------------------------------------------------------------------
    def to_plotly(self, full_results: dict, metric: str = "bias_mean_pct"):
        """
        Интерактивный график смещения vs доля цензурирования (plotly).
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly не установлен")
            return None

        data = self._extract_metric_series(full_results, metric)
        fig = go.Figure()
        for method, series in data.items():
            rates = sorted(series.keys())
            fig.add_trace(go.Scatter(
                x=[r * 100 for r in rates],
                y=[series[r] for r in rates],
                mode="lines+markers",
                name=method,
                line=dict(color=self._color(method), width=2),
            ))
        fig.update_layout(
            title=f"{metric} vs {self.labels['censoring_rate']}",
            xaxis_title=f"{self.labels['censoring_rate']}, %",
            yaxis_title=metric,
            template="plotly_white",
            hovermode="x unified",
        )
        return fig


# ---------------------------------------------------------------------------
# Демо
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Демо-запуск модуля визуализации")

    rng = np.random.default_rng(42)

    # Симулируем данные
    true_data = rng.lognormal(mean=0.5, sigma=0.8, size=1000)

    original_data = {
        f"Metal_{i}": {"values": rng.lognormal(0.5 + i*0.1, 0.7, 500),
                       "mpc": 2.5 + i * 0.3}
        for i in range(6)
    }

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    censored_datasets = {}
    for r in rates:
        lod = np.quantile(true_data, r)
        obs = true_data[true_data >= lod]
        censored_datasets[r] = {"observed": obs, "lod": lod}

    methods = list(DEFAULT_PALETTE.keys())
    methods_results = {}
    for m in methods:
        noise = rng.normal(0, 0.1 * (1 + methods.index(m) * 0.05), len(true_data))
        imputed = np.clip(true_data + noise, 0.01, None)
        ks, _ = stats.ks_2samp(true_data, imputed)
        methods_results[m] = {"imputed": imputed, "ks_stat": ks}

    # full_results: {rate: {method: {metric: value}}}
    full_results = {}
    for r in rates:
        full_results[r] = {}
        for i, m in enumerate(methods):
            bias_factor = (i - 6) * 0.5 * (1 + r)
            full_results[r][m] = {
                "bias_mean_pct": bias_factor * 3 + rng.normal(0, 1),
                "bias_var_pct": bias_factor * 5 + rng.normal(0, 2),
                "bias_median_pct": bias_factor * 2 + rng.normal(0, 1),
                "bias_p90_pct": bias_factor * 4 + rng.normal(0, 2),
                "bias_p95_pct": bias_factor * 5 + rng.normal(0, 2),
                "error_exceed_pp": bias_factor * 0.5 + rng.normal(0, 0.3),
                "ks_stat": abs(bias_factor) * 0.02 + 0.05,
                "rmse_mean": abs(bias_factor) * 0.1 + 0.5,
                "exceed_prob": 0.1 + bias_factor * 0.005,
                "true_exceed_prob": 0.1,
                "ci_coverage_mean": max(0.5, 0.95 - abs(bias_factor) * 0.02),
                "ci_width_mean": 0.5 + abs(bias_factor) * 0.05,
            }

    evaluation_results = pd.DataFrame([
        {"method": m,
         "bias_mean_pct": full_results[0.3][m]["bias_mean_pct"],
         "bias_var_pct": full_results[0.3][m]["bias_var_pct"],
         "bias_median_pct": full_results[0.3][m]["bias_median_pct"],
         "bias_p90_pct": full_results[0.3][m]["bias_p90_pct"],
         "bias_p95_pct": full_results[0.3][m]["bias_p95_pct"],
         "error_exceed_pp": full_results[0.3][m]["error_exceed_pp"]}
        for m in methods
    ])

    viz = Visualizer(config={"language": "ru", "output_dir": "reports/figures_demo"})
    saved = viz.save_all_plots(
        full_results=full_results,
        original_data=original_data,
        true_data=true_data,
        censored_datasets=censored_datasets,
        methods_results=methods_results,
        evaluation_results=evaluation_results,
        example_censoring_rate=0.3,
    )
    logger.info("Готово. Сохранено %d файлов.", len(saved))