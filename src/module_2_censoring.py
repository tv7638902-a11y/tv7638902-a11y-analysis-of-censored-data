"""
Модуль для искусственного создания цензурированных данных из полных наборов.

Этот модуль предоставляет инструменты для имитации различных типов цензурирования
данных о концентрациях тяжёлых металлов, что необходимо для последующей проверки
качества методов восстановления статистик из цензурированных выборок.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure


# Настройка логгера модуля
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class CensoringEngine:
    """
    Класс для искусственного создания цензурированных данных.
    
    Поддерживает несколько типов цензурирования:
      - Фиксированный LOD (один порог на всю выборку)
      - Переменный LOD (несколько порогов)
      - Случайный LOD (каждому наблюдению свой порог)
      - Интервальное цензурирование
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Инициализация движка цензурирования.
        
        Параметры:
            random_seed: Зерно для генератора случайных чисел (воспроизводимость).
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"CensoringEngine инициализирован с seed={random_seed}")
    
    # ------------------------------------------------------------------
    # 1. Фиксированный LOD
    # ------------------------------------------------------------------
    def apply_fixed_lod_censoring(
        self,
        data: np.ndarray,
        target_censoring_rate: float,
        method: str = "quantile",
        fixed_value: Optional[float] = None,
        mean_fraction: float = 0.1,
    ) -> dict:
        """
        Применяет единый фиксированный порог LOD ко всем наблюдениям.
        
        Параметры:
            data: Массив истинных значений.
            target_censoring_rate: Желаемая доля цензурированных значений [0.0, 0.95].
            method: Способ определения LOD:
                - "quantile":        LOD = квантиль, соответствующий target_censoring_rate
                - "fixed_value":     LOD задаётся через аргумент fixed_value
                - "fraction_of_mean":LOD = mean_fraction * mean(data)
            fixed_value: Значение LOD (используется если method='fixed_value').
            mean_fraction: Доля от среднего (используется если method='fraction_of_mean').
        
        Возвращает:
            Словарь с результатами цензурирования.
        """
        data = np.asarray(data, dtype=float).ravel()
        self.logger.info(
            f"Фиксированный LOD: method={method}, "
            f"target_rate={target_censoring_rate:.3f}, N={len(data)}"
        )
        
        if not 0.0 <= target_censoring_rate <= 0.95:
            raise ValueError("target_censoring_rate должен быть в [0.0, 0.95]")
        
        # Определяем значение LOD
        if method == "quantile":
            if target_censoring_rate == 0.0:
                lod_value = float(np.min(data) - 1e-12)
            else:
                lod_value = float(np.quantile(data, target_censoring_rate))
        elif method == "fixed_value":
            if fixed_value is None:
                raise ValueError("Для method='fixed_value' нужно указать fixed_value")
            lod_value = float(fixed_value)
        elif method == "fraction_of_mean":
            lod_value = float(mean_fraction * np.mean(data))
        else:
            raise ValueError(f"Неизвестный method: {method}")
        
        # Применяем цензурирование: value <= LOD -> цензурировано
        censored_flags = data <= lod_value
        observed_values = np.where(censored_flags, lod_value, data)
        
        actual_rate = float(np.mean(censored_flags))
        
        result = {
            "observed_values": observed_values,
            "censored_flags": censored_flags,
            "lod_value": lod_value,
            "actual_censoring_rate": actual_rate,
            "n_censored": int(np.sum(censored_flags)),
            "n_uncensored": int(np.sum(~censored_flags)),
            "true_values": data.copy(),
            "censoring_type": "fixed_lod",
        }
        
        self.logger.info(
            f"Фиксированный LOD готов: LOD={lod_value:.6g}, "
            f"actual_rate={actual_rate:.3f}, "
            f"n_censored={result['n_censored']}/{len(data)}"
        )
        return result
    
    # ------------------------------------------------------------------
    # 2. Переменный LOD (несколько уровней)
    # ------------------------------------------------------------------
    def apply_variable_lod_censoring(
        self,
        data: np.ndarray,
        target_censoring_rate: float,
        n_lod_levels: int = 3,
    ) -> dict:
        """
        Применяет несколько порогов LOD, имитируя данные из разных источников/лабораторий.
        
        Параметры:
            data: Массив истинных значений.
            target_censoring_rate: Желаемая общая доля цензурирования.
            n_lod_levels: Количество различных уровней LOD.
        
        Возвращает:
            Словарь с результатами + массив lod_per_observation.
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)
        self.logger.info(
            f"Переменный LOD: target_rate={target_censoring_rate:.3f}, "
            f"n_levels={n_lod_levels}, N={n}"
        )
        
        if n_lod_levels < 1:
            raise ValueError("n_lod_levels должен быть >= 1")
        
        # Базовый LOD как квантиль данных
        base_lod = float(np.quantile(data, target_censoring_rate)) if target_censoring_rate > 0 \
            else float(np.min(data) - 1e-12)
        
        # Создаём несколько уровней LOD вокруг базового (0.5x .. 1.5x)
        if n_lod_levels == 1:
            lod_levels = np.array([base_lod])
        else:
            factors = np.linspace(0.5, 1.5, n_lod_levels)
            lod_levels = base_lod * factors
        
        # Если нужна точная доля цензурирования — слегка корректируем LOD
        # через поиск общего множителя (простая итеративная корректировка)
        multiplier = 1.0
        for _ in range(20):
            lods_candidate = lod_levels * multiplier
            lod_assignment = self.rng.integers(0, n_lod_levels, size=n)
            lod_per_obs = lods_candidate[lod_assignment]
            cur_rate = float(np.mean(data <= lod_per_obs))
            if abs(cur_rate - target_censoring_rate) < 0.02:
                break
            # Регулируем в сторону цели
            if cur_rate < target_censoring_rate:
                multiplier *= 1.05
            else:
                multiplier *= 0.95
        
        lod_levels = lod_levels * multiplier
        # Финальное присвоение (с тем же seed-состоянием уже сдвинутым — переустановим)
        self.rng = np.random.default_rng(self.random_seed)
        lod_assignment = self.rng.integers(0, n_lod_levels, size=n)
        lod_per_observation = lod_levels[lod_assignment]
        
        censored_flags = data <= lod_per_observation
        observed_values = np.where(censored_flags, lod_per_observation, data)
        actual_rate = float(np.mean(censored_flags))
        
        result = {
            "observed_values": observed_values,
            "censored_flags": censored_flags,
            "lod_value": float(np.mean(lod_levels)),  # среднее для информации
            "lod_levels": lod_levels,
            "lod_per_observation": lod_per_observation,
            "actual_censoring_rate": actual_rate,
            "n_censored": int(np.sum(censored_flags)),
            "n_uncensored": int(np.sum(~censored_flags)),
            "true_values": data.copy(),
            "censoring_type": "variable_lod",
        }
        
        self.logger.info(
            f"Переменный LOD готов: уровни={np.round(lod_levels, 6)}, "
            f"actual_rate={actual_rate:.3f}"
        )
        return result
    
    # ------------------------------------------------------------------
    # 3. Случайный LOD
    # ------------------------------------------------------------------
    def apply_random_lod_censoring(
        self,
        data: np.ndarray,
        target_censoring_rate: float,
        lod_distribution: str = "uniform",
    ) -> dict:
        """
        Назначает каждому наблюдению случайный LOD из заданного распределения.
        
        Параметры:
            data: Массив истинных значений.
            target_censoring_rate: Желаемая доля цензурированных значений.
            lod_distribution: "uniform" | "normal" (усечённое).
        
        Возвращает:
            Словарь с результатами + lod_per_observation.
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)
        self.logger.info(
            f"Случайный LOD: distribution={lod_distribution}, "
            f"target_rate={target_censoring_rate:.3f}, N={n}"
        )
        
        if target_censoring_rate <= 0:
            # нет цензурирования
            lod_per_observation = np.full(n, np.min(data) - 1e-12)
            censored_flags = np.zeros(n, dtype=bool)
            observed_values = data.copy()
            return {
                "observed_values": observed_values,
                "censored_flags": censored_flags,
                "lod_value": float(lod_per_observation[0]),
                "lod_per_observation": lod_per_observation,
                "actual_censoring_rate": 0.0,
                "n_censored": 0,
                "n_uncensored": n,
                "true_values": data.copy(),
                "censoring_type": "random_lod",
            }
        
        # Центр распределения LOD = квантиль, соответствующий target rate
        center = float(np.quantile(data, target_censoring_rate))
        spread = max(center * 0.3, 1e-9)
        
        multiplier = 1.0
        best = None
        for _ in range(30):
            self.rng = np.random.default_rng(self.random_seed)
            c = center * multiplier
            if lod_distribution == "uniform":
                lod_per_observation = self.rng.uniform(
                    low=max(c - spread, 1e-12), high=c + spread, size=n
                )
            elif lod_distribution == "normal":
                lod_per_observation = self.rng.normal(loc=c, scale=spread, size=n)
                lod_per_observation = np.clip(lod_per_observation, 1e-12, None)
            else:
                raise ValueError(f"Неизвестный lod_distribution: {lod_distribution}")
            
            censored_flags = data <= lod_per_observation
            cur_rate = float(np.mean(censored_flags))
            best = (lod_per_observation, censored_flags, cur_rate)
            if abs(cur_rate - target_censoring_rate) < 0.02:
                break
            if cur_rate < target_censoring_rate:
                multiplier *= 1.05
            else:
                multiplier *= 0.95
        
        lod_per_observation, censored_flags, actual_rate = best
        observed_values = np.where(censored_flags, lod_per_observation, data)
        
        result = {
            "observed_values": observed_values,
            "censored_flags": censored_flags,
            "lod_value": float(np.mean(lod_per_observation)),
            "lod_per_observation": lod_per_observation,
            "actual_censoring_rate": actual_rate,
            "n_censored": int(np.sum(censored_flags)),
            "n_uncensored": int(np.sum(~censored_flags)),
            "true_values": data.copy(),
            "censoring_type": "random_lod",
        }
        
        self.logger.info(
            f"Случайный LOD готов: mean_LOD={result['lod_value']:.6g}, "
            f"actual_rate={actual_rate:.3f}"
        )
        return result
    
    # ------------------------------------------------------------------
    # 4. Интервальное цензурирование
    # ------------------------------------------------------------------
    def apply_interval_censoring(
        self,
        data: np.ndarray,
        n_intervals: int = 5,
    ) -> dict:
        """
        Создаёт интервальное цензурирование: каждое наблюдение известно лишь
        в пределах интервала [low, high], основанного на квантилях.
        
        Параметры:
            data: Массив истинных значений.
            n_intervals: Количество интервалов.
        
        Возвращает:
            Словарь с массивами границ интервалов.
        """
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)
        self.logger.info(f"Интервальное цензурирование: n_intervals={n_intervals}, N={n}")
        
        if n_intervals < 1:
            raise ValueError("n_intervals должен быть >= 1")
        
        quantile_edges = np.linspace(0, 1, n_intervals + 1)
        edges = np.quantile(data, quantile_edges)
        # Защита от повторяющихся границ (например, если все значения равны)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([data.min() - 1e-9, data.max() + 1e-9])
        
        # bin index для каждого наблюдения
        bin_idx = np.clip(np.searchsorted(edges, data, side="right") - 1, 0, len(edges) - 2)
        
        low_bounds = edges[bin_idx]
        high_bounds = edges[bin_idx + 1]
        
        result = {
            "low_bounds": low_bounds,
            "high_bounds": high_bounds,
            "interval_midpoints": (low_bounds + high_bounds) / 2.0,
            "edges": edges,
            "bin_indices": bin_idx,
            "true_values": data.copy(),
            "n_intervals": len(edges) - 1,
            "censoring_type": "interval",
        }
        
        self.logger.info(
            f"Интервальное цензурирование готово: "
            f"использовано {result['n_intervals']} интервалов"
        )
        return result
    
    # ------------------------------------------------------------------
    # 5. Полный набор цензурированных датасетов
    # ------------------------------------------------------------------
    def create_censored_dataset_suite(
        self,
        data: np.ndarray,
        censoring_rates: Optional[list] = None,
    ) -> dict:
        """
        Создаёт комплексный набор цензурированных датасетов.
        
        Для каждой доли цензурирования и каждого типа цензурирования создаётся
        один датасет.
        
        Параметры:
            data: Массив истинных значений.
            censoring_rates: Список долей (по умолчанию 0.1..0.8 шагом 0.1).
        
        Возвращает:
            Вложенный словарь {тип: {доля: результат}}.
        """
        if censoring_rates is None:
            censoring_rates = [round(x, 1) for x in np.arange(0.1, 0.81, 0.1)]
        
        self.logger.info(
            f"Создание suite: rates={censoring_rates}, N={len(data)}"
        )
        
        suite = {"fixed_lod": {}, "variable_lod": {}, "random_lod": {}}
        
        for rate in censoring_rates:
            # Переинициализация RNG для воспроизводимости каждого эксперимента
            self.rng = np.random.default_rng(self.random_seed)
            suite["fixed_lod"][rate] = self.apply_fixed_lod_censoring(
                data, target_censoring_rate=rate, method="quantile"
            )
            self.rng = np.random.default_rng(self.random_seed)
            suite["variable_lod"][rate] = self.apply_variable_lod_censoring(
                data, target_censoring_rate=rate, n_lod_levels=3
            )
            self.rng = np.random.default_rng(self.random_seed)
            suite["random_lod"][rate] = self.apply_random_lod_censoring(
                data, target_censoring_rate=rate, lod_distribution="uniform"
            )
        
        self.logger.info("Suite цензурированных датасетов создан.")
        return suite
    
    # ------------------------------------------------------------------
    # 6. Вспомогательные методы
    # ------------------------------------------------------------------
    def get_censoring_summary(self, censored_data: dict) -> pd.DataFrame:
        """
        Формирует сводную таблицу по цензурированному датасету.
        
        Параметры:
            censored_data: Результат одного из методов apply_*.
        
        Возвращает:
            pandas.DataFrame со сводкой.
        """
        observed = censored_data.get("observed_values")
        flags = censored_data.get("censored_flags")
        uncensored_vals = observed[~flags] if flags is not None else observed
        
        row = {
            "LOD": censored_data.get("lod_value", np.nan),
            "N_total": len(observed) if observed is not None else np.nan,
            "N_censored": censored_data.get("n_censored", np.nan),
            "Censoring_rate": censored_data.get("actual_censoring_rate", np.nan),
            "Min_observed": float(np.min(uncensored_vals)) if len(uncensored_vals) > 0 else np.nan,
            "Max_observed": float(np.max(uncensored_vals)) if len(uncensored_vals) > 0 else np.nan,
            "Mean_observed": float(np.mean(uncensored_vals)) if len(uncensored_vals) > 0 else np.nan,
            "Censoring_type": censored_data.get("censoring_type", "unknown"),
        }
        return pd.DataFrame([row])
    
    def visualize_censoring(
        self,
        true_data: np.ndarray,
        censored_data: dict,
        title: str = "",
    ) -> matplotlib.figure.Figure:
        """
        Визуализирует цензурирование.
        
        Параметры:
            true_data: Истинные данные.
            censored_data: Результат метода apply_*.
            title: Заголовок графика.
        
        Возвращает:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        observed = censored_data["observed_values"]
        flags = censored_data["censored_flags"]
        lod = censored_data.get("lod_value", np.nan)
        
        # Истинные данные
        ax.hist(true_data, bins=40, alpha=0.4, color="steelblue",
                label="Истинные данные", edgecolor="black")
        # Наблюдаемые (нецензурированные) значения
        ax.hist(observed[~flags], bins=40, alpha=0.6, color="green",
                label="Наблюдаемые (>LOD)", edgecolor="black")
        
        # Вертикальная линия LOD
        if np.isfinite(lod):
            ax.axvline(lod, color="red", linestyle="--", linewidth=2,
                       label=f"LOD = {lod:.4g}")
            ax.axvspan(ax.get_xlim()[0], lod, color="red", alpha=0.1,
                       label="Область цензурирования")
        
        rate = censored_data.get("actual_censoring_rate", np.nan)
        full_title = title or f"Цензурирование ({censored_data.get('censoring_type','')})"
        full_title += f"\nДоля цензурирования: {rate:.1%}"
        ax.set_title(full_title)
        ax.set_xlabel("Концентрация")
        ax.set_ylabel("Частота")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig
    
    def validate_censoring(self, censored_data: dict) -> bool:
        """
        Проверяет корректность цензурирования.
        
        Параметры:
            censored_data: Результат метода apply_*.
        
        Возвращает:
            True, если проверки пройдены.
        """
        try:
            observed = censored_data["observed_values"]
            flags = censored_data["censored_flags"]
            true_vals = censored_data["true_values"]
            
            # Нет NaN
            if np.any(np.isnan(observed)) or np.any(np.isnan(true_vals)):
                self.logger.error("Обнаружены NaN в данных!")
                return False
            
            # Пороги: массив или скаляр
            lod_per_obs = censored_data.get("lod_per_observation")
            if lod_per_obs is None:
                lod_per_obs = np.full_like(true_vals, censored_data["lod_value"])
            
            # Цензурированные значения: true <= LOD
            if np.any(true_vals[flags] > lod_per_obs[flags]):
                self.logger.error("Есть цензурированные значения с true > LOD!")
                return False
            
            # Нецензурированные: true > LOD
            if np.any(true_vals[~flags] <= lod_per_obs[~flags]):
                self.logger.error("Есть нецензурированные значения с true <= LOD!")
                return False
            
            self.logger.info("Валидация цензурирования пройдена успешно.")
            return True
        except KeyError as e:
            self.logger.error(f"Отсутствует ключ в censored_data: {e}")
            return False


# ======================================================================
# Демонстрация
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ МОДУЛЯ CENSORING ENGINE")
    print("=" * 70)
    
    # Генерируем "истинные" данные - логнормальное распределение (типично для концентраций)
    rng_demo = np.random.default_rng(123)
    true_data = rng_demo.lognormal(mean=-5.0, sigma=1.0, size=500)  # мг/л
    
    print(f"\nИсходные данные: N={len(true_data)}, "
          f"mean={true_data.mean():.5f}, median={np.median(true_data):.5f}")
    
    engine = CensoringEngine(random_seed=42)
    
    # --- 1. Фиксированный LOD ---
    print("\n[1] Фиксированный LOD (quantile, 30%)")
    r1 = engine.apply_fixed_lod_censoring(true_data, 0.3, method="quantile")
    print(f"  LOD={r1['lod_value']:.5f}, actual_rate={r1['actual_censoring_rate']:.3f}, "
          f"censored={r1['n_censored']}")
    print(f"  Validation: {engine.validate_censoring(r1)}")
    
    # --- 2. Переменный LOD ---
    print("\n[2] Переменный LOD (3 уровня, цель 40%)")
    r2 = engine.apply_variable_lod_censoring(true_data, 0.4, n_lod_levels=3)
    print(f"  LOD levels={np.round(r2['lod_levels'], 5)}")
    print(f"  actual_rate={r2['actual_censoring_rate']:.3f}")
    print(f"  Validation: {engine.validate_censoring(r2)}")
    
    # --- 3. Случайный LOD ---
    print("\n[3] Случайный LOD (uniform, цель 25%)")
    r3 = engine.apply_random_lod_censoring(true_data, 0.25, lod_distribution="uniform")
    print(f"  mean_LOD={r3['lod_value']:.5f}, actual_rate={r3['actual_censoring_rate']:.3f}")
    print(f"  Validation: {engine.validate_censoring(r3)}")
    
    # --- 4. Интервальное цензурирование ---
    print("\n[4] Интервальное цензурирование (5 интервалов)")
    r4 = engine.apply_interval_censoring(true_data, n_intervals=5)
    print(f"  Интервалы: {np.round(r4['edges'], 5)}")
    print(f"  n_intervals={r4['n_intervals']}")
    
    # --- 5. Полный набор ---
    print("\n[5] Полный набор цензурированных датасетов (rates 0.1..0.5)")
    suite = engine.create_censored_dataset_suite(
        true_data, censoring_rates=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    for ctype in suite:
        print(f"\n  --- {ctype} ---")
        for rate, res in suite[ctype].items():
            print(f"    target={rate:.1f} -> actual={res['actual_censoring_rate']:.3f}")
    
    # --- 6. Сводная таблица ---
    print("\n[6] Сводная таблица для фиксированного LOD (30%)")
    summary = engine.get_censoring_summary(r1)
    print(summary.to_string(index=False))
    
    # --- 7. Граничные случаи ---
    print("\n[7] Граничные случаи:")
    r_zero = engine.apply_fixed_lod_censoring(true_data, 0.0)
    print(f"  rate=0.0: n_censored={r_zero['n_censored']} (ожидалось 0)")
    r_high = engine.apply_fixed_lod_censoring(true_data, 0.95)
    print(f"  rate=0.95: actual_rate={r_high['actual_censoring_rate']:.3f}")
    
    # Данные с нулями
    data_with_zeros = np.concatenate([np.zeros(50), true_data])
    r_zeros = engine.apply_fixed_lod_censoring(data_with_zeros, 0.3)
    print(f"  Данные с нулями: actual_rate={r_zeros['actual_censoring_rate']:.3f}")
    
    # Одинаковые значения
    const_data = np.full(100, 0.01)
    r_const = engine.apply_fixed_lod_censoring(const_data, 0.5, method="quantile")
    print(f"  Константные данные: LOD={r_const['lod_value']:.5f}, "
          f"rate={r_const['actual_censoring_rate']:.3f}")
    
    # --- 8. Визуализация ---
    print("\n[8] Создаём визуализацию (фиксированный LOD)")
    fig = engine.visualize_censoring(true_data, r1, title="Демо: Фиксированный LOD")
    out_path = "censoring_demo.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Сохранено: {out_path}")
    
    # --- 9. Воспроизводимость ---
    print("\n[9] Проверка воспроизводимости (два запуска с одним seed)")
    e_a = CensoringEngine(random_seed=42)
    e_b = CensoringEngine(random_seed=42)
    a = e_a.apply_random_lod_censoring(true_data, 0.3)
    b = e_b.apply_random_lod_censoring(true_data, 0.3)
    reproducible = np.allclose(a["lod_per_observation"], b["lod_per_observation"])
    print(f"  Результаты идентичны: {reproducible}")
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
    print("=" * 70)