"""
ДОСЛІД E5. Програмна компенсація похибки вимірювальних каналів.

Ключове питання для практики: чи можна повернути втрачену точність обробкою
даних, не міняючи апаратуру. Якщо так — прилад лишається дешевим. Якщо ні —
жодна обробка не замінить нормального датчика, і це треба знати до того, як
пристрій піде в серію.

Порівнюються способи обробки, доступні розробникові приладу:

  без обробки              показання датчика як є
  медіанний фільтр         придушує викиди й одиничні збої
  експоненційне згладж.    придушує випадковий шум, але вносить запізнення
  калібрування за еталоном лінійна поправка, знайдена звірянням з образцовим
                           приладом у період пусконалагодження
  калібрування + фільтр    обидва заходи разом
  навчання з аугментацією  модель навчається на кількох реалізаціях похибки
                           і стає до неї стійкішою

Окремо показані дві межі: точність при ідеальних вимірюваннях (стільки можна
було б отримати з образцовою апаратурою) і точність без обробки.

ПРО КАЛІБРУВАННЯ. Поправка обчислюється ТІЛЬКИ за навчальною частиною вибірки —
так само, як у реальності її знайшли б при пусконалагодженні, звіривши прилад з
образцовим протягом обмеженого часу. Використовувати для цього контрольну
частину не можна: це був би витік інформації з майбутнього.

Запуск:  .venv/bin/python scripts/exp06_compensation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gims import config as cfg  # noqa: E402
from gims.features import chronological_split, make_supervised  # noqa: E402
from gims.metrics import evaluate  # noqa: E402
from gims.models import make_hist_gb  # noqa: E402
from gims.sensors import corrupt_frame  # noqa: E402

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"

HORIZON = 6
N_SEEDS = 3
N_AUGMENT = 4          # кількість реалізацій похибки при навчанні з аугментацією
MEDIAN_WINDOW = 5      # 25 хвилин — придушує викиди, не з'їдаючи динаміку
EMA_ALPHA = 0.4

ORDER = ["без обробки", "медіанний фільтр", "експоненційне згладжування",
         "калібрування за еталоном", "калібрування + медіанний фільтр",
         "навчання з аугментацією", "ідеальні вимірювання (межа)"]
SHORT = ["без обробки", "медіанний фільтр", "згладжування",
         "калібрування", "калібрування + фільтр", "навчання на варіантах",
         "ідеальні вимірювання"]


def apply_median(obs: pd.DataFrame) -> pd.DataFrame:
    """Ковзна медіана: знімає одиничні викиди й залиплі значення."""
    return obs.rolling(MEDIAN_WINDOW, center=False, min_periods=1).median()


def apply_ema(obs: pd.DataFrame) -> pd.DataFrame:
    """Експоненційне згладжування: давить шум ціною запізнення."""
    return obs.ewm(alpha=EMA_ALPHA, adjust=False).mean()


def fit_calibration(obs: pd.DataFrame, truth: pd.DataFrame, n_train: int) -> dict:
    """Знайти лінійну поправку за навчальною частиною (імітація пусконалагодження)."""
    coeffs = {}
    for c in obs.columns:
        x = obs[c].to_numpy(dtype=float)[:n_train]
        y = truth[c].to_numpy(dtype=float)[:n_train]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100 or np.std(x[ok]) == 0:
            coeffs[c] = (1.0, 0.0)
            continue
        a, b = np.polyfit(x[ok], y[ok], 1)
        coeffs[c] = (float(a), float(b))
    return coeffs


def apply_calibration(obs: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    out = obs.copy()
    for c, (a, b) in coeffs.items():
        if c in out.columns:
            out[c] = a * out[c] + b
    return out


def build_and_fit(ref: pd.DataFrame, observed: pd.DataFrame, target: str,
                  extra_train: list[pd.DataFrame] | None = None) -> float:
    """Зібрати вибірку, навчити модель, оцінити прогноз за істинними значеннями.

    extra_train — додаткові реалізації похибки; їхні навчальні частини
    дописуються до навчальної вибірки (навчання з аугментацією).
    """
    obs = ref.copy()
    for c in observed.columns:
        obs[c] = observed[c]
    data = make_supervised(obs, target, horizon=HORIZON, df_truth=ref)
    s = chronological_split(data)

    X_tr, y_tr = s["train"]["X"], s["train"]["y_obs"]

    if extra_train:
        Xs, ys = [X_tr], [y_tr]
        for other in extra_train:
            o = ref.copy()
            for c in other.columns:
                o[c] = other[c]
            d2 = make_supervised(o, target, horizon=HORIZON, df_truth=ref)
            s2 = chronological_split(d2)
            Xs.append(s2["train"]["X"])
            ys.append(s2["train"]["y_obs"])
        X_tr = pd.concat(Xs, axis=0)
        y_tr = pd.concat(ys, axis=0)

    model = make_hist_gb(seed=cfg.SEED)
    model.fit(X_tr, y_tr, s["val"]["X"], s["val"]["y_obs"])
    return evaluate(s["test"]["y_true"], model.predict(s["test"]["X"]))["RMSE"]


def run(ref: pd.DataFrame) -> pd.DataFrame:
    device_cols = list(cfg.CHANNELS)
    n_train = int(len(ref) * cfg.SPLIT_TRAIN)
    rows = []

    for target in cfg.TARGETS:
        print(f"\n  Канал {target}")
        results: dict[str, list[float]] = {}

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(cfg.SEED + seed)
            obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, scale=1.0)
            obs = obs.ffill().bfill()

            coeffs = fit_calibration(obs, ref[device_cols], n_train)

            variants = {
                "без обробки": obs,
                "медіанний фільтр": apply_median(obs),
                "експоненційне згладжування": apply_ema(obs),
                "калібрування за еталоном": apply_calibration(obs, coeffs),
                "калібрування + медіанний фільтр":
                    apply_median(apply_calibration(obs, coeffs)),
            }
            for label, variant in variants.items():
                results.setdefault(label, []).append(
                    build_and_fit(ref, variant, target))

            # Навчання з аугментацією: показуємо моделі кілька різних
            # реалізацій похибки, щоб вона не підлаштовувалася під одну
            extra = []
            for k in range(1, N_AUGMENT):
                r2 = np.random.default_rng(cfg.SEED + 1000 * k + seed)
                extra.append(corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, r2,
                                           scale=1.0).ffill().bfill())
            results.setdefault("навчання з аугментацією", []).append(
                build_and_fit(ref, obs, target, extra_train=extra))

            # Межа: ідеальні вимірювання
            results.setdefault("ідеальні вимірювання (межа)", []).append(
                build_and_fit(ref, ref[device_cols], target))

        base = float(np.mean(results["без обробки"]))
        ideal = float(np.mean(results["ідеальні вимірювання (межа)"]))
        for label, vals in results.items():
            val = float(np.mean(vals))
            # Яка частка втраченої точності відіграна обробкою
            recovered = (base - val) / (base - ideal) * 100 if base > ideal else np.nan
            rows.append({
                "канал": target,
                "од. вим.": cfg.CHANNELS[target]["unit"],
                "спосіб обробки": label,
                "RMSE": round(val, 3),
                "СКВ за прогонами": round(float(np.std(vals)), 3),
                "покращення до «без обробки»,%": round((1 - val / base) * 100, 1),
                "відіграно втрат,%": round(float(recovered), 1),
            })
            print(f"    {label:32} RMSE={val:8.3f}  "
                  f"відіграно={recovered:6.1f}% втрат")

    return pd.DataFrame(rows)


def figure(res: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cfg.TARGETS), figsize=(5.4 * len(cfg.TARGETS), 5.4))
    for ax, target in zip(axes, cfg.TARGETS):
        s = res[res["канал"] == target].set_index("спосіб обробки").loc[ORDER]
        colors = ["#8fa8bf"] * len(ORDER)
        colors[0] = "#c62828"          # вихідний стан
        colors[-1] = "#2e7d32"         # недосяжна межа
        best = s.iloc[1:-1]["RMSE"].idxmin()
        colors[ORDER.index(best)] = "#1f4e79"

        ax.bar(range(len(ORDER)), s["RMSE"], color=colors,
               yerr=s["СКВ за прогонами"], capsize=3)
        ax.axhline(float(s.loc["ідеальні вимірювання (межа)", "RMSE"]),
                   color="#2e7d32", ls="--", lw=1.2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(SHORT, fontsize=8, rotation=38, ha="right")
        meta = cfg.CHANNELS[target]
        ax.set_ylabel(f"RMSE прогнозу, {meta['unit']}")
        ax.set_title(meta["descr"], fontsize=10)
        ax.grid(axis="y", alpha=0.3, lw=0.5)

    fig.suptitle("E5. Чи можна повернути точність програмно "
                 "(зелена лінія — межа при ідеальних вимірюваннях)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E5. Програмна компенсація похибки")
    print("=" * 78)
    print(f"Горизонт {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв, {N_SEEDS} прогони")

    res = run(ref)
    res.to_csv(cfg.TABLES / "table07_compensation.csv", index=False)

    print()
    print("=" * 78)
    print("ВИСНОВКИ")
    print("=" * 78)
    for target in cfg.TARGETS:
        s = res[(res["канал"] == target) &
                (~res["спосіб обробки"].isin(
                    ["без обробки", "ідеальні вимірювання (межа)"]))]
        best = s.loc[s["RMSE"].idxmin()]
        print(f"  {target:8} найкращий спосіб: {best['спосіб обробки']:32} "
              f"відіграно {best['відіграно втрат,%']:5.1f} % втраченої точності")

    fig_path = cfg.FIGURES / "fig09_compensation.png"
    figure(res, fig_path)
    print()
    print(f"Таблиця: {(cfg.TABLES / 'table07_compensation.csv').relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
