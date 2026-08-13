"""
ДОСЛІД E2-CO2. Обґрунтування вибору датчика концентрації CO2.

Попередній аналіз показав, що канал CO2 на датчику SGP30 непридатний:
відношення сигнал/похибка менше одиниці. Однак цей висновок спирається на
гіпотезу про поведінку металооксидного датчика в теплиці (коефіцієнт
measurand_gain = 0.5), а не на паспортні дані. Гіпотезу треба перевірити,
інакше весь висновок повисає в повітрі.

Дослід складається з двох частин.

ЧАСТИНА А. Аналіз чутливості.
Коефіцієнт measurand_gain змінюється в широких межах — від украй
песимістичного 0.1 до оптимістичного 1.0 (датчик ідеально стежить за CO2).
Якщо висновок про непридатність зберігається на всьому діапазоні, значить він
не залежить від спірного припущення і обґрунтований.

ЧАСТИНА Б. Порівняння з альтернативами.
Канал CO2 почергово реалізується на SGP30, MH-Z19B і SCD30. Порівнюється
підсумкова точність прогнозування концентрації CO2. Це дає кількісне
обґрунтування рекомендації щодо вибору датчика.

Запуск:  .venv/bin/python scripts/exp02_co2_sensor.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
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
from gims.sensors import CO2_SENSOR_OPTIONS, SGP30_CO2, corrupt_frame  # noqa: E402

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"

TARGET = "CO2_air"
HORIZON = 6          # 30 хвилин — типова випереджальна уставка для дозування CO2
GAINS = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
N_SEEDS = 3


def channel_snr(truth: pd.Series, meas: pd.Series) -> tuple[float, float]:
    """Відношення сигнал/похибка каналу: як є і після калібрування."""
    t = truth.to_numpy(dtype=float)
    m = meas.to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(m)
    err = m[ok] - t[ok]
    snr = float(np.std(t[ok]) / np.std(err)) if np.std(err) > 0 else np.inf
    r = float(np.corrcoef(t[ok], m[ok])[0, 1])
    snr_cal = 1.0 / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
    return snr, snr_cal


def forecast_rmse(ref: pd.DataFrame, observed: pd.DataFrame) -> tuple[float, float]:
    """Навчити модель на показаннях датчика, оцінити прогноз за істиною.

    Повертає (RMSE прогнозу, виграш над наївним прогнозом у відсотках).
    """
    # Прилад бачить лише свої показання; решта стовпців (погода, виконавчі
    # механізми) доступні системі керування теплицею як є
    obs = ref.copy()
    for c in observed.columns:
        obs[c] = observed[c]

    data = make_supervised(obs, TARGET, horizon=HORIZON, df_truth=ref)
    s = chronological_split(data)

    model = make_hist_gb(seed=cfg.SEED)
    model.fit(s["train"]["X"], s["train"]["y_obs"], s["val"]["X"], s["val"]["y_obs"])
    pred = model.predict(s["test"]["X"])
    m = evaluate(s["test"]["y_true"], pred, s["test"]["y_naive"])
    return m["RMSE"], m["виграш над наївним,%"]


def part_a(ref: pd.DataFrame) -> pd.DataFrame:
    """Чутливість висновку до гіпотези про коефіцієнт measurand_gain."""
    print("\nЧАСТИНА А. Аналіз чутливості до гіпотези measurand_gain")
    print("-" * 78)
    rows = []
    for gain in GAINS:
        spec = replace(SGP30_CO2, measurand_gain=gain)
        snrs, rmses, skills = [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(cfg.SEED + seed)
            obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng,
                                overrides={TARGET: spec})
            snr, _ = channel_snr(ref[TARGET], obs[TARGET])
            rmse, skill = forecast_rmse(ref, obs)
            snrs.append(snr)
            rmses.append(rmse)
            skills.append(skill)

        row = {
            "measurand_gain": gain,
            "сигнал/похибка": round(float(np.mean(snrs)), 2),
            "RMSE прогнозу, ppm": round(float(np.mean(rmses)), 1),
            "СКВ за прогонами": round(float(np.std(rmses)), 1),
            "виграш над наївним,%": round(float(np.mean(skills)), 1),
        }
        rows.append(row)
        print(f"  gain={gain:4.2f}  SNR={row['сигнал/похибка']:5.2f}  "
              f"RMSE={row['RMSE прогнозу, ppm']:7.1f} ppm  "
              f"виграш={row['виграш над наївним,%']:6.1f}%")
    return pd.DataFrame(rows)


def part_b(ref: pd.DataFrame) -> pd.DataFrame:
    """Порівняння варіантів виконання каналу CO2."""
    print("\nЧАСТИНА Б. Порівняння датчиків концентрації CO2")
    print("-" * 78)
    rows = []

    # Верхня межа досяжного: ідеальний вимірювальний канал
    rmse_ideal, skill_ideal = forecast_rmse(ref, ref[[TARGET]])
    rows.append({
        "датчик": "ідеальний канал (межа)",
        "принцип": "—",
        "сигнал/похибка": np.inf,
        "після калібрування": np.inf,
        "RMSE прогнозу, ppm": round(rmse_ideal, 1),
        "СКВ за прогонами": 0.0,
        "виграш над наївним,%": round(skill_ideal, 1),
    })
    print(f"  {'ідеальний канал (межа)':26} RMSE={rmse_ideal:7.1f} ppm")

    for label, spec in CO2_SENSOR_OPTIONS.items():
        snrs, snrs_cal, rmses, skills = [], [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(cfg.SEED + seed)
            obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng,
                                overrides={TARGET: spec})
            snr, snr_cal = channel_snr(ref[TARGET], obs[TARGET])
            rmse, skill = forecast_rmse(ref, obs)
            snrs.append(snr)
            snrs_cal.append(snr_cal)
            rmses.append(rmse)
            skills.append(skill)

        rows.append({
            "датчик": label,
            "принцип": "MOX" if "SGP30" in label else "NDIR",
            "сигнал/похибка": round(float(np.mean(snrs)), 2),
            "після калібрування": round(float(np.mean(snrs_cal)), 2),
            "RMSE прогнозу, ppm": round(float(np.mean(rmses)), 1),
            "СКВ за прогонами": round(float(np.std(rmses)), 1),
            "виграш над наївним,%": round(float(np.mean(skills)), 1),
        })
        print(f"  {label:26} SNR={np.mean(snrs):5.2f}  "
              f"RMSE={np.mean(rmses):7.1f} ppm  "
              f"виграш={np.mean(skills):6.1f}%")

    return pd.DataFrame(rows)


def figure(res_a: pd.DataFrame, res_b: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Ліва панель: стійкість висновку до спірного припущення
    ax1.plot(res_a["measurand_gain"], res_a["сигнал/похибка"],
             marker="o", lw=1.8, color="#1f4e79")
    ax1.axhline(1, color="#c62828", ls="--", lw=1.2)
    ax1.text(0.10, 0.985, "поріг непридатності каналу", fontsize=8,
             color="#c62828", va="top", ha="left")
    ax1.fill_between([0.05, 1.05], 0, 1, color="#c62828", alpha=0.07)
    ax1.set_xlim(0.05, 1.05)
    ax1.set_ylim(0, 1.12)
    ax1.set_xlabel("коефіцієнт measurand_gain (гіпотеза)")
    ax1.set_ylabel("відношення сигнал / похибка")
    ax1.set_title("А. Стійкість висновку до прийнятого припущення", fontsize=10)
    ax1.grid(alpha=0.3, lw=0.5)

    # Права панель: порівняння датчиків
    sub = res_b[res_b["датчик"] != "ідеальний канал (межа)"]
    labels = [s.split(" (")[0] for s in sub["датчик"]]
    colors = ["#c62828" if p == "MOX" else "#2e7d32" for p in sub["принцип"]]
    vals = sub["RMSE прогнозу, ppm"].to_numpy(dtype=float)
    errs = sub["СКВ за прогонами"].to_numpy(dtype=float)
    bars = ax2.bar(labels, vals, color=colors, yerr=errs, capsize=4)

    ideal = float(res_b.loc[res_b["датчик"] == "ідеальний канал (межа)",
                            "RMSE прогнозу, ppm"].iloc[0])
    ax2.axhline(ideal, color="#1f4e79", ls="--", lw=1.4,
                label=f"межа при ідеальному каналі ({ideal:.0f} ppm)")
    ax2.set_ylim(0, float(np.max(vals + errs)) * 1.34)
    ax2.set_ylabel("RMSE прогнозу CO$_2$ на 30 хв, ppm")
    ax2.set_title("Б. Точність прогнозу з різними датчиками", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, lw=0.5)
    ax2.legend(fontsize=8, loc="upper right")
    # Підписи ставимо вище верхнього кінця планки похибки, інакше накладуться
    for b, v, e in zip(bars, vals, errs):
        ax2.text(b.get_x() + b.get_width() / 2, v + e + float(np.max(vals)) * 0.045,
                 f"{v:.0f} ppm", ha="center", fontsize=9)

    fig.suptitle("Обґрунтування вибору датчика концентрації CO₂", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E2-CO2. Обґрунтування вибору датчика концентрації CO2")
    print("=" * 78)
    print(f"Канал: {TARGET}, горизонт прогнозу {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв, "
          f"{N_SEEDS} незалежні прогони")

    res_a = part_a(ref)
    res_b = part_b(ref)

    res_a.to_csv(cfg.TABLES / "table03a_co2_sensitivity.csv", index=False)
    res_b.to_csv(cfg.TABLES / "table03b_co2_sensor_choice.csv", index=False)

    print()
    print("=" * 78)
    print("ВИСНОВКИ")
    print("=" * 78)

    worst_snr = res_a["сигнал/похибка"].max()
    if worst_snr < 2:
        print("А. Навіть за найсприятливішого припущення (measurand_gain = 1.0)")
        print(f"   відношення сигнал/похибка не перевищує {worst_snr:.2f}.")
        print("   Висновок про непридатність SGP30 не залежить від гіпотези.")
    else:
        print(f"А. При measurand_gain від {GAINS[0]} до {GAINS[-1]} відношення")
        print(f"   сигнал/похибка змінюється від {res_a['сигнал/похибка'].min():.2f} "
              f"до {worst_snr:.2f} — висновок залежить від гіпотези.")

    sgp = res_b[res_b["принцип"] == "MOX"]["RMSE прогнозу, ppm"].iloc[0]
    best_ndir = res_b[res_b["принцип"] == "NDIR"].nsmallest(1, "RMSE прогнозу, ppm")
    print()
    print(f"Б. Заміна SGP30 на {best_ndir['датчик'].iloc[0]} знижує похибку")
    print(f"   прогнозу CO2 з {sgp:.0f} до {best_ndir['RMSE прогнозу, ppm'].iloc[0]:.0f} ppm "
          f"(у {sgp / best_ndir['RMSE прогнозу, ppm'].iloc[0]:.1f} раза).")

    fig_path = cfg.FIGURES / "fig05_co2_sensor_choice.png"
    figure(res_a, res_b, fig_path)
    print()
    print("Таблиці: table03a_co2_sensitivity.csv, table03b_co2_sensor_choice.csv")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
