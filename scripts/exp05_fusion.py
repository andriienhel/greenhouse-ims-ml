"""
ДОСЛІД E4. Сенсорна фузія: чи окупається кожен вимірювальний канал.

Питання просте й практичне: чи потрібні в приладі всі п'ять датчиків, чи
частина з них не дає нічого понад те, що вже відомо з решти.

ЧАСТИНА А. Нарощування набору ознак.
Модель навчається послідовно на дедалі ширшому наборі даних: спочатку лише за
історією самої прогнозованої величини, потім додаються решта каналів приладу,
зовнішня погода і стани виконавчих механізмів.

ЧАСТИНА Б. Виключення каналів по одному.
З повного набору почергово прибирається один канал приладу. Приріст похибки
показує, скільки саме цей канал вартий. Це чесніша оцінка, ніж нарощування:
вона враховує, що канали дублюють один одного.

Канали приладу беруться спотвореними (як у реальному приладі), а погода й
виконавчі механізми — точними, бо їх міряє не наш прилад, а система керування
теплицею.

Запуск:  .venv/bin/python scripts/exp05_fusion.py
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
COLORS = {"T_air": "#1f4e79", "RH_air": "#2e7d32", "CO2_air": "#c62828"}

DEVICE = list(cfg.CHANNELS)                       # те, що міряє наш прилад
WEATHER = ["T_out", "RH_out", "I_glob", "wind", "rain"]
ACTUATORS = ["vent_lee", "vent_wind", "pipe_low", "pipe_grow",
             "lamps", "scr_energy", "scr_black", "co2_dosing"]

SETS_ORDER = ["лише свій канал", "усі датчики приладу",
              "+ зовнішня погода", "+ виконавчі механізми"]


def fit_eval(ref: pd.DataFrame, observed: pd.DataFrame, target: str,
             feature_cols: list[str]) -> float:
    obs = ref.copy()
    for c in observed.columns:
        obs[c] = observed[c]
    cols = [c for c in feature_cols if c in obs.columns]
    data = make_supervised(obs, target, horizon=HORIZON,
                           feature_cols=cols, df_truth=ref)
    s = chronological_split(data)
    model = make_hist_gb(seed=cfg.SEED)
    model.fit(s["train"]["X"], s["train"]["y_obs"], s["val"]["X"], s["val"]["y_obs"])
    return evaluate(s["test"]["y_true"], model.predict(s["test"]["X"]))["RMSE"]


def mean_rmse(ref: pd.DataFrame, target: str,
              feature_cols: list[str]) -> tuple[float, float]:
    vals = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(cfg.SEED + seed)
        obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, scale=1.0)
        vals.append(fit_eval(ref, obs, target, feature_cols))
    return float(np.mean(vals)), float(np.std(vals))


def part_a(ref: pd.DataFrame) -> pd.DataFrame:
    print("\nЧАСТИНА А. Нарощування набору ознак")
    print("-" * 78)
    rows = []
    for target in cfg.TARGETS:
        sets = {
            "лише свій канал": [target],
            "усі датчики приладу": DEVICE,
            "+ зовнішня погода": DEVICE + WEATHER,
            "+ виконавчі механізми": DEVICE + WEATHER + ACTUATORS,
        }
        base = None
        for label, cols in sets.items():
            val, sd = mean_rmse(ref, target, cols)
            if base is None:
                base = val
            rows.append({
                "канал": target,
                "од. вим.": cfg.CHANNELS[target]["unit"],
                "набір ознак": label,
                "кількість каналів": len([c for c in cols if c in ref.columns]),
                "RMSE": round(val, 3),
                "СКВ за прогонами": round(sd, 3),
                "покращення до «лише свій канал»,%": round((1 - val / base) * 100, 1),
            })
            print(f"  {target:8} {label:24} RMSE={val:8.3f}  "
                  f"покращення={(1 - val / base) * 100:6.1f}%")
    return pd.DataFrame(rows)


def part_b(ref: pd.DataFrame) -> pd.DataFrame:
    print("\nЧАСТИНА Б. Виключення каналів приладу по одному")
    print("-" * 78)
    full_cols = DEVICE + WEATHER + ACTUATORS
    rows = []
    for target in cfg.TARGETS:
        full, full_sd = mean_rmse(ref, target, full_cols)
        print(f"  {target:8} повний набір       RMSE={full:8.3f}")
        for drop in DEVICE:
            cols = [c for c in full_cols if c != drop]
            val, sd = mean_rmse(ref, target, cols)
            loss = (val / full - 1) * 100
            rows.append({
                "канал": target,
                "виключений датчик": drop,
                "RMSE без нього": round(val, 3),
                "RMSE повного набору": round(full, 3),
                "СКВ за прогонами": round(sd, 3),
                "втрата точності,%": round(loss, 1),
                "значущий": "так" if abs(val - full) > 2 * sd / np.sqrt(N_SEEDS) else "ні",
            })
            print(f"           без {drop:8} RMSE={val:8.3f}  втрата={loss:+6.1f}%")
    return pd.DataFrame(rows)


def figure(res_a: pd.DataFrame, res_b: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Ліва панель: що дає розширення набору даних
    x = np.arange(len(SETS_ORDER))
    for target in cfg.TARGETS:
        s = res_a[res_a["канал"] == target].set_index("набір ознак").loc[SETS_ORDER]
        ax1.plot(x, s["покращення до «лише свій канал»,%"], marker="o", lw=1.8,
                 color=COLORS[target], label=cfg.CHANNELS[target]["descr"])
    ax1.axhline(0, color="#999", ls="--", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["лише свій\nканал", "усі датчики\nприладу",
                         "+ зовнішня\nпогода", "+ виконавчі\nмеханізми"], fontsize=8)
    ax1.set_ylabel("покращення точності прогнозу, %")
    ax1.set_title("А. Що дає розширення набору даних", fontsize=10)
    ax1.grid(alpha=0.3, lw=0.5)
    ax1.legend(fontsize=8)

    # Права панель: ціна виключення кожного каналу
    xs = np.arange(len(DEVICE))
    w = 0.26
    for i, target in enumerate(cfg.TARGETS):
        s = res_b[res_b["канал"] == target].set_index("виключений датчик").loc[DEVICE]
        ax2.bar(xs + (i - 1) * w, s["втрата точності,%"], w,
                color=COLORS[target], label=cfg.CHANNELS[target]["descr"])
    ax2.axhline(0, color="#333", lw=1)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([cfg.CHANNELS[c]["descr"].replace(" ", "\n") for c in DEVICE],
                        fontsize=8)
    ax2.set_ylabel("зростання похибки без цього каналу, %")
    ax2.set_title("Б. Скільки вартий кожен датчик\n"
                  "(вищий стовпець — датчик потрібніший)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, lw=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("E4. Цінність вимірювальних каналів (горизонт 30 хв)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E4. Сенсорна фузія: чи окупається кожен канал")
    print("=" * 78)
    print(f"Горизонт {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв, {N_SEEDS} прогони, "
          f"канали приладу спотворені паспортною похибкою")

    res_a = part_a(ref)
    res_b = part_b(ref)
    res_a.to_csv(cfg.TABLES / "table06a_fusion_growth.csv", index=False)
    res_b.to_csv(cfg.TABLES / "table06b_fusion_dropout.csv", index=False)

    print()
    print("=" * 78)
    print("ВИСНОВКИ")
    print("=" * 78)
    for target in cfg.TARGETS:
        s = res_b[res_b["канал"] == target].sort_values("втрата точності,%",
                                                        ascending=False)
        top = s.iloc[0]
        useless = s[s["втрата точності,%"] <= 0.5]["виключений датчик"].tolist()
        print(f"  {target:8} найважливіший: {top['виключений датчик']:8} "
              f"(+{top['втрата точності,%']:.1f} % похибки при виключенні)")
        if useless:
            print(f"           практично не потрібні: {', '.join(useless)}")

    fig_path = cfg.FIGURES / "fig08_fusion.png"
    figure(res_a, res_b, fig_path)
    print()
    print("Таблиці: table06a_fusion_growth.csv, table06b_fusion_dropout.csv")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
