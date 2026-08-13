"""
ДОСЛІД E2. Деградація точності прогнозу від рівня похибки каналів.

Центральний дослід роботи. Він відповідає на головне питання: скільки точності
прогнозування втрачає система через те, що вимірювання виконуються бюджетними
датчиками, а не повіреною апаратурою.

Похибка всіх каналів масштабується спільним коефіцієнтом:
  scale = 0   ідеальні вимірювання (верхня межа досяжного)
  scale = 1   паспортна похибка обраних датчиків
  scale = 3   помітно деградовані канали
  scale = 5   канали в передаварійному стані

Модель навчається на спотворених даних (як реальний прилад), а точність
оцінюється за істинними значеннями (як потрібно системі керування теплицею).

ПРО ВИБІР МОДЕЛІ. Тут використовується градієнтний бустинг, а не переможець
досліду E1 — випадковий ліс. Причина практична: розгортка потребує кількох
десятків навчань, а бустинг за майже тієї самої точності навчається приблизно
вдесятеро швидше. Усередині досліду модель одна й та сама, тому порівняння
рівнів похибки між собою коректне.

Запуск:  .venv/bin/python scripts/exp03_noise_level.py
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

HORIZON = 6      # 30 хвилин
N_SEEDS = 3
COLORS = {"T_air": "#1f4e79", "RH_air": "#2e7d32", "CO2_air": "#c62828"}


def run_once(ref: pd.DataFrame, observed: pd.DataFrame, target: str) -> dict:
    """Навчити модель на показаннях датчиків і оцінити прогноз за істиною."""
    obs = ref.copy()
    for c in observed.columns:
        obs[c] = observed[c]

    data = make_supervised(obs, target, horizon=HORIZON, df_truth=ref)
    s = chronological_split(data)

    model = make_hist_gb(seed=cfg.SEED)
    model.fit(s["train"]["X"], s["train"]["y_obs"], s["val"]["X"], s["val"]["y_obs"])
    pred = model.predict(s["test"]["X"])
    return evaluate(s["test"]["y_true"], pred, s["test"]["y_naive"])


def run(ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        for scale in cfg.NOISE_SCALES:
            rmses, skills = [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(cfg.SEED + seed)
                obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, scale=scale)
                m = run_once(ref, obs, target)
                rmses.append(m["RMSE"])
                skills.append(m["виграш над наївним,%"])

            rows.append({
                "канал": target,
                "од. вим.": cfg.CHANNELS[target]["unit"],
                "рівень похибки": scale,
                "RMSE": round(float(np.mean(rmses)), 3),
                "СКВ за прогонами": round(float(np.std(rmses)), 3),
                "виграш над наївним,%": round(float(np.mean(skills)), 1),
            })
            print(f"  {target:8} scale={scale:4.1f}  "
                  f"RMSE={np.mean(rmses):8.3f} ± {np.std(rmses):.3f}  "
                  f"виграш={np.mean(skills):6.1f}%")

    res = pd.DataFrame(rows)

    # Відносна деградація: у скільки разів зросла похибка прогнозу порівняно з
    # ідеальними вимірюваннями. Потрібна, щоб канали з різними одиницями
    # вимірювання можна було показати на одному рисунку.
    base = res[res["рівень похибки"] == 0].set_index("канал")["RMSE"]
    res["зростання RMSE, разів"] = res.apply(
        lambda r: round(r["RMSE"] / base[r["канал"]], 2), axis=1
    )
    return res


def figure(res: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for target in cfg.TARGETS:
        s = res[res["канал"] == target].sort_values("рівень похибки")
        label = cfg.CHANNELS[target]["descr"]
        ax1.plot(s["рівень похибки"], s["зростання RMSE, разів"],
                 marker="o", lw=1.8, color=COLORS[target], label=label)
        ax2.plot(s["рівень похибки"], s["виграш над наївним,%"],
                 marker="o", lw=1.8, color=COLORS[target], label=label)

    ax1.axvline(1.0, color="#666", ls=":", lw=1.2)
    ax1.text(1.06, ax1.get_ylim()[1] * 0.95, "паспортна\nпохибка",
             fontsize=8, color="#666", va="top")
    ax1.axhline(1, color="#999", ls="--", lw=1)
    ax1.set_xlabel("масштаб похибки каналів")
    ax1.set_ylabel("зростання RMSE прогнозу, разів")
    ax1.set_title("А. У скільки разів падає точність\n"
                  "(відносно ідеальних вимірювань)", fontsize=10)
    ax1.grid(alpha=0.3, lw=0.5)
    ax1.legend(fontsize=8)

    ax2.axvline(1.0, color="#666", ls=":", lw=1.2)
    ax2.axhline(0, color="#c62828", ls="--", lw=1)
    ax2.set_xlabel("масштаб похибки каналів")
    ax2.set_ylabel("виграш над наївним прогнозом, %")
    ax2.set_title("Б. Чи лишається сенс у моделі\n"
                  "(наскільки вона краща за «величина не зміниться»)", fontsize=10)
    ax2.grid(alpha=0.3, lw=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("E2. Вплив рівня похибки вимірювальних каналів "
                 "на точність прогнозу (горизонт 30 хв)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E2. Деградація точності від рівня похибки каналів")
    print("=" * 78)
    print(f"Горизонт {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв, "
          f"{N_SEEDS} незалежні прогони на кожен рівень")
    print()

    res = run(ref)
    res.to_csv(cfg.TABLES / "table04_noise_level.csv", index=False)

    print()
    print("=" * 78)
    print("ПІДСУМОК: зростання похибки прогнозу відносно ідеальних вимірювань")
    print("=" * 78)
    pivot = res.pivot(index="рівень похибки", columns="канал",
                      values="зростання RMSE, разів")
    print(pivot.to_string())

    print()
    print("ВИСНОВКИ")
    print("-" * 78)
    for target in cfg.TARGETS:
        s = res[res["канал"] == target]
        at1 = float(s[s["рівень похибки"] == 1.0]["зростання RMSE, разів"].iloc[0])
        skill1 = float(s[s["рівень похибки"] == 1.0]["виграш над наївним,%"].iloc[0])
        print(f"  {target:8} за паспортної похибки датчиків похибка прогнозу "
              f"вища у {at1:.2f} раза,")
        print(f"           виграш над наївним прогнозом {skill1:.1f} %")

    fig_path = cfg.FIGURES / "fig06_noise_level.png"
    figure(res, fig_path)
    print()
    print(f"Таблиця: {(cfg.TABLES / 'table04_noise_level.csv').relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
