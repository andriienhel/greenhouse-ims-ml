"""
ДОСЛІД E3. Внесок окремих складових похибки.

Дослід E2 показав, наскільки падає точність прогнозу при переході на бюджетні
датчики. Але він не відповідає на інженерне питання: що саме в цих датчиках
погано і куди вкладати зусилля.

Тут похибка розкладається на складові. Кожна вмикається окремо при вимкнених
решті, і вимірюється, скільки точності втрачається саме через неї:

  noise   випадковий шум            → знімається фільтрацією та усередненням
  bias    систематичне зміщення     → знімається калібруванням за еталоном
  drift   відхід у часі             → потребує періодичної повірки
  quant   квантування               → визначається розрядністю, лише заміна
  lag     інерційність датчика      → лише заміна датчика
  cross   перехресна чутливість     → знімається програмною компенсацією
  faults  відмови (пропуски, залипання) → знімається алгоритмами відновлення
  method  методична похибка         → не знімається нічим, лише заміна датчика

Практичний сенс: складові з верхньої частини списку лікуються програмно й
дешево, з нижньої — лише заміною апаратури. Знаючи, яка з них переважає, можна
обґрунтовано вирішити, що робити з приладом.

Запуск:  .venv/bin/python scripts/exp04_error_components.py
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
# Шість прогонів, а не три. Систематичне зміщення розігрується заново в кожному
# прогоні (у кожного примірника датчика воно своє), тому розкид оцінки його
# внеску великий і трьох прогонів не вистачає, щоб відділити внесок від
# випадкового розкиду.
N_SEEDS = 6

COMPONENTS = {
    "noise": "випадковий шум",
    "bias": "систематичне зміщення",
    "drift": "дрейф у часі",
    "quant": "квантування",
    "lag": "інерційність",
    "cross": "перехресна чутливість",
    "faults": "відмови каналу",
    "method": "методична похибка",
}

# Що можна зробити з кожною складовою, не міняючи датчик
REMEDY = {
    "noise": "фільтрація",
    "bias": "калібрування",
    "drift": "періодична повірка",
    "quant": "заміна датчика",
    "lag": "заміна датчика",
    "cross": "програмна компенсація",
    "faults": "відновлення даних",
    "method": "ЛИШЕ заміна датчика",
}


def run_once(ref: pd.DataFrame, observed: pd.DataFrame, target: str) -> float:
    obs = ref.copy()
    for c in observed.columns:
        obs[c] = observed[c]
    data = make_supervised(obs, target, horizon=HORIZON, df_truth=ref)
    s = chronological_split(data)
    model = make_hist_gb(seed=cfg.SEED)
    model.fit(s["train"]["X"], s["train"]["y_obs"], s["val"]["X"], s["val"]["y_obs"])
    return evaluate(s["test"]["y_true"], model.predict(s["test"]["X"]))["RMSE"]


def mean_rmse(ref: pd.DataFrame, target: str, only: tuple[str, ...]) -> tuple[float, float]:
    vals = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(cfg.SEED + seed)
        obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, only=only)
        vals.append(run_once(ref, obs, target))
    return float(np.mean(vals)), float(np.std(vals))


def postprocess(res: pd.DataFrame) -> pd.DataFrame:
    """Перерахувати частки внеску й оцінити їхню достовірність.

    ЧОМУ ЧАСТКИ РАХУЮТЬСЯ ЗА КВАДРАТАМИ. Середньоквадратичні похибки від різних
    джерел не складаються напряму: за незалежних джерел складаються їхні
    дисперсії. Тому частка внеску рахується як

        (RMSE_складової² − RMSE_ідеального²) / (RMSE_повної² − RMSE_ідеального²)

    Навіть так сума часток не зобов'язана дорівнювати рівно 100 %: джерела
    похибки не повністю незалежні (наприклад, дрейф і зміщення діють на канал
    схоже), а модель прогнозування реагує на них нелінійно. Це треба чесно
    застерегти, а не підганяти числа.

    ДОСТОВІРНІСТЬ. Оцінка порівнюється зі стандартною похибкою середнього,
    тобто з розкидом між прогонами, поділеним на корінь із їх кількості. Саме
    вона характеризує точність оцінки середнього, а не розкид окремих
    реалізацій. Внесок вважається значущим, якщо приріст похибки перевищує дві
    такі похибки.
    """
    out = res.copy()
    for ch in out["канал"].unique():
        m = out["канал"] == ch
        base = float(out.loc[m & (out["код"] == "none"), "RMSE"].iloc[0])
        full = float(out.loc[m & (out["код"] == "all"), "RMSE"].iloc[0])
        denom = full**2 - base**2

        vals = out.loc[m, "RMSE"].astype(float)
        share = (vals**2 - base**2) / denom * 100 if denom > 0 else np.nan
        out.loc[m, "частка внеску,%"] = share.round(1)
        out.loc[m & (out["код"] == "all"), "частка внеску,%"] = 100.0
        out.loc[m & (out["код"] == "none"), "частка внеску,%"] = 0.0

        delta = vals - base
        sd = out.loc[m, "СКВ за прогонами"].astype(float)
        se = sd / np.sqrt(N_SEEDS)          # стандартна похибка середнього
        out.loc[m, "достовірність"] = np.where(
            delta.abs() > 2 * se.clip(lower=1e-9), "значущий", "у межах розкиду"
        )
        out.loc[m & out["код"].isin(["all", "none"]), "достовірність"] = "—"
    return out


def run(ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        print(f"\n  Канал {target}")
        # Опорна точка: усі складові вимкнені — ідеальний вимірювальний канал
        base, _ = mean_rmse(ref, target, only=())
        # Усі складові ввімкнені — повна паспортна похибка
        full, full_sd = mean_rmse(ref, target, only=tuple(COMPONENTS))
        print(f"    ідеальний канал      RMSE={base:8.3f}")
        print(f"    усі складові         RMSE={full:8.3f}")

        for comp, label in COMPONENTS.items():
            val, sd = mean_rmse(ref, target, only=(comp,))
            rows.append({
                "канал": target,
                "од. вим.": cfg.CHANNELS[target]["unit"],
                "складова": label,
                "код": comp,
                "RMSE": round(val, 3),
                "СКВ за прогонами": round(sd, 3),
                "приріст RMSE": round(val - base, 3),
                "чим лікується": REMEDY[comp],
            })
            print(f"    {label:28} RMSE={val:8.3f}  приріст={val - base:+8.3f}")

        rows.append({
            "канал": target, "од. вим.": cfg.CHANNELS[target]["unit"],
            "складова": "УСІ РАЗОМ", "код": "all",
            "RMSE": round(full, 3), "СКВ за прогонами": round(full_sd, 3),
            "приріст RMSE": round(full - base, 3), "чим лікується": "—",
        })
        rows.append({
            "канал": target, "од. вим.": cfg.CHANNELS[target]["unit"],
            "складова": "ідеальний канал", "код": "none",
            "RMSE": round(base, 3), "СКВ за прогонами": 0.0,
            "приріст RMSE": 0.0, "чим лікується": "—",
        })

    return postprocess(pd.DataFrame(rows))


def figure(res: pd.DataFrame, path: Path) -> None:
    # Панелі одна під одною, а не поруч: у документі рисунок вписується в
    # ширину сторінки, і при горизонтальному розташуванні підписи складових
    # стають нечитабельними
    targets = cfg.TARGETS
    fig, axes = plt.subplots(len(targets), 1, figsize=(9.5, 3.4 * len(targets)))

    for ax, target in zip(axes, targets):
        sub = (res[(res["канал"] == target) & (~res["код"].isin(["all", "none"]))]
               .sort_values("частка внеску,%"))
        # Червоним — те, що лікується лише заміною датчика.
        # Блідим і штрихуванням — внески, невідрізнювані від випадкового розкиду
        colors, hatches = [], []
        for _, r in sub.iterrows():
            base_color = "#c62828" if "заміна" in r["чим лікується"] else "#2e7d32"
            significant = r.get("достовірність") == "значущий"
            colors.append(base_color if significant else "#bdbdbd")
            hatches.append("" if significant else "//")

        bars = ax.barh(sub["складова"], sub["частка внеску,%"], color=colors)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        meta = cfg.CHANNELS[target]
        ax.set_title(f"{meta['descr']} ({meta['unit']})", fontsize=10)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel("частка в загальному погіршенні прогнозу, %")

    from matplotlib.patches import Patch

    # Легенду виносимо під рисунки: усередині панелей вона перекриває стовпці
    fig.legend(
        handles=[Patch(color="#2e7d32", label="лікується програмно"),
                 Patch(color="#c62828", label="потрібна заміна датчика"),
                 Patch(facecolor="#bdbdbd", hatch="//",
                       label="внесок у межах розкиду прогонів")],
        fontsize=9, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("E3. Внесок окремих складових похибки в деградацію прогнозу",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E3. Внесок окремих складових похибки")
    print("=" * 78)
    print(f"Горизонт {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв, "
          f"{N_SEEDS} незалежних прогонів на кожну складову")

    res = run(ref)
    res.to_csv(cfg.TABLES / "table05_error_components.csv", index=False)

    print()
    print("=" * 78)
    print("ГОЛОВНЕ ДЖЕРЕЛО ПОХИБКИ ПО КОЖНОМУ КАНАЛУ")
    print("=" * 78)
    for target in cfg.TARGETS:
        sub = res[(res["канал"] == target) & (~res["код"].isin(["all", "none"]))]
        sig = sub[sub["достовірність"] == "значущий"]
        if sig.empty:
            base = float(res[(res["канал"] == target) & (res["код"] == "none")]["RMSE"].iloc[0])
            full = float(res[(res["канал"] == target) & (res["код"] == "all")]["RMSE"].iloc[0])
            print(f"  {target:8} загальна деградація мала ({base:.3f} -> {full:.3f}), "
                  f"внески окремих складових невідрізнювані від розкиду")
            continue
        top = sig.nlargest(1, "частка внеску,%").iloc[0]
        print(f"  {target:8} {top['складова']:28} "
              f"{top['частка внеску,%']:5.1f} %   лікується: {top['чим лікується']}")

    print()
    print("Примітка: частки внеску не зобов'язані сумуватися в 100 %. Похибки")
    print("від різних джерел складаються не лінійно, а самі джерела не")
    print("повністю незалежні. Штрихуванням на рисунку позначені внески, які")
    print("не перевищують розкиду між незалежними прогонами.")

    fig_path = cfg.FIGURES / "fig07_error_components.png"
    figure(res, fig_path)
    print()
    print(f"Таблиця: {(cfg.TABLES / 'table05_error_components.csv').relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
