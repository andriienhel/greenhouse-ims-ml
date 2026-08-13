"""
ДОСЛІД E1. Базова точність прогнозування на опорних даних.

Завдання етапу — встановити точку відліку. Перш ніж вивчати, скільки точності
з'їдають похибки вимірювальних каналів, треба знати, яка точність узагалі
досяжна на ідеальних даних і яка модель її забезпечує.

Перевірювані питання:
  1. Наскільки моделі перевершують наївний прогноз «величина не зміниться»?
     На горизонті 15 хвилин інерція теплиці велика, і це неочевидно.
  2. Яка модель краща і чи варте ускладнення?
  3. Як деградує точність зі зростанням горизонту прогнозу?

Запуск:  .venv/bin/python scripts/exp01_baseline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from gims import config as cfg  # noqa: E402
from gims.features import chronological_split, make_supervised  # noqa: E402
from gims.metrics import diebold_mariano, evaluate  # noqa: E402
from gims.models import default_models  # noqa: E402

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"


def run(ref: pd.DataFrame, with_gru: bool = True) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        for h, h_min in zip(cfg.HORIZONS, cfg.HORIZONS_MIN):
            data = make_supervised(ref, target, horizon=h)
            split = chronological_split(data)
            tr, va, te = split["train"], split["val"], split["test"]

            preds: dict[str, "pd.Series"] = {}
            for model in default_models(target, seed=cfg.SEED, with_gru=with_gru):
                t0 = time.time()
                model.fit(tr["X"], tr["y_obs"], va["X"], va["y_obs"])
                p = model.predict(te["X"])
                preds[model.name] = p

                m = evaluate(te["y_true"], p, te["y_naive"])
                rows.append(
                    {
                        "канал": target,
                        "горизонт,хв": h_min,
                        "модель": model.name,
                        "RMSE": round(m["RMSE"], 4),
                        "MAE": round(m["MAE"], 4),
                        "R2": round(m["R2"], 4),
                        "виграш над наївним,%": round(m["виграш над наївним,%"], 1),
                        "час навчання,с": round(time.time() - t0, 1),
                    }
                )
                print(f"  {target:8} h={h_min:3} хв  {model.name:20} "
                      f"RMSE={m['RMSE']:8.4f}  "
                      f"виграш={m['виграш над наївним,%']:6.1f}%")

            # Перевіряємо, чи значуще найкраща модель перевершує наївний
            # прогноз. Без цієї перевірки твердження «модель краща» лишається
            # бездоказовим.
            best = min(
                (r for r in rows if r["канал"] == target and r["горизонт,хв"] == h_min
                 and r["модель"] != "Наївний"),
                key=lambda r: r["RMSE"],
            )
            dm, p_val = diebold_mariano(
                te["y_true"], preds[best["модель"]], preds["Наївний"], horizon=h
            )
            best["DM-статистика"] = round(dm, 2)
            best["p-значення"] = f"{p_val:.2e}"

    return pd.DataFrame(rows)


def figure(res: pd.DataFrame, path: Path) -> None:
    """Виграш моделей над наївним прогнозом по каналах і горизонтах."""
    targets = cfg.TARGETS
    models = [m for m in res["модель"].unique() if m != "Наївний"]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 3.9),
                             sharey=True)

    for ax, target in zip(axes, targets):
        sub = res[res["канал"] == target]
        for model in models:
            s = sub[sub["модель"] == model].sort_values("горизонт,хв")
            ax.plot(s["горизонт,хв"], s["виграш над наївним,%"],
                    marker="o", lw=1.6, ms=5, label=model)
        ax.axhline(0, color="#c62828", ls="--", lw=1)
        meta = cfg.CHANNELS[target]
        ax.set_title(f"{meta['descr']} ({meta['unit']})", fontsize=10)
        ax.set_xlabel("горизонт прогнозу, хв")
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xticks(cfg.HORIZONS_MIN)

    axes[0].set_ylabel("виграш над наївним прогнозом, %")
    axes[-1].legend(fontsize=8)
    fig.suptitle("E1. Точність прогнозування на опорних даних", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("Навчання моделей на опорних (неспотворених) даних...")
    res = run(ref)

    out_csv = cfg.TABLES / "table02_baseline_accuracy.csv"
    res.to_csv(out_csv, index=False)

    print()
    print("=" * 78)
    print("E1. НАЙКРАЩА МОДЕЛЬ ПО КОЖНОМУ КАНАЛУ Й ГОРИЗОНТУ")
    print("=" * 78)
    best = (res[res["модель"] != "Наївний"]
            .sort_values("RMSE")
            .groupby(["канал", "горизонт,хв"], as_index=False)
            .first()
            .sort_values(["канал", "горизонт,хв"]))
    print(best[["канал", "горизонт,хв", "модель", "RMSE",
                "виграш над наївним,%", "DM-статистика",
                "p-значення"]].to_string(index=False))

    fig_path = cfg.FIGURES / "fig04_baseline_accuracy.png"
    figure(res, fig_path)

    print()
    print(f"Таблиця: {out_csv.relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
