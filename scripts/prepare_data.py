"""
Підготовка опорного ряду для дослідів.

Бере сирі дані AGC, чистить їх і зберігає готову таблицю, яку далі
використовують усі досліди. Заразом друкує підсумок і будує оглядовий рисунок.

Запуск:  .venv/bin/python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from gims import config as cfg  # noqa: E402
from gims.datasets import (  # noqa: E402
    COMPARTMENTS,
    clean_reference,
    describe,
    load_compartment,
)

COMPARTMENT = "Reference"  # відсік із ручним керуванням досвідченим агрономом


def survey() -> pd.DataFrame:
    """Порівняти всі відсіки за довжиною придатної ділянки."""
    rows = []
    for comp in COMPARTMENTS:
        df = load_compartment(comp)
        c = clean_reference(df)
        rows.append(
            {
                "відсік": comp,
                "усього відліків": len(df),
                "суцільна ділянка": len(c),
                "діб": round(len(c) * cfg.SAMPLE_PERIOD_S / 86400, 1),
                "початок": c.index.min().date(),
                "кінець": c.index.max().date(),
            }
        )
    return pd.DataFrame(rows)


def overview_figure(df: pd.DataFrame, path: Path) -> None:
    """Оглядовий рисунок опорних рядів."""
    chans = [c for c in cfg.CHANNELS if c in df.columns]
    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 2.0 * len(chans)), sharex=True)
    week = df.iloc[: 7 * 24 * 12]  # перший тиждень, щоб динаміка була видна

    for ax, ch in zip(axes, chans):
        meta = cfg.CHANNELS[ch]
        ax.plot(week.index, week[ch], lw=0.8, color="#1f4e79")
        ax.set_ylabel(f"{meta['descr']}, {meta['unit']}", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)

    axes[0].set_title(
        f"Опорні ряди мікроклімату, відсік {df.attrs.get('compartment', '')} "
        f"(перший тиждень, крок {cfg.SAMPLE_PERIOD_S // 60} хв)",
        fontsize=10,
    )
    axes[-1].set_xlabel("час")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("Огляд відсіків теплиці")
    print("=" * 70)
    s = survey()
    print(s.to_string(index=False))
    s.to_csv(cfg.TABLES / "compartments_survey.csv", index=False)

    print()
    print("=" * 70)
    print(f"Готуємо опорний ряд за відсіком: {COMPARTMENT}")
    print("=" * 70)
    raw = load_compartment(COMPARTMENT)
    ref = clean_reference(raw, report=True)

    print()
    print(f"Підсумок: {len(ref)} відліків, "
          f"{len(ref) * cfg.SAMPLE_PERIOD_S / 86400:.1f} діб "
          f"({ref.index.min()} — {ref.index.max()})")
    print()

    tab = describe(ref)
    print(tab.to_string(index=False))
    tab.to_csv(cfg.TABLES / "channels_summary.csv", index=False)

    out = cfg.DATA_PROCESSED / f"reference_{COMPARTMENT}.parquet"
    ref.to_parquet(out)
    print(f"\nОпорний ряд збережено: {out.relative_to(cfg.ROOT)}")

    figpath = cfg.FIGURES / "fig01_reference_series.png"
    overview_figure(ref, figpath)
    print(f"Рисунок збережено:     {figpath.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
