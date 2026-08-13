"""
Аналіз придатності вимірювальних каналів приладу.

Це той скрипт, який можна запустити й показати. Він відповідає на питання:
«наскільки показання наших бюджетних датчиків відповідають тому, що
відбувається в теплиці насправді».

Що робить:
  1. бере опорний (істинний) ряд із набору даних промислової теплиці;
  2. пропускає його через моделі похибок наших датчиків;
  3. рахує метрики придатності кожного каналу;
  4. будує рисунок «істина проти показань» і діаграму придатності.

Запуск:  .venv/bin/python scripts/demo.py
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
from gims.sensors import SENSOR_SPECS, corrupt_frame  # noqa: E402

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"


def load_reference() -> pd.DataFrame:
    if not REF_FILE.exists():
        raise SystemExit(
            "Немає підготовлених даних. Спочатку виконайте:\n"
            "  .venv/bin/python scripts/download_data.py\n"
            "  .venv/bin/python scripts/prepare_data.py"
        )
    return pd.read_parquet(REF_FILE)


def channel_metrics(truth: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Метрики придатності вимірювального каналу.

    Ключова метрика — відношення сигнал/похибка:
        SNR = СКВ(істинного сигналу) / СКВ(похибки)

    Сенс простий. Прогнозувати має сенс зміни величини. Якщо похибка каналу
    більша, ніж сам розмах змін (SNR < 1), то датчик показує переважно власну
    похибку, а не те, що відбувається в теплиці. Такий канал для задачі
    керування непридатний, скільки б даних ми в модель не завантажили.

    Друга метрика — те саме ПІСЛЯ ідеального лінійного калібрування каналу.
    Систематичні складові (зміщення нуля і похибка коефіцієнта перетворення)
    усуваються калібруванням за еталоном, лишається невитравна частина —
    випадковий шум і методична похибка. Для лінійного калібрування СКВ залишку
    дорівнює СКВ(сигналу)·√(1−r²), звідки SNR = 1/√(1−r²).

    Різниця між двома оцінками і є інженерним висновком: чи достатньо
    відкалібрувати датчик, чи його треба міняти.
    """
    rows = []
    for ch in SENSOR_SPECS:
        if ch not in truth.columns or ch not in meas.columns:
            continue
        t = truth[ch].to_numpy(dtype=float)
        m = meas[ch].to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(m)
        err = m[ok] - t[ok]

        sigma_signal = float(np.std(t[ok]))
        sigma_err = float(np.std(err))
        rmse = float(np.sqrt(np.mean(err**2)))
        snr = sigma_signal / sigma_err if sigma_err > 0 else np.inf
        corr = float(np.corrcoef(t[ok], m[ok])[0, 1])
        snr_cal = 1.0 / np.sqrt(1.0 - corr**2) if abs(corr) < 1 else np.inf

        if snr_cal >= 5:
            verdict = "придатний"
        elif snr_cal >= 2:
            verdict = "придатний із застереженнями"
        elif snr_cal >= 1.5:
            verdict = "на межі"
        else:
            verdict = "НЕПРИДАТНИЙ"

        rows.append(
            {
                "канал": ch,
                "датчик": cfg.CHANNELS[ch]["sensor"],
                "од. вим.": cfg.CHANNELS[ch]["unit"],
                "СКВ сигналу": round(sigma_signal, 2),
                "СКВ похибки": round(sigma_err, 2),
                "RMSE": round(rmse, 2),
                "зміщення": round(float(np.mean(err)), 2),
                "сигнал/похибка": round(snr, 2),
                "після калібрування": round(snr_cal, 2),
                "кореляція": round(corr, 3),
                "пропусків,%": round(float((~np.isfinite(m)).mean() * 100), 2),
                "висновок": verdict,
            }
        )
    return pd.DataFrame(rows)


def figure_truth_vs_measured(truth: pd.DataFrame, meas: pd.DataFrame, path: Path) -> None:
    """Накладення: що відбувалося насправді і що показав би наш датчик."""
    chans = [c for c in SENSOR_SPECS if c in truth.columns]
    n_show = 2 * 24 * 12  # дві доби
    t = truth.iloc[:n_show]
    m = meas.iloc[:n_show]

    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 2.1 * len(chans)), sharex=True)
    for ax, ch in zip(axes, chans):
        meta = cfg.CHANNELS[ch]
        ax.plot(t.index, t[ch], lw=1.4, color="#1f4e79", label="істинне значення",
                zorder=2)
        ax.plot(m.index, m[ch], lw=0.9, color="#c00000", alpha=0.85,
                label="показання нашого датчика", zorder=3)
        ax.set_ylabel(f"{meta['descr']}, {meta['unit']}", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[0].set_title(
        "Опорне значення і модельне показання вимірювального каналу (дві доби)",
        fontsize=10,
    )
    axes[-1].set_xlabel("час")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_snr(metrics: pd.DataFrame, path: Path) -> None:
    """Діаграма придатності каналів за відношенням сигнал/похибка."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    raw = metrics["сигнал/похибка"].to_numpy(dtype=float)
    cal = metrics["після калібрування"].to_numpy(dtype=float)
    labels = [f"{cfg.CHANNELS[r['канал']]['descr']}\n{r['датчик'].split('(')[0].strip()}"
              for _, r in metrics.iterrows()]
    x = np.arange(len(labels))
    w = 0.38

    ax.bar(x - w / 2, raw, w, color="#8fa8bf", label="як є")
    ax.bar(x + w / 2, cal, w, color="#1f4e79", label="після калібрування")

    ax.axhline(1, color="#c62828", ls="--", lw=1.2)
    ax.text(len(labels) - 0.45, 1.05, "поріг непридатності", fontsize=8, color="#c62828")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("відношення сигнал / похибка")
    ax.set_title(
        "Придатність вимірювальних каналів приладу\n"
        "(розрив між стовпцями — та частина похибки, що знімається калібруванням)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(fontsize=8)
    for xi, v in zip(x - w / 2, raw):
        ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, cal):
        ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ref = load_reference()
    rng = np.random.default_rng(cfg.SEED)
    meas = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, scale=1.0)

    print("=" * 78)
    print("ПРИДАТНІСТЬ ВИМІРЮВАЛЬНИХ КАНАЛІВ ПРИЛАДУ")
    print("=" * 78)
    print(f"Опорні дані: відсік {ref.attrs.get('compartment', 'Reference')}, "
          f"{len(ref)} відліків, {len(ref) * cfg.SAMPLE_PERIOD_S / 86400:.1f} діб")
    print("Рівень похибки: паспортний (scale = 1.0)")
    print()

    metrics = channel_metrics(ref, meas)
    print(metrics.to_string(index=False))
    metrics.to_csv(cfg.TABLES / "table01_channel_suitability.csv", index=False)

    print()
    print("-" * 78)
    print("Як читати таблицю:")
    print("  «сигнал/похибка»     — у скільки разів розмах вимірюваної величини")
    print("                         більший за похибку каналу. Менше 1 означає,")
    print("                         що датчик показує більше власної похибки,")
    print("                         ніж реальної динаміки теплиці.")
    print("  «після калібрування» — те саме, але якщо прибрати систематичну")
    print("                         частину похибки звірянням з еталоном.")
    print()
    print("  Великий розрив між стовпцями = датчик достатньо відкалібрувати.")
    print("  Розриву немає і значення низьке = калібрування не врятує, потрібен")
    print("  інший датчик.")
    print("-" * 78)

    f2 = cfg.FIGURES / "fig02_truth_vs_measured.png"
    f3 = cfg.FIGURES / "fig03_channel_snr.png"
    figure_truth_vs_measured(ref, meas, f2)
    figure_snr(metrics, f3)

    print()
    print("Збережено:")
    print(f"  {(cfg.TABLES / 'table01_channel_suitability.csv').relative_to(cfg.ROOT)}")
    print(f"  {f2.relative_to(cfg.ROOT)}")
    print(f"  {f3.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
