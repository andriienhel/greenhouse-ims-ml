"""
ДОСЛІД E7. Реалізовність моделей прогнозування на мікроконтролері ESP32.

Питання: чи можна виконувати прогнозування прямо в приладі, а не надсилати дані
на сервер. Обчислення на місці вимірювання знімають залежність від зв'язку,
прибирають затримку каналу і дозволяють приладу зберігати працездатність при
відмові мережі — усе це суттєво для системи керування теплицею.

ЩО ТУТ ВИМІРЮЄТЬСЯ І ЩО ОЦІНЮЄТЬСЯ — РІЗНІ РЕЧІ.

Вимірюється за фактично навченими моделями (не за літературними даними):
  - кількість параметрів і вузлів дерев;
  - обсяг пам'яті під модель при зберіганні у float32 та в int8;
  - кількість елементарних операцій на один прогноз;
  - час обчислення на настільному процесорі.

Оцінюється розрахунком, із явно вказаними припущеннями:
  - час обчислення на ESP32 за тактової частоти 240 МГц.

НАТУРНИХ ВИМІРЮВАНЬ НА ПЛАТІ ТУТ НЕМАЄ. Оцінка часу отримана діленням кількості
операцій на продуктивність процесора і слугує для відбраковування завідомо
непридатних варіантів, а не для точного передбачення. Розбіжність із дійсністю
у два-три рази цілком звичайна: вона залежить від компілятора, влучань у кеш і
розташування моделі у флеш-пам'яті або в ОЗП. Перевірка на зібраному приладі —
окремий етап роботи.

Запуск:  .venv/bin/python scripts/exp08_esp32_footprint.py
"""
from __future__ import annotations

import sys
import time
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
from gims.models import (  # noqa: E402
    GRUModel,
    Persistence,
    make_hist_gb,
    make_random_forest,
    make_ridge,
)

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"

TARGET = "T_air"
HORIZON = 6

# --------------------------------------------------------------------------
# Характеристики мікроконтролера і прийняті припущення.
# Усі вони зібрані тут, щоб читач міг перерахувати висновки під свої умови, а
# не розшукувати числа по тексту.
# --------------------------------------------------------------------------
CPU_MHZ = 240                  # ESP32, робоча тактова частота

# ОЗП: в ESP32 усього 520 КБ статичної пам'яті, але значну частину займають стек
# бездротового з'єднання, буфери і сама програма. Реально прикладній задачі
# лишається близько 160 КБ при працюючому Wi-Fi.
SRAM_TOTAL_KB = 520
SRAM_AVAILABLE_KB = 160

# Флеш-пам'ять: типовий модуль несе 4 МБ, з яких під розділ застосунку зазвичай
# відводиться близько 1.5 МБ. Модель можна зберігати у флеш-пам'яті й читати
# звідти, але це помітно повільніше, ніж із ОЗП.
FLASH_APP_KB = 1536

# Вартість операцій у тактах. Значення навмисно консервативні.
# Множення з накопиченням над числами одинарної точності виконується
# співпроцесором ESP32 за один такт, однак з урахуванням завантаження операндів
# із пам'яті й накладних витрат циклу приймаємо три такти.
CYCLES_PER_MAC = 3
# Прохід вузлом дерева: читання ознаки, порівняння, перехід.
CYCLES_PER_NODE = 5

# Розмір вузла дерева при зберіганні у вигляді таблиці:
#   номер ознаки     uint16  2 байти
#   поріг            float32 4 байти
#   лівий нащадок    uint16  2 байти
#   правий нащадок   uint16  2 байти
#   значення в листі float32 4 байти
# разом 14 байтів, з вирівнюванням до 16.
BYTES_PER_TREE_NODE = 16

# Період опитування датчиків задає гранично допустимий час обчислення.
# Додатково розглядаємо режим швидкого контуру керування з кроком 1 с.
BUDGET_SLOW_S = cfg.SAMPLE_PERIOD_S
BUDGET_FAST_S = 1.0

UA_MODEL = {"Ridge": "Ridge", "ВипадковийЛіс": "випадковий ліс",
            "ГрадієнтнийБустинг": "градієнтний бустинг", "GRU": "GRU",
            "Наївний прогноз": "наївний прогноз"}


def measure_ridge(model, n_features: int) -> dict:
    est = model.estimator
    n_params = int(est.coef_.size) + 1
    return {
        "параметрів/вузлів": n_params,
        "пам'ять float32, КБ": n_params * 4 / 1024,
        "пам'ять int8, КБ": n_params / 1024,
        "операцій на прогноз": n_features,
        "тактів на прогноз": n_features * CYCLES_PER_MAC,
    }


def measure_forest(model, n_features: int) -> dict:
    est = model.estimator
    nodes = int(sum(t.tree_.node_count for t in est.estimators_))
    depths = [int(t.tree_.max_depth) for t in est.estimators_]
    # На одному прогнозі обходиться по одному шляху в кожному дереві, довжина
    # шляху не перевищує глибини дерева
    ops = int(sum(depths))
    mem = nodes * BYTES_PER_TREE_NODE / 1024
    return {
        "параметрів/вузлів": nodes,
        "пам'ять float32, КБ": mem,
        # Деревам квантування дає мало: вузол зберігає номер ознаки і посилання
        # на нащадків, а не лише число. Приймаємо скорочення вдвічі.
        "пам'ять int8, КБ": mem / 2,
        "операцій на прогноз": ops,
        "тактів на прогноз": ops * CYCLES_PER_NODE,
    }


def measure_hist_gb(model, n_features: int) -> dict:
    est = model.estimator
    nodes, ops = 0, 0
    for stage in est._predictors:
        for pred in stage:
            nodes += int(pred.nodes.shape[0])
            # Глибина дерева обмежена налаштуванням; беремо її як довжину шляху
            ops += int(est.max_depth or 6)
    mem = nodes * BYTES_PER_TREE_NODE / 1024
    return {
        "параметрів/вузлів": nodes,
        "пам'ять float32, КБ": mem,
        "пам'ять int8, КБ": mem / 2,
        "операцій на прогноз": ops,
        "тактів на прогноз": ops * CYCLES_PER_NODE,
    }


def measure_gru(model: GRUModel, n_features: int) -> dict:
    n_params = int(sum(p.numel() for p in model._net.parameters()))
    hidden = model.hidden
    seq = model.seq_len
    # На кожному кроці послідовності три вентилі, кожен множить вхід на матрицю
    # вхідних ваг і попередній стан на матрицю станів
    per_step = 3 * (n_features * hidden + hidden * hidden)
    head = hidden * (hidden // 2) + (hidden // 2)
    ops = seq * per_step + head
    return {
        "параметрів/вузлів": n_params,
        "пам'ять float32, КБ": n_params * 4 / 1024,
        "пам'ять int8, КБ": n_params / 1024,
        "операцій на прогноз": ops,
        "тактів на прогноз": ops * CYCLES_PER_MAC,
    }


def figure(res: pd.DataFrame, path: Path) -> None:
    sub = res[res["модель"] != "Наївний прогноз"].copy()
    sub["ua"] = sub["модель"].map(lambda m: UA_MODEL.get(m, m))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Ліва панель: компроміс між точністю і пам'яттю
    colors = ["#2e7d32" if m <= SRAM_AVAILABLE_KB else "#c62828"
              for m in sub["пам'ять int8, КБ"]]
    ax1.scatter(sub["пам'ять int8, КБ"], sub["RMSE"], s=110, c=colors, zorder=3)
    for _, r in sub.iterrows():
        ax1.annotate(r["ua"], (r["пам'ять int8, КБ"], r["RMSE"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=8.5)
    ax1.axvline(SRAM_AVAILABLE_KB, color="#c62828", ls="--", lw=1.3)
    ax1.text(SRAM_AVAILABLE_KB * 1.08, ax1.get_ylim()[1],
             f"межа ОЗП\n{SRAM_AVAILABLE_KB} КБ", fontsize=8,
             color="#c62828", va="top")
    ax1.set_xscale("log")
    ax1.set_xlabel("пам'ять під модель при зберіганні в int8, КБ")
    ax1.set_ylabel(f"RMSE прогнозу, {cfg.CHANNELS[TARGET]['unit']}")
    ax1.set_title("А. Точність проти обсягу пам'яті", fontsize=10)
    ax1.grid(alpha=0.3, lw=0.5)

    # Права панель: пам'ять порівняно з ресурсами мікроконтролера
    x = np.arange(len(sub))
    ax2.bar(x - 0.2, sub["пам'ять float32, КБ"], 0.4, color="#8fa8bf",
            label="float32")
    ax2.bar(x + 0.2, sub["пам'ять int8, КБ"], 0.4, color="#1f4e79", label="int8")
    ax2.axhline(SRAM_AVAILABLE_KB, color="#c62828", ls="--", lw=1.3,
                label=f"доступно ОЗП ({SRAM_AVAILABLE_KB} КБ)")
    ax2.axhline(FLASH_APP_KB, color="#2e7d32", ls="--", lw=1.3,
                label=f"розділ флеш-пам'яті ({FLASH_APP_KB} КБ)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["ua"], fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("обсяг пам'яті під модель, КБ")
    ax2.set_title("Б. Вимоги до пам'яті та ресурси ESP32", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, lw=0.5)
    ax2.legend(fontsize=7.5)

    fig.suptitle("E7. Реалізовність моделей прогнозування на ESP32", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E7. Реалізовність моделей на мікроконтролері ESP32")
    print("=" * 78)
    print(f"Канал {TARGET}, горизонт {HORIZON * cfg.SAMPLE_PERIOD_S // 60} хв")
    print(f"Припущення: {CPU_MHZ} МГц, {CYCLES_PER_MAC} такти на множення з "
          f"накопиченням, {CYCLES_PER_NODE} тактів на вузол дерева")
    print()

    data = make_supervised(ref, TARGET, horizon=HORIZON)
    s = chronological_split(data)
    tr, va, te = s["train"], s["val"], s["test"]
    n_features = tr["X"].shape[1]

    builders = [
        ("Ridge", make_ridge(seed=cfg.SEED), measure_ridge),
        ("ВипадковийЛіс", make_random_forest(seed=cfg.SEED), measure_forest),
        ("ГрадієнтнийБустинг", make_hist_gb(seed=cfg.SEED), measure_hist_gb),
        ("GRU", GRUModel(seed=cfg.SEED), measure_gru),
    ]

    # Наївний прогноз як нижня межа витрат: він не потребує ні пам'яті, ні
    # обчислень, але й точність дає відповідну
    naive = Persistence(TARGET).fit(tr["X"], tr["y_obs"])
    rmse_naive = evaluate(te["y_true"], naive.predict(te["X"]))["RMSE"]

    rows = [{
        "модель": "Наївний прогноз",
        "RMSE": round(rmse_naive, 4),
        "параметрів/вузлів": 1,
        "пам'ять float32, КБ": 0.004,
        "пам'ять int8, КБ": 0.001,
        "операцій на прогноз": 1,
        "час на ESP32, мс": 0.0,
        "час на ПК, мс": 0.0,
    }]

    for name, model, measure in builders:
        print(f"  навчаю {name}...", flush=True)
        model.fit(tr["X"], tr["y_obs"], va["X"], va["y_obs"])

        pred = model.predict(te["X"])
        rmse = evaluate(te["y_true"], pred)["RMSE"]

        # Реальний замір часу на настільному процесорі: один прогноз
        x1 = te["X"].iloc[:1]
        t0 = time.perf_counter()
        for _ in range(50):
            model.predict(x1)
        host_ms = (time.perf_counter() - t0) / 50 * 1000

        m = measure(model, n_features)
        esp_ms = m["тактів на прогноз"] / (CPU_MHZ * 1e6) * 1000

        rows.append({
            "модель": name,
            "RMSE": round(rmse, 4),
            "параметрів/вузлів": m["параметрів/вузлів"],
            "пам'ять float32, КБ": round(m["пам'ять float32, КБ"], 1),
            "пам'ять int8, КБ": round(m["пам'ять int8, КБ"], 1),
            "операцій на прогноз": m["операцій на прогноз"],
            "час на ESP32, мс": round(esp_ms, 2),
            "час на ПК, мс": round(host_ms, 3),
        })

    res = pd.DataFrame(rows)

    # Чи вміщується модель і чи вкладається в період опитування
    res["в ОЗП (int8)"] = np.where(res["пам'ять int8, КБ"] <= SRAM_AVAILABLE_KB,
                                   "так", "ні")
    res["у флеш-пам'ять"] = np.where(res["пам'ять float32, КБ"] <= FLASH_APP_KB,
                                     "так", "ні")
    res["встигає за 5 хв"] = np.where(
        res["час на ESP32, мс"] / 1000 <= BUDGET_SLOW_S, "так", "ні")
    res["встигає за 1 с"] = np.where(
        res["час на ESP32, мс"] / 1000 <= BUDGET_FAST_S, "так", "ні")

    res.to_csv(cfg.TABLES / "table09_esp32_footprint.csv", index=False)

    print()
    print(res[["модель", "RMSE", "параметрів/вузлів", "пам'ять float32, КБ",
               "пам'ять int8, КБ", "час на ESP32, мс"]].to_string(index=False))
    print()
    print(res[["модель", "в ОЗП (int8)", "у флеш-пам'ять",
               "встигає за 5 хв", "встигає за 1 с"]].to_string(index=False))

    print()
    print("=" * 78)
    print("ВИСНОВКИ")
    print("=" * 78)
    col_int8 = "пам'ять int8, КБ"
    col_f32 = "пам'ять float32, КБ"

    fits = res[(res["в ОЗП (int8)"] == "так") & (res["модель"] != "Наївний прогноз")]
    best = None
    if not fits.empty:
        best = fits.loc[fits["RMSE"].idxmin()]
        print(f"  Найкраща з тих, що вміщуються в ОЗП: {best['модель']}")
        print(f"    RMSE {best['RMSE']}, пам'ять {best[col_int8]} КБ (int8), "
              f"час {best['час на ESP32, мс']} мс")
    overall = res[res["модель"] != "Наївний прогноз"]
    champion = overall.loc[overall["RMSE"].idxmin()]
    if best is None or champion["модель"] != best["модель"]:
        print(f"  Найточніша модель узагалі: {champion['модель']} "
              f"(RMSE {champion['RMSE']}), але потребує "
              f"{champion[col_f32]} КБ і в ОЗП не вміщується")

    fig_path = cfg.FIGURES / "fig11_esp32_footprint.png"
    figure(res, fig_path)
    print()
    print(f"Таблиця: {(cfg.TABLES / 'table09_esp32_footprint.csv').relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
