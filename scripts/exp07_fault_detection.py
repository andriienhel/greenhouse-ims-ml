"""
ДОСЛІД E6. Виявлення відмов вимірювального каналу.

Прогноз мікроклімату марний, якщо система не помічає, що датчик бреше. Тут
перевіряється, чи можна виявляти відмови каналів тією самою моделлю, що вже
побудована для прогнозування, без додаткової апаратури.

ПРИНЦИП: АНАЛІТИЧНА НАДМІРНІСТЬ.
Величина оцінюється за ІНШИМИ каналами — рештою датчиків приладу, погодою і
станами виконавчих механізмів, — але не за власною історією. Розбіжність між
показанням датчика і такою оцінкою (нев'язка) і слугує ознакою відмови.

Чому не можна передбачати канал за його ж минулими значеннями: при відмові
спотворюється і вхід алгоритму, і його вихід. Залиплий датчик чудово передбачає
сам себе, і нев'язка лишається нульовою — відмова проходить непоміченою.

ДВА ПРАВИЛА ВИЯВЛЕННЯ:
  поріг за нев'язкою — спрацьовує на різкі відмови (викиди, стрибки);
  накопичувальна сума (CUSUM) — накопичує малі односпрямовані відхилення і тому
  ловить повільне сповзання показань, яке за миттєвим порогом невідрізниме
  від норми.

Сповзання — найнебезпечніший вид відмови: показання лишаються правдоподібними,
зовнішніх ознак несправності немає, і система керування впевнено підтримує
неправильний режим.

Запуск:  .venv/bin/python scripts/exp07_fault_detection.py
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
from gims.faults import FAULT_TYPES, FaultEpisode, inject, make_episodes  # noqa: E402
from gims.features import build_features, impute_device_side  # noqa: E402
from gims.models import make_hist_gb  # noqa: E402
from gims.sensors import corrupt_frame  # noqa: E402

REF_FILE = cfg.DATA_PROCESSED / "reference_Reference.parquet"

N_SEEDS = 3
EPISODE_LEN = 24        # 2 години
N_EPISODES = 6
COLORS = {"T_air": "#1f4e79", "RH_air": "#2e7d32", "CO2_air": "#c62828"}

WEATHER = ["T_out", "RH_out", "I_glob", "wind", "rain"]
ACTUATORS = ["vent_lee", "vent_wind", "pipe_low", "pipe_grow",
             "lamps", "scr_energy", "scr_black", "co2_dosing"]

# Величина відмови по каналах: приблизно вдвічі більша за добовий розкид шуму,
# але в межах правдоподібного для показань датчика
MAGNITUDE = {"T_air": 2.0, "RH_air": 8.0, "CO2_air": 150.0}

# Параметри правил виявлення.
# Пороги не призначаються довільно, а підбираються за допустимою частотою
# хибних тривог на справному каналі. Так порівняння каналів і видів відмов стає
# чесним: усі алгоритми працюють за однакової ціни помилки. Одна хибна тривога
# на добу — розумний компроміс для теплиці: агроном не реагуватиме на алгоритм,
# який кричить щогодини.
TARGET_FALSE_ALARMS_PER_DAY = 1.0
CUSUM_SLACK = 0.5       # нечутливість накопичувача, в СКВ нев'язки


def build_estimator(ref: pd.DataFrame, obs: pd.DataFrame, target: str):
    """Навчити оцінку каналу за іншими джерелами інформації.

    Навчання ведеться на справному каналі (зі звичайною паспортною похибкою,
    але без внесених відмов) — так само, як алгоритм налаштовувався б у період
    нормальної експлуатації.
    """
    src = ref.copy()
    for c in obs.columns:
        src[c] = obs[c]

    # Ключовий момент: сам цільовий канал в ознаки НЕ входить
    others = [c for c in cfg.CHANNELS if c != target]
    cols = [c for c in others + WEATHER + ACTUATORS if c in src.columns]

    X = build_features(src, cols)
    y = impute_device_side(src[[target]])[target]

    frame = pd.concat([X, y.rename("__y")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    # Вибірка ділиться на три частини, і це принципово.
    #   fit  — на ній навчається оцінка каналу;
    #   cal  — на ній вимірюється розкид нев'язки для налаштування порогів;
    #   test — на ній вносяться відмови і перевіряється виявлення.
    # Налаштовувати пороги за навчальною частиною не можна: на ній модель
    # підігнана, нев'язка занижена, і пороги виходять недосяжно вузькими. На
    # реальних даних такий алгоритм спрацьовує майже на кожному відліку.
    n_fit = int(len(frame) * cfg.SPLIT_TRAIN)
    n_cal = int(len(frame) * (cfg.SPLIT_TRAIN + cfg.SPLIT_VAL))

    X_fit, y_fit = frame.iloc[:n_fit].drop(columns="__y"), frame.iloc[:n_fit]["__y"]
    X_cal, y_cal = (frame.iloc[n_fit:n_cal].drop(columns="__y"),
                    frame.iloc[n_fit:n_cal]["__y"])

    model = make_hist_gb(seed=cfg.SEED)
    model.fit(X_fit, y_fit)

    # Робастні оцінки: медіана і медіанне абсолютне відхилення. Звичайні
    # середнє і СКВ чутливі до рідкісних великих нев'язок, через які поріг
    # роздувається і дрібні відмови перестають виявлятися.
    resid_cal = y_cal.to_numpy() - model.predict(X_cal)
    mu = float(np.median(resid_cal))
    sigma = float(1.4826 * np.median(np.abs(resid_cal - mu)))
    return model, frame, n_cal, mu, sigma, resid_cal


def _cusum_stat(z: np.ndarray, slack: float) -> np.ndarray:
    """Накопичена сума відхилень без скидання — для налаштування порога."""
    s_pos = np.zeros(z.size)
    s_neg = np.zeros(z.size)
    for i in range(1, z.size):
        s_pos[i] = max(0.0, s_pos[i - 1] + z[i] - slack)
        s_neg[i] = max(0.0, s_neg[i - 1] - z[i] - slack)
    return np.maximum(s_pos, s_neg)


def calibrate_thresholds(resid_cal: np.ndarray, mu: float,
                         sigma: float) -> tuple[float, float]:
    """Підібрати пороги за допустимою частотою хибних тривог.

    Береться нев'язка на справному каналі і знаходиться такий поріг, вище якого
    опиняється рівно задана частка відліків. Тоді на справному обладнанні
    алгоритм спрацьовуватиме в середньому стільки разів на добу, скільки задано.

    Це стандартний для технічної діагностики прийом: спочатку фіксується ціна
    хибної тривоги, і вже за неї порівнюється чутливість.
    """
    z = (resid_cal - mu) / max(sigma, 1e-9)
    per_day = 86400 / cfg.SAMPLE_PERIOD_S
    q = min(1.0 - TARGET_FALSE_ALARMS_PER_DAY / per_day, 0.9999)

    z_thr = float(np.quantile(np.abs(z), q))
    h = float(np.quantile(_cusum_stat(z, CUSUM_SLACK), q))
    return z_thr, max(h, 1.0)


def detect(resid: np.ndarray, mu: float, sigma: float,
           z_thr: float, cusum_limit: float) -> np.ndarray:
    """Правила виявлення: миттєвий поріг плюс накопичувальна сума."""
    z = (resid - mu) / max(sigma, 1e-9)

    alarm_threshold = np.abs(z) > z_thr

    # Двобічний CUSUM: накопичує відхилення одного знака.
    # Після спрацювання накопичувач обнуляється — інакше, раз перевищивши
    # поріг, він лишається вище нього до кінця ряду і оголошує відмовою весь
    # решту час.
    alarm_cusum = np.zeros(z.size, dtype=bool)
    s_pos = s_neg = 0.0
    for i in range(z.size):
        s_pos = max(0.0, s_pos + z[i] - CUSUM_SLACK)
        s_neg = max(0.0, s_neg - z[i] - CUSUM_SLACK)
        if s_pos > cusum_limit or s_neg > cusum_limit:
            alarm_cusum[i] = True
            s_pos = s_neg = 0.0

    return alarm_threshold | alarm_cusum


def evaluate_detection(alarm: np.ndarray, truth: np.ndarray,
                       episodes_idx: list[tuple[int, int]]) -> dict:
    """Оцінити якість виявлення.

    Рахуємо за епізодами, а не за окремими відліками: на практиці важливо, що
    відмову помічено і як швидко, а не скільки саме відліків позначено.
    """
    detected, delays = 0, []
    for a, b in episodes_idx:
        seg = alarm[a:b]
        if seg.any():
            detected += 1
            delays.append(int(np.argmax(seg)))

    # Хибні тривоги рахуємо лише там, де відмов завідомо не було
    normal = ~truth
    false_alarms = int((alarm & normal).sum())
    hours = normal.sum() * cfg.SAMPLE_PERIOD_S / 3600

    return {
        "частка виявлення,%": (round(detected / len(episodes_idx) * 100, 1)
                               if episodes_idx else np.nan),
        "затримка, хв": (round(float(np.median(delays)) * cfg.SAMPLE_PERIOD_S / 60, 1)
                         if delays else np.nan),
        "хибних тривог на добу": round(false_alarms / max(hours, 1e-9) * 24, 2),
    }


def run(ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        print(f"\n  Канал {target}")
        for kind, kind_ua in FAULT_TYPES.items():
            acc = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(cfg.SEED + seed)
                obs = corrupt_frame(ref, cfg.SAMPLE_PERIOD_S, rng, scale=1.0)
                obs = obs.ffill().bfill()

                model, frame, n_cal, mu, sigma, resid_cal = build_estimator(
                    ref, obs, target)
                z_thr, cusum_limit = calibrate_thresholds(resid_cal, mu, sigma)

                # Відмови вносимо лише в контрольну частину: алгоритм навчений і
                # налаштований на справному каналі й бачить відмову вперше
                test_slice = slice(n_cal, len(frame))
                test_idx = frame.index[test_slice]
                n_test = len(test_idx)

                if kind == "drift":
                    # Сповзання не усувається саме собою: канал лишається
                    # розлагодженим до обслуговування. Тому епізод один і
                    # ставиться в середину контрольної вибірки — так лишається
                    # достатньо справної ділянки, щоб виміряти частоту хибних
                    # тривог, і достатньо несправної, щоб оцінити виявлення.
                    eps = [FaultEpisode(target, kind, int(n_test * 0.4),
                                        EPISODE_LEN, MAGNITUDE[target])]
                else:
                    eps = make_episodes(
                        target, n_test, kind, magnitude=MAGNITUDE[target],
                        n_episodes=N_EPISODES, length=EPISODE_LEN,
                        rng=np.random.default_rng(cfg.SEED + 100 + seed))

                faulty, mask = inject(obs.loc[test_idx, target], eps)

                # Обрив каналу прилад приховує протяжкою останнього значення:
                # типовий драйвер при невдалому читанні повертає попередній
                # результат. ВАЖЛИВИЙ НАСЛІДОК: після такої протяжки обрив стає
                # невідрізнимим від залипання — за даними це буквально один і
                # той самий сигнал, тому й показники виявлення в них збігаються.
                # Це не збіг чисел, а властивість постановки. Звідси практична
                # рекомендація: драйвер має позначати відсутні відліки явно, а
                # не підставляти останній вдалий — тоді обрив виявляється
                # тривіально і миттєво.
                faulty = faulty.ffill().bfill()

                X_te = frame.iloc[test_slice].drop(columns="__y")
                pred = model.predict(X_te)
                resid = faulty.to_numpy(dtype=float) - pred

                alarm = detect(resid, mu, sigma, z_thr, cusum_limit)
                # Для сповзання зоною відмови вважається весь час після початку
                if kind == "drift":
                    ep_idx = [(e.start, n_test) for e in eps if e.start < n_test]
                else:
                    ep_idx = [(e.start, min(e.end, n_test))
                              for e in eps if e.start < n_test]
                acc.append(evaluate_detection(alarm, mask, ep_idx))

            row = {
                "канал": target,
                "вид відмови": kind_ua,
                "код": kind,
                "частка виявлення,%": round(
                    float(np.mean([a["частка виявлення,%"] for a in acc])), 1),
                "затримка, хв": round(
                    float(np.nanmean([a["затримка, хв"] for a in acc])), 1),
                "хибних тривог на добу": round(
                    float(np.mean([a["хибних тривог на добу"] for a in acc])), 2),
            }
            rows.append(row)
            print(f"    {kind_ua:22} виявлено {row['частка виявлення,%']:5.1f} %  "
                  f"затримка {row['затримка, хв']:6.1f} хв  "
                  f"хибних тривог {row['хибних тривог на добу']:5.2f}/добу")
    return pd.DataFrame(rows)


def figure(res: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    kinds = list(FAULT_TYPES.values())
    x = np.arange(len(kinds))
    w = 0.26

    for i, target in enumerate(cfg.TARGETS):
        s = res[res["канал"] == target].set_index("вид відмови").loc[kinds]
        label = cfg.CHANNELS[target]["descr"]
        ax1.bar(x + (i - 1) * w, s["частка виявлення,%"], w,
                color=COLORS[target], label=label)
        ax2.bar(x + (i - 1) * w, s["затримка, хв"], w,
                color=COLORS[target], label=label)

    for ax, ylab, title in (
        (ax1, "виявлено епізодів, %", "А. Що вдається виявити"),
        (ax2, "затримка виявлення, хв", "Б. Як швидко"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(" ", "\n") for k in kinds], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.legend(fontsize=8)
    ax1.axhline(100, color="#999", ls="--", lw=1)

    fig.suptitle("E6. Виявлення відмов вимірювального каналу "
                 "методом аналітичної надмірності", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not REF_FILE.exists():
        raise SystemExit("Спочатку виконайте: .venv/bin/python scripts/prepare_data.py")
    ref = pd.read_parquet(REF_FILE)

    print("=" * 78)
    print("ДОСЛІД E6. Виявлення відмов вимірювального каналу")
    print("=" * 78)
    print(f"{N_EPISODES} епізодів по {EPISODE_LEN * cfg.SAMPLE_PERIOD_S // 60} хв "
          f"на кожен вид відмови, {N_SEEDS} прогони")

    res = run(ref)
    res.to_csv(cfg.TABLES / "table08_fault_detection.csv", index=False)

    print()
    print("=" * 78)
    print("ВИСНОВКИ")
    print("=" * 78)
    for kind, kind_ua in FAULT_TYPES.items():
        s = res[res["код"] == kind]
        print(f"  {kind_ua:22} виявлення "
              f"{s['частка виявлення,%'].mean():5.1f} % у середньому по каналах, "
              f"затримка {s['затримка, хв'].mean():5.1f} хв")

    fig_path = cfg.FIGURES / "fig10_fault_detection.png"
    figure(res, fig_path)
    print()
    print(f"Таблиця: {(cfg.TABLES / 'table08_fault_detection.csv').relative_to(cfg.ROOT)}")
    print(f"Рисунок: {fig_path.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
