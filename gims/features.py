"""
Формування навчальної вибірки для задачі прогнозування.

Тут закладено два принципові методичні рішення, які варто розуміти.

РІШЕННЯ 1. Що подаємо на вхід і за чим навчаємо.
Реальний прилад бачить лише показання своїх датчиків — з усіма похибками. Він
не має доступу до істинних значень. Тому й ознаки, і цільову змінну для
навчання беремо зі СПОТВОРЕНИХ даних: це чесна імітація того, що станеться
при реальному розгортанні приладу.

РІШЕННЯ 2. За чим оцінюємо якість.
А ось оцінку точності ведемо за ІСТИННИМИ значеннями. Причина в тому, що
системі керування теплицею потрібна фактична температура, а не те, що показав
датчик. Прогноз, який точно передбачає хибні показання, марний.

Різниця між цими двома ролями даних і є суттю дослідження: ми вимірюємо,
скільки точності втрачає система через те, що навчається на спотворених даних.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# Лаги, які подаємо як ознаки (у відліках по 5 хвилин).
# Беремо не всі поспіль, а логарифмічно розріджені: 5 хв, 10, 15, 30 хв,
# 1 година, 2 години тому. Так охоплюємо і швидку, і повільну динаміку, не
# роздуваючи кількість ознак.
LAGS = [0, 1, 2, 3, 6, 12, 24]

# Вікна для ковзних статистик (середнє і розкид за останню годину / дві)
ROLL_WINDOWS = [12, 24]


def impute_device_side(df: pd.DataFrame) -> pd.DataFrame:
    """Заповнити пропуски так, як це зробив би сам прилад.

    При втраті відліку (збій CRC, тайм-аут шини) мікроконтролер не може
    зазирнути в майбутнє — він повторює останнє вдале вимірювання. Саме це й
    моделюємо протяжкою вперед. Досконаліші способи відновлення перевіряються
    окремо в досліді E5.
    """
    return df.ffill().bfill()


def build_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
    time_features: bool = True,
) -> pd.DataFrame:
    """Побудувати матрицю ознак із часових рядів каналів."""
    lags = LAGS if lags is None else lags
    roll_windows = ROLL_WINDOWS if roll_windows is None else roll_windows

    src = impute_device_side(df[feature_cols])
    parts: list[pd.DataFrame] = []

    # Запізнілі значення самих каналів
    for lag in lags:
        block = src.shift(lag)
        block.columns = [f"{c}_lag{lag}" for c in block.columns]
        parts.append(block)

    # Ковзні статистики: середнє задає рівень, розкид — мінливість
    for w in roll_windows:
        m = src.rolling(w).mean()
        m.columns = [f"{c}_mean{w}" for c in m.columns]
        parts.append(m)
        s = src.rolling(w).std()
        s.columns = [f"{c}_std{w}" for c in s.columns]
        parts.append(s)

    # Швидкість зміни за останні 15 хвилин — важлива для інерційних величин
    d = src - src.shift(3)
    d.columns = [f"{c}_d15" for c in d.columns]
    parts.append(d)

    X = pd.concat(parts, axis=1)

    if time_features:
        # Час доби кодуємо парою синус/косинус, щоб 23:55 і 00:05 опинилися
        # поруч, а не на різних кінцях шкали
        minutes = X.index.hour * 60 + X.index.minute
        angle = 2 * np.pi * minutes / (24 * 60)
        X["tod_sin"] = np.sin(angle)
        X["tod_cos"] = np.cos(angle)

    return X


def make_supervised(
    df_observed: pd.DataFrame,
    target: str,
    horizon: int,
    feature_cols: list[str] | None = None,
    df_truth: pd.DataFrame | None = None,
) -> dict:
    """Сформувати задачу навчання з учителем для одного каналу й горизонту.

    df_observed — те, що бачить прилад (показання датчиків)
    df_truth    — істинні значення, лише для оцінки якості
    horizon     — на скільки відліків уперед прогнозуємо

    Повертає словник із матрицею ознак, цільовою змінною (спостережуваною),
    істинним значенням для оцінки та наївним прогнозом як точкою відліку.
    """
    feature_cols = feature_cols or [c for c in cfg.CHANNELS if c in df_observed.columns]

    X = build_features(df_observed, feature_cols)

    obs = impute_device_side(df_observed[[target]])[target]
    # Ціль навчання — спостережуване значення через horizon відліків
    y_obs = obs.shift(-horizon)
    # Наївний прогноз: «величина не зміниться». Точка відліку для skill score
    y_naive = obs.copy()

    if df_truth is not None:
        y_true = df_truth[target].shift(-horizon)
    else:
        y_true = y_obs.copy()

    frame = pd.concat(
        [X, y_obs.rename("__y_obs"), y_true.rename("__y_true"),
         y_naive.rename("__y_naive")],
        axis=1,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "X": frame.drop(columns=["__y_obs", "__y_true", "__y_naive"]),
        "y_obs": frame["__y_obs"],
        "y_true": frame["__y_true"],
        "y_naive": frame["__y_naive"],
        "index": frame.index,
        "target": target,
        "horizon": horizon,
    }


def chronological_split(
    data: dict,
    train: float | None = None,
    val: float | None = None,
) -> dict:
    """Поділити вибірку за часом: навчання — перевірка — контроль.

    Перемішувати часовий ряд не можна категорично: модель, навчена на даних із
    майбутнього, покаже фіктивно високу точність (витік інформації). Тому
    ріжемо строго за хронологією.
    """
    train = cfg.SPLIT_TRAIN if train is None else train
    val = cfg.SPLIT_VAL if val is None else val

    n = len(data["index"])
    i_tr = int(n * train)
    i_va = int(n * (train + val))

    out = {"target": data["target"], "horizon": data["horizon"]}
    for part, sl in (("train", slice(0, i_tr)),
                     ("val", slice(i_tr, i_va)),
                     ("test", slice(i_va, n))):
        out[part] = {
            "X": data["X"].iloc[sl],
            "y_obs": data["y_obs"].iloc[sl],
            "y_true": data["y_true"].iloc[sl],
            "y_naive": data["y_naive"].iloc[sl],
            "index": data["index"][sl],
        }
    return out
