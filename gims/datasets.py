"""
Завантаження та підготовка набору даних Autonomous Greenhouse Challenge.

Що всередині набору. Шість теплиць-відсіків вирощували черрі-помідор ~6
місяців (грудень 2019 — травень 2020) у Блейсвейку, Нідерланди. П'ять відсіків
керувалися алгоритмами команд-учасників, шостий («Reference») — досвідченим
агрономом. Дані писалися з кроком 5 хвилин.

Для нашої роботи важливо ось що: клімат у відсіках вимірювався повіреними
промисловими датчиками. Ми приймаємо ці показання за опорне (істинне) значення
вимірюваної величини і на них перевіряємо, що станеться, якщо ту саму величину
міряти бюджетними датчиками нашого приладу.

Файли, які використовуємо:
  <Відсік>/GreenhouseClimate.csv — клімат усередині + уставки + механізми
  <Відсік>/GrodanSens.csv        — вологість і температура мінераловатного субстрату
  Weather/Weather.csv            — зовнішня погода (спільна для всіх відсіків)
"""
from __future__ import annotations

import pandas as pd

from . import config as cfg

# Відсіки теплиці. Reference — керування людиною, решта — командами ШІ.
COMPARTMENTS = ["Reference", "AICU", "Automatoes", "Digilog", "IUACAAS", "TheAutomators"]

# --------------------------------------------------------------------------
# Відповідність стовпців набору даних нашим вимірювальним каналам.
# Ліворуч — як в AGC, праворуч — як у нас.
# --------------------------------------------------------------------------
CLIMATE_MAP = {
    "Tair": "T_air",        # температура повітря, °C        -> канал DHT22
    "Rhair": "RH_air",      # відносна вологість, %          -> канал DHT22
    "CO2air": "CO2_air",    # концентрація CO2, ppm          -> канал SGP30
    "Tot_PAR": "PAR",       # сумарний ФАР, мкмоль/(м²·с)    -> канал BH1750
}

SUBSTRATE_MAP = {
    "WC_slab1": "SOIL",     # вологовміст субстрату, %       -> ємнісний датчик
    "t_slab1": "T_soil",    # температура субстрату, °C
}

# Зовнішні умови — сильні предиктори; наш прилад їх не міряє, але в реальній
# системі вони доступні (метеостанція теплиці або прогноз погоди)
WEATHER_MAP = {
    "Tout": "T_out",        # температура надворі, °C
    "Rhout": "RH_out",      # вологість надворі, %
    "Iglob": "I_glob",      # сумарна сонячна радіація, Вт/м²
    "Windsp": "wind",       # швидкість вітру, м/с
    "Rain": "rain",         # опади, 0/1
}

# Стани виконавчих механізмів: саме вони є причиною зміни клімату, тому як
# ознаки для прогнозу дуже корисні
ACTUATOR_MAP = {
    "VentLee": "vent_lee",       # положення підвітряних кватирок, %
    "Ventwind": "vent_wind",     # положення навітряних кватирок, %
    "PipeLow": "pipe_low",       # температура нижнього контуру опалення, °C
    "PipeGrow": "pipe_grow",     # температура ростового контуру опалення, °C
    "AssimLight": "lamps",       # досвічування, %
    "EnScr": "scr_energy",       # енергозбережна завіса, %
    "BlackScr": "scr_black",     # затінювальна завіса, %
    "co2_dos": "co2_dosing",     # подача CO2, кг/(га·год)
}

# Переведення ФАР в освітленість. Для сонячного спектра прийнято ~54 лк на
# 1 мкмоль/(м²·с). Коефіцієнт СИЛЬНО залежить від спектра джерела, під
# світлодіодним досвічуванням він інший. У роботі це треба застерегти як
# прийняте припущення (і це саме по собі методична похибка каналу BH1750,
# який міряє освітленість, а рослині потрібен ФАР).
PAR_TO_LUX = 54.0

# Фізично допустимі межі вимірюваних величин.
# Опорний ряд AGC — це теж показання реальної апаратури, і в ньому трапляються
# явні артефакти (від'ємна вологість, від'ємний CO2 — слід збоїв апаратури або
# обробки). Такі відліки не можна вважати «істинним значенням», тому позначаємо
# їх як пропуски ДО того, як будувати дослід.
PHYSICAL_LIMITS = {
    "T_air": (0.0, 45.0),        # повітря в опалюваній теплиці
    "RH_air": (10.0, 100.0),     # відносна вологість за визначенням
    "CO2_air": (200.0, 3000.0),  # від вуличного фону до межі подачі CO2
    "LUX": (0.0, 150000.0),      # від 0 уночі до прямого сонця
    "SOIL": (0.0, 100.0),        # вологовміст субстрату, %
    "T_soil": (0.0, 45.0),
    "T_out": (-30.0, 45.0),
    "RH_out": (0.0, 100.0),
    "I_glob": (0.0, 1200.0),     # сонячна стала на поверхні Землі
}


def validate_physical(df: pd.DataFrame, report: bool = False) -> pd.DataFrame:
    """Позначити як пропуски відліки поза фізично допустимими межами."""
    out = df.copy()
    stats = {}
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col not in out.columns:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        n_bad = int(bad.sum())
        if n_bad:
            stats[col] = n_bad
            out.loc[bad, col] = pd.NA
    if report and stats:
        print("Відбраковано відліків поза фізичними межами:")
        for col, n in stats.items():
            print(f"  {col:9} {n:6d}  ({n / len(out) * 100:.2f} %)")
    return out


def despike(
    df: pd.DataFrame,
    channels: list[str],
    window: int = 13,
    n_sigma: float = 6.0,
    report: bool = False,
) -> pd.DataFrame:
    """Прибрати одиничні викиди опорного ряду фільтром Хампеля.

    Навіть повірена промислова апаратура дає рідкісні одиничні збої: у даних
    AGC трапляються провали температури до 0.5 °C за нормальних сусідніх
    значень. Такі відліки — не фізика теплиці, а збій апаратури, і в опорному
    ряді їм не місце.

    Фільтр Хампеля: відлік визнається викидом, якщо він відстоїть від ковзної
    медіани більше ніж на n_sigma робастних СКВ, оцінених через медіанне
    абсолютне відхилення (MAD). Робастна оцінка потрібна, щоб сам викид не
    роздував поріг.

    Поріг n_sigma=6 обрано навмисно консервативним: мета — зняти явні збої, а
    не згладити справжню динаміку клімату.
    """
    out = df.copy()
    stats = {}
    for col in channels:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        med = s.rolling(window, center=True, min_periods=3).median()
        mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
        # 1.4826 — множник, що приводить MAD до СКВ для нормального закону
        sigma = 1.4826 * mad
        bad = (s - med).abs() > n_sigma * sigma
        bad &= sigma > 0  # там, де сигнал сталий, MAD=0 — викидів немає
        n_bad = int(bad.sum())
        if n_bad:
            stats[col] = n_bad
            out.loc[bad, col] = pd.NA
    if report and stats:
        print("Знято одиничних викидів (фільтр Хампеля):")
        for col, n in stats.items():
            print(f"  {col:9} {n:6d}  ({n / len(out) * 100:.3f} %)")
    return out


def _excel_time_to_datetime(series: pd.Series) -> pd.Series:
    """Стовпець %time — дата у форматі Excel (днів від 1899-12-30)."""
    days = pd.to_numeric(series, errors="coerce")
    return pd.Timestamp("1899-12-30") + pd.to_timedelta(days, unit="D")


def _read_agc_csv(path, colmap: dict[str, str]) -> pd.DataFrame:
    """Прочитати csv з AGC: розібрати час, відібрати і перейменувати стовпці.

    У файлах трапляються нечислові заглушки, тому всі значення приводимо до
    числа примусово (що не розібралося — стає пропуском).
    """
    df = pd.read_csv(path, low_memory=False)
    ts = _excel_time_to_datetime(df["%time"])

    present = {src: dst for src, dst in colmap.items() if src in df.columns}
    out = df[list(present)].apply(pd.to_numeric, errors="coerce")
    out = out.rename(columns=present)
    out.index = ts
    out.index.name = "time"

    # Відліки без дійсного часу непотрібні
    out = out[out.index.notna()]
    # Набір даних подекуди містить повтори відліків за часом
    out = out[~out.index.duplicated(keep="first")]
    return out.sort_index()


def load_compartment(
    compartment: str = "Reference",
    with_weather: bool = True,
    with_actuators: bool = True,
) -> pd.DataFrame:
    """Зібрати єдину таблицю опорних («істинних») значень по одному відсіку.

    Повертає DataFrame з індексом-часом і кроком 5 хвилин.
    """
    if compartment not in COMPARTMENTS:
        raise ValueError(f"Невідомий відсік {compartment!r}, доступні: {COMPARTMENTS}")

    base = cfg.AGC_EXTRACT_DIR / compartment
    if not base.exists():
        raise FileNotFoundError(
            f"Немає даних у {base}. Спочатку виконайте:\n"
            "  .venv/bin/python scripts/download_data.py"
        )

    climate_map = dict(CLIMATE_MAP)
    if with_actuators:
        climate_map.update(ACTUATOR_MAP)

    df = _read_agc_csv(base / "GreenhouseClimate.csv", climate_map)

    sub = _read_agc_csv(base / "GrodanSens.csv", SUBSTRATE_MAP)
    df = df.join(sub, how="left")

    if with_weather:
        wx = _read_agc_csv(cfg.AGC_EXTRACT_DIR / "Weather" / "Weather.csv", WEATHER_MAP)
        df = df.join(wx, how="left")

    # ФАР -> освітленість: BH1750 міряє саме освітленість у люксах
    if "PAR" in df.columns:
        df["LUX"] = df["PAR"] * PAR_TO_LUX

    # Приводимо до строгої сітки 5 хвилин: набір даних подекуди має розриви
    step = f"{cfg.SAMPLE_PERIOD_S}s"
    df = df.resample(step).mean()

    df.attrs["compartment"] = compartment
    df.attrs["sample_period_s"] = cfg.SAMPLE_PERIOD_S
    return df


def clean_reference(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    max_gap: int = 6,
    report: bool = False,
) -> pd.DataFrame:
    """Підготувати опорний ряд: прибрати пропуски, лишити суцільний період.

    Опорний сигнал має бути без дірок — усі пропуски в досліді ми вносимо
    самі, моделлю похибки. Тому:
      1) відбраковуємо фізично неможливі значення;
      2) знімаємо одиничні збої апаратури;
      3) короткі пропуски (до max_gap відліків = 30 хв) закриваємо
         інтерполяцією;
      4) беремо найдовшу суцільну ділянку без пропусків у цільових каналах.
    """
    channels = channels or list(cfg.CHANNELS)
    have = [c for c in channels if c in df.columns]

    # 1) відбраковуємо фізично неможливі значення, інакше вони «протечуть»
    #    в опорний ряд і спотворять увесь дослід
    out = validate_physical(df, report=report)
    out[have] = out[have].astype(float)
    # 2) знімаємо одиничні збої апаратури
    out = despike(out, have, report=report)
    # 3) короткі пропуски закриваємо інтерполяцією
    out[have] = out[have].interpolate(method="time", limit=max_gap, limit_area="inside")

    ok = out[have].notna().all(axis=1)
    if not ok.any():
        raise ValueError("Не лишилося жодного відліку без пропусків")

    # 4) шукаємо найдовшу неперервну ділянку, де всі канали дійсні
    grp = (~ok).cumsum()[ok]
    longest = grp.value_counts().idxmax()
    keep = grp[grp == longest].index
    out = out.loc[keep]

    # Решту (допоміжних) стовпців дозаповнюємо протяжкою
    out = out.ffill().bfill()

    out.attrs.update(df.attrs)
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Короткий підсумок по каналах — знадобиться як таблиця в роботу."""
    rows = []
    for ch, meta in cfg.CHANNELS.items():
        if ch not in df.columns:
            continue
        s = df[ch]
        rows.append(
            {
                "канал": ch,
                "величина": meta["descr"],
                "датчик": meta["sensor"],
                "од. вим.": meta["unit"],
                "мін": s.min(),
                "середнє": s.mean(),
                "макс": s.max(),
                "СКВ": s.std(),
                "пропусків,%": s.isna().mean() * 100,
            }
        )
    return pd.DataFrame(rows).round(2)
