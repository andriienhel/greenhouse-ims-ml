"""
Внесення відмов вимірювального каналу з точним розмічуванням.

У модулі sensors.py відмови задані ймовірнісно — як фонова властивість каналу.
Тут інше: потрібні відмови з відомими межами в часі, щоб можна було перевірити,
чи знаходить їх алгоритм виявлення і з якою затримкою.

Види відмов відповідають тому, що реально трапляється з бюджетними датчиками:

  stuck   залипання: датчик віддає одне й те саме значення. Типово для DHT22
          при збої обміну, коли драйвер повертає останнє вдале читання.
  outage  обрив: канал перестає відповідати, відліки втрачаються. Відгнилий
          провід, відійшов роз'єм, сів акумулятор.
  drift   сповзання: показання повільно відходять від істинних. Забруднення
          чутливого елемента, корозія електродів, втрата нульової лінії.
  spikes  викиди: одиничні грубо неправильні відліки. Наведення на довгій
          лінії, збої живлення.

Відмова типу «сповзання» — найнебезпечніша для системи керування: показання
лишаються правдоподібними, жодних ознак несправності немає, і система впевнено
підтримує не той режим. Саме її найважче виявити.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FAULT_TYPES = {
    "stuck": "залипання значення",
    "outage": "обрив каналу",
    "drift": "сповзання показань",
    "spikes": "одиничні викиди",
}


@dataclass
class FaultEpisode:
    """Один епізод відмови з відомими межами."""

    channel: str
    kind: str
    start: int          # індекс початку, у відліках
    length: int         # тривалість, у відліках
    magnitude: float    # величина відмови в одиницях вимірювання каналу

    @property
    def end(self) -> int:
        return self.start + self.length


def inject(
    series: pd.Series,
    episodes: list[FaultEpisode],
) -> tuple[pd.Series, np.ndarray]:
    """Внести відмови в ряд показань.

    Повертає зіпсований ряд і булеву маску: True там, де відмова діє.
    Маска — це еталон, за яким оцінюється якість виявлення.
    """
    x = series.to_numpy(dtype=float).copy()
    mask = np.zeros(x.size, dtype=bool)

    for ep in episodes:
        a, b = ep.start, min(ep.end, x.size)
        if a >= x.size:
            continue

        if ep.kind == "stuck":
            # Значення завмирає на тому, що було в момент відмови
            x[a:b] = x[a - 1] if a > 0 else x[0]

        elif ep.kind == "outage":
            x[a:b] = np.nan

        elif ep.kind == "drift":
            # Лінійно наростаюче відхилення: до кінця епізоду досягає magnitude
            n = b - a
            x[a:b] = x[a:b] + np.linspace(0, ep.magnitude, n)
            # Після завершення епізоду зміщення зберігається: датчик так і
            # лишається розлагодженим, поки його не обслужать
            x[b:] = x[b:] + ep.magnitude
            # Тому й несправним канал вважається весь час після початку
            # сповзання, а не лише вікно наростання. Якщо розмітити лише
            # вікно, то правильні спрацювання на подальшій ділянці будуть
            # помилково зараховані як хибні тривоги.
            mask[a:] = True
            continue

        elif ep.kind == "spikes":
            n = b - a
            rng = np.random.default_rng(ep.start)
            # Викиди рідкісні: приблизно кожен п'ятий відлік епізоду
            hit = rng.random(n) < 0.2
            sign = rng.choice([-1.0, 1.0], size=n)
            x[a:b] = np.where(hit, x[a:b] + sign * ep.magnitude, x[a:b])
            mask[a:b] = hit
            continue

        else:
            raise ValueError(f"Невідомий вид відмови: {ep.kind}")

        mask[a:b] = True

    return pd.Series(x, index=series.index, name=series.name), mask


def make_episodes(
    channel: str,
    n_samples: int,
    kind: str,
    magnitude: float,
    n_episodes: int = 6,
    length: int = 24,
    rng: np.random.Generator | None = None,
    margin: int = 48,
) -> list[FaultEpisode]:
    """Розкласти кілька епізодів відмови по ряду так, щоб вони не перетиналися.

    margin — відступ від країв і між епізодами, щоб вони не зливалися і
    алгоритм виявлення встигав повернутися в нормальний режим.
    """
    rng = rng or np.random.default_rng(0)
    usable = n_samples - 2 * margin - length
    if usable <= 0 or n_episodes <= 0:
        return []

    # Розкладаємо епізоди по рівних відрізках, усередині відрізка — випадково.
    # Так вони гарантовано не перетинаються.
    step = usable // n_episodes
    episodes = []
    for i in range(n_episodes):
        lo = margin + i * step
        hi = max(lo + 1, lo + step - length - margin)
        start = int(rng.integers(lo, hi))
        episodes.append(FaultEpisode(channel, kind, start, length, magnitude))
    return episodes
