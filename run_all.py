"""
Запуск усього дослідження однією командою.

    .venv/bin/python run_all.py

Скрипт послідовно виконує всі етапи роботи і друкує звіт. Якщо набір даних ще
не завантажено — завантажить (потрібен інтернет, 8.4 МБ). Усі результати
складаються в results/figures і results/tables.

УВАГА: повний прогін триває близько години — навчається кілька сотень моделей.
Для швидкого показу запускайте окремі етапи (див. README).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gims import config as cfg  # noqa: E402

STEPS = [
    ("Завантаження набору даних AGC з 4TU.ResearchData", "scripts.download_data"),
    ("Підготовка опорного ряду (чистка, перевірка меж)", "scripts.prepare_data"),
    ("Аналіз придатності вимірювальних каналів", "scripts.demo"),
    ("E1. Базова точність прогнозування", "scripts.exp01_baseline"),
    ("E2-CO2. Обґрунтування вибору датчика CO2", "scripts.exp02_co2_sensor"),
    ("E2. Деградація точності від рівня похибки", "scripts.exp03_noise_level"),
    ("E3. Внесок складових похибки", "scripts.exp04_error_components"),
    ("E4. Сенсорна фузія: цінність каналів", "scripts.exp05_fusion"),
    ("E5. Програмна компенсація похибки", "scripts.exp06_compensation"),
    ("E6. Виявлення відмов каналу", "scripts.exp07_fault_detection"),
    ("E7. Реалізовність моделей на ESP32", "scripts.exp08_esp32_footprint"),
]


def banner(text: str, char: str = "=") -> None:
    print()
    print(char * 78)
    print(text)
    print(char * 78)


def main() -> None:
    t_start = time.time()

    for i, (title, module) in enumerate(STEPS, 1):
        banner(f"ЕТАП {i} з {len(STEPS)}: {title}")
        mod = __import__(module, fromlist=["main"])
        mod.main()

    banner("ГОТОВО", "=")
    print(f"Час виконання: {time.time() - t_start:.1f} с")
    print()

    print("Рисунки:")
    for p in sorted(cfg.FIGURES.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")
    print()
    print("Таблиці:")
    for p in sorted(cfg.TABLES.glob("*.csv")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
