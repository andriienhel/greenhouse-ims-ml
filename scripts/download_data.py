"""
Завантаження та розпакування набору даних Autonomous Greenhouse Challenge.

Джерело: 4TU.ResearchData, ліцензія CC0 (public domain).
DOI: 10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7

Запуск:  .venv/bin/python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gims import config as cfg  # noqa: E402


def _ssl_context() -> ssl.SSLContext:
    """Контекст TLS з кореневими сертифікатами.

    Збірка Python із python.org не використовує системне сховище сертифікатів
    macOS, тому беремо набір кореневих сертифікатів із пакета certifi.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_file_info() -> dict:
    """Спитати в API 4TU, де лежить файл і яка в нього контрольна сума."""
    req = urllib.request.Request(f"{cfg.AGC_API}/files")
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        files = json.load(r)
    if not files:
        raise RuntimeError("API 4TU не повернув жодного файлу")
    return files[0]


def download(info: dict, dest: Path) -> Path:
    if dest.exists():
        print(f"Архів уже завантажено: {dest}")
        return dest

    url = info["download_url"]
    size_mb = info["size"] / 1024 / 1024
    print(f"Завантажую {info['name']} ({size_mb:.1f} МБ)...")

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as r:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    tmp.rename(dest)

    # Перевіряємо цілісність за контрольною сумою з метаданих
    md5 = hashlib.md5(dest.read_bytes()).hexdigest()
    expected = info.get("computed_md5") or info.get("supplied_md5")
    if expected and md5 != expected:
        dest.unlink()
        raise RuntimeError(f"Контрольна сума не зійшлася: {md5} != {expected}")
    print(f"Готово, MD5 збігся: {md5}")
    return dest


def extract(archive: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Уже розпаковано: {out_dir}")
        return
    try:
        import py7zr
    except ImportError:
        raise SystemExit(
            "Потрібен py7zr для розпакування .7z:\n  .venv/bin/pip install py7zr"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Розпаковую в {out_dir}...")
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)
    print("Розпаковано.")


def main() -> None:
    info = fetch_file_info()
    archive = download(info, cfg.AGC_ARCHIVE)
    extract(archive, cfg.AGC_EXTRACT_DIR)

    files = sorted(p for p in cfg.AGC_EXTRACT_DIR.rglob("*") if p.is_file())
    print(f"\nУ наборі даних {len(files)} файлів.")


if __name__ == "__main__":
    main()
