"""Настройки приложения: переменные окружения и необязательный файл .env."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROLOG_DIR = PROJECT_ROOT / "prolog"
ENGINE_FILE = PROLOG_DIR / "ilp_engine.pl"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DATASET = DATA_DIR / "demo_dataset.json"


def _load_dotenv(path: Path) -> None:
    """Минимальный разбор .env (KEY=VALUE); уже заданные переменные не трогаем."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def swipl_path_setting() -> str | None:
    """Путь к swipl из переменной SWIPL_PATH (пусто -> автоопределение)."""
    value = os.environ.get("SWIPL_PATH", "").strip()
    return value or None


def timeout_setting(default: int = 300) -> int:
    try:
        return max(5, int(os.environ.get("ILP_TIMEOUT_SECONDS", default)))
    except ValueError:
        return default
