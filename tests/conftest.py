import os

import pytest

from core.generator import generate_projects
from core.prolog_runner import PrologEngine, PrologError, check_swipl


@pytest.fixture(scope="session")
def engine():
    """Движок SWI-Prolog. Тесты, которым он нужен, пропускаются, если swipl не найден.
    Путь можно задать переменной ILP_TEST_SWIPL."""
    custom = os.environ.get("ILP_TEST_SWIPL") or None
    try:
        info = check_swipl(custom)
    except PrologError as exc:
        pytest.skip(f"SWI-Prolog недоступен: {exc}")
    return PrologEngine(info.path, timeout=600)


@pytest.fixture(scope="session")
def small_dataset():
    return generate_projects(200, seed=11)
