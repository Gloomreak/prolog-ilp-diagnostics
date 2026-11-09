"""Проверка Python-зависимостей и короткого сквозного сценария."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluation import environment_info  # noqa: E402
from core.generator import generate_projects  # noqa: E402
from core.prolog_runner import PrologEngine, PrologError, check_swipl  # noqa: E402


def main() -> int:
    try:
        info = check_swipl()
        engine = PrologEngine(info.path)
        ping = engine.ping()
        if ping.get("status") != "ok":
            raise PrologError("SWI-Prolog ping не вернул status=ok.")
        projects = generate_projects(40, seed=7)
        trained = engine.train(projects, {
            "max_body_length": 2, "min_support": 1, "min_precision": 0.5,
            "max_rules_per_class": 2, "max_candidates": 2000,
        })
        if trained.get("status") != "ok":
            raise PrologError("Пробное обучение не вернуло status=ok.")
    except PrologError as exc:
        print(f"Проверка не пройдена: {exc}", file=sys.stderr)
        return 1

    print("Проверка пройдена.")
    print(f"Python: {sys.version.split()[0]}")
    for name, version in environment_info(engine).items():
        print(f"{name}: {version}")
    print(f"Пробное обучение: {len(trained['model']['rules'])} правил.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
