"""Генератор синтетических проектов.

Метки строятся «по построению»: сначала решается, какие ошибки внедрить в проект,
затем факты создаются так, чтобы внедрённая ошибка в них присутствовала, а для
остальных классов — гарантированно отсутствовала. Готовых целевых меток в фактах
нет, и обучающий алгоритм (Prolog) генератора не использует.

Отвлекающие признаки (дистракторы), из-за которых одного предиката недостаточно:
  * пакеты/ключи/переменные, которые ОТСУТСТВУЮТ, но проекту НЕ нужны;
  * несоответствия версий у пакетов, которые проект НЕ требует (ограничения из
    constraints-файла) — это не ошибка проекта;
  * дополнительные системы сборки и среды выполнения.
Поэтому правило вида «пакет отсутствует» ошибочно, а «пакет требуется И отсутствует»
— верно; именно такую связь алгоритм должен найти сам.
"""
from __future__ import annotations

import random

from .dataset import ERROR_CLASSES, FACT_FIELDS, SCHEMA_VERSION

PACKAGES = [
    "numpy", "pandas", "scipy", "requests", "flask", "django", "fastapi", "sqlalchemy",
    "pytest", "matplotlib", "scikit-learn", "torch", "tensorflow", "pillow", "celery",
    "redis", "boto3", "pydantic", "jinja2", "click", "lxml", "beautifulsoup4", "gunicorn",
    "uvicorn", "httpx", "openpyxl", "psycopg2", "pyyaml", "cryptography", "aiohttp",
]
CONFIG_KEYS = [
    "DATABASE_URL", "SECRET_KEY", "DEBUG", "ALLOWED_HOSTS", "LOG_LEVEL", "CACHE_BACKEND",
    "REDIS_URL", "PORT", "HOST", "STATIC_ROOT", "TIMEZONE", "SENTRY_DSN", "MAX_WORKERS",
    "UPLOAD_DIR",
]
ENV_VARIABLES = [
    "API_TOKEN", "AWS_REGION", "DB_PASSWORD", "HOME_DIR", "SMTP_HOST", "SERVICE_URL",
    "JWT_SECRET", "REDIS_HOST", "STORAGE_BUCKET", "WORKER_COUNT", "FEATURE_FLAGS",
    "OAUTH_CLIENT_ID", "TMP_PATH", "APP_ENV",
]
BUILD_TOOLS = ["pip", "poetry", "conda", "pipenv", "npm", "maven", "gradle", "make"]
RUNTIMES = ["python", "node", "java", "go"]


def true_rule_specs(mode: str = "easy") -> list[dict]:
    """Структурные целевые правила для проверки восстановления гипотез генератора."""
    specs = [
        ("missing_dependency", "required_packages", "absent_packages"),
        ("incompatible_version", "required_packages", "version_mismatches"),
        ("missing_config_key", "required_config_keys", "absent_config_keys"),
        ("missing_environment_variable", "required_env_variables", "absent_env_variables"),
    ]
    predicates = {
        "required_packages": "requires_package",
        "absent_packages": "package_absent",
        "version_mismatches": "version_mismatch",
        "required_config_keys": "requires_config_key",
        "absent_config_keys": "config_key_absent",
        "required_env_variables": "requires_env_variable",
        "absent_env_variables": "env_variable_absent",
    }
    result = []
    for target, required, broken in specs:
        body = [{"predicate": predicates[required], "var": 1},
                {"predicate": predicates[broken], "var": 1}]
        if mode == "hard" and target in {"missing_dependency", "incompatible_version"}:
            predicate, value = (("uses_build_tool", "poetry") if target == "missing_dependency"
                                else ("uses_runtime", "python"))
            body.append({"predicate": predicate, "const": value})
        result.append({"target": target, "body": body})
    return result


def _inject_pairs(rng: random.Random, pool: list[str], n_required: tuple[int, int],
                  active: bool, p_distractor: float) -> tuple[list[str], list[str]]:
    """required, absent для одной пары (requires_X, X_absent)."""
    required = rng.sample(pool, rng.randint(*n_required))
    others = [x for x in pool if x not in required]
    absent: list[str] = []
    if active:
        absent += rng.sample(required, rng.randint(1, min(2, len(required))))
    if others and rng.random() < p_distractor:
        absent += rng.sample(others, rng.randint(1, min(2, len(others))))
    return sorted(required), sorted(set(absent))


def generate_project(rng: random.Random, index: int, error_rate: float,
                     mode: str = "easy") -> dict:
    active = {c: rng.random() < error_rate for c in ERROR_CLASSES}

    # --- пакеты и версии
    required_pkgs, absent_pkgs = _inject_pairs(
        rng, PACKAGES, (2, 6), active["missing_dependency"], 0.45)
    mismatches: list[str] = []
    present_required = [p for p in required_pkgs if p not in absent_pkgs]
    if active["incompatible_version"]:
        if not present_required:        # нужен установленный пакет с неверной версией
            # пакет не должен быть ни требуемым, ни отсутствующим, иначе возникла бы
            # неразмеченная ошибка отсутствия зависимости
            extra = rng.choice([p for p in PACKAGES
                                if p not in required_pkgs and p not in absent_pkgs])
            required_pkgs = sorted(required_pkgs + [extra])
            present_required = [extra]
        mismatches += rng.sample(present_required, rng.randint(1, min(2, len(present_required))))
    free = [p for p in PACKAGES if p not in required_pkgs]
    if free and rng.random() < 0.35:    # дистрактор: ограничение для НЕ требуемого пакета
        mismatches += rng.sample(free, rng.randint(1, min(2, len(free))))

    # --- ключи конфигурации и переменные окружения
    req_keys, abs_keys = _inject_pairs(rng, CONFIG_KEYS, (1, 4), active["missing_config_key"], 0.40)
    req_env, abs_env = _inject_pairs(rng, ENV_VARIABLES, (1, 4),
                                     active["missing_environment_variable"], 0.40)

    tools = set(rng.sample(BUILD_TOOLS, rng.randint(1, 2)))
    runtimes = set(rng.sample(RUNTIMES, rng.randint(1, 2)))

    package_conflict = bool(set(required_pkgs) & set(absent_pkgs))
    version_conflict = bool(set(required_pkgs) & set(mismatches))

    if mode == "hard":
        # Ошибка missing_dependency наступает только при сборке poetry, а
        # incompatible_version — только для python. Отрицательные проекты с
        # теми же «пакетными» фактами намеренно лишены контекста, поэтому
        # правило длины 2 даёт FP, а правило длины 3 — нет.
        if active["missing_dependency"]:
            tools.add("poetry")
        else:
            if package_conflict:
                tools.discard("poetry")
            elif present_required and rng.random() < 0.35:
                # «почти ошибка»: пакет требуется и отсутствует, но контекста
                # poetry нет, поэтому по условию задачи это не ошибка.
                near = rng.choice(present_required)
                absent_pkgs = sorted(set(absent_pkgs) | {near})
                present_required = [p for p in present_required if p != near]
                package_conflict = True
                tools.discard("poetry")
            elif rng.random() < 0.5:
                # контекст сам по себе не должен быть диагнозом
                tools.add("poetry")
            else:
                tools.discard("poetry")
        if active["incompatible_version"]:
            runtimes.add("python")
        else:
            if version_conflict:
                runtimes.discard("python")
            elif present_required and rng.random() < 0.30:
                near = rng.choice(present_required)
                mismatches.append(near)
                version_conflict = True
                runtimes.discard("python")
            elif rng.random() < 0.5:
                runtimes.add("python")
            else:
                runtimes.discard("python")

    project = {
        "id": f"project_{index + 1:03d}",
        "labels": [c for c in ERROR_CLASSES if active[c]],
        "facts": {
            "required_packages": required_pkgs,
            "absent_packages": absent_pkgs,
            "version_mismatches": sorted(set(mismatches)),
            "required_config_keys": req_keys,
            "absent_config_keys": abs_keys,
            "required_env_variables": req_env,
            "absent_env_variables": abs_env,
            "build_tools": sorted(tools),
            "runtimes": sorted(runtimes),
        },
    }

    if mode == "partial":
        # Частичная наблюдаемость: часть полей неизвестна. Это не меняет
        # метки, но позволяет проверить третий статус диагностики.
        for fld in FACT_FIELDS:
            if rng.random() < 0.12:
                project["facts"][fld] = None
    return project


def generate_projects(n: int = 500, seed: int = 42, error_rate: float = 0.30,
                      mode: str = "easy") -> list[dict]:
    """Воспроизводимо: один и тот же (n, seed, error_rate, mode) даёт один и тот же набор."""
    if n < 1:
        raise ValueError("n должно быть положительным")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate должен лежать строго между 0 и 1")
    if mode not in {"easy", "hard", "partial"}:
        raise ValueError("mode должен быть easy, hard или partial")
    rng = random.Random(seed)
    return [generate_project(rng, i, error_rate, mode) for i in range(n)]


def generate_document(n: int = 500, seed: int = 42, error_rate: float = 0.30,
                      mode: str = "easy") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {"generator": "core.generator", "n_projects": n, "seed": seed,
                 "error_rate": error_rate, "mode": mode, "synthetic": True,
                 "true_rules": true_rule_specs(mode)},
        "projects": generate_projects(n, seed, error_rate, mode),
    }
