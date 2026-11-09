"""Независимый Python-проверяльщик правил (ТОЛЬКО для тестов).

Он не обучает и не формирует правила: берёт уже выученное Prolog правило и
считает его покрытие на данных простым перебором. Совпадение с подсчётом Prolog
подтверждает корректность логического вывода и метрик TP/FP/FN.
"""
from __future__ import annotations

from core.dataset import PREDICATE_TO_FIELD

# тип переменной задаётся предикатом: пакеты, ключи, переменные, ...
_TYPE = {"requires_package": "package", "package_absent": "package", "version_mismatch": "package",
         "requires_config_key": "key", "config_key_absent": "key",
         "requires_env_variable": "env", "env_variable_absent": "env",
         "uses_build_tool": "tool", "uses_runtime": "runtime"}


def rule_covers(body: list[dict], project: dict) -> bool:
    lits = [(literal["predicate"], (_TYPE[literal["predicate"]], literal["var"]))
            for literal in body]

    def values(pred):
        return project["facts"].get(PREDICATE_TO_FIELD[pred]) or []

    def solve(i, env):
        if i == len(lits):
            return True
        pred, var = lits[i]
        for v in values(pred):
            if var in env and env[var] != v:
                continue
            new = dict(env)
            new[var] = v
            if solve(i + 1, new):
                return True
        return False

    return solve(0, {})


def counts(rule: dict, projects: list[dict]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for p in projects:
        cov = rule_covers(rule["body"], p)
        pos = rule["target"] in (p["labels"] or [])
        tp += cov and pos
        fp += cov and not pos
        fn += (not cov) and pos
    return tp, fp, fn


def ground_truth(project: dict) -> set[str]:
    """Независимое определение «истинных» ошибок по фактам (для проверки генератора)."""
    f = project["facts"]
    pairs = {"missing_dependency": ("required_packages", "absent_packages"),
             "incompatible_version": ("required_packages", "version_mismatches"),
             "missing_config_key": ("required_config_keys", "absent_config_keys"),
             "missing_environment_variable": ("required_env_variables", "absent_env_variables")}
    return {c for c, (a, b) in pairs.items() if set(f[a] or []) & set(f[b] or [])}
