"""Независимая проверка выученных правил на Python.

Нужна только для анализа и интерфейса (FP/FN по правилу, «почему не сработало»).
Правила по-прежнему выводит Prolog; этот модуль ничего не обучает.
"""
from __future__ import annotations

from collections.abc import Iterable

from .dataset import PREDICATE_TO_FIELD

TYPE_BY_PREDICATE = {
    "requires_package": "package", "package_absent": "package", "version_mismatch": "package",
    "requires_config_key": "config_key", "config_key_absent": "config_key",
    "requires_env_variable": "env_variable", "env_variable_absent": "env_variable",
    "uses_build_tool": "build_tool", "uses_runtime": "runtime",
}


def _values(project: dict, predicate: str) -> list[str]:
    fld = PREDICATE_TO_FIELD.get(predicate)
    if not fld:
        return []
    values = project.get("facts", {}).get(fld)
    return list(values or [])


def _known(project: dict, predicate: str) -> bool:
    field = PREDICATE_TO_FIELD.get(predicate)
    return field is not None and project.get("facts", {}).get(field) is not None


def _ordered(body: Iterable[dict]) -> list[dict]:
    positives = [lit for lit in body if not lit.get("negated") and not lit.get("neq")]
    negatives = [lit for lit in body if lit.get("negated")]
    neqs = [lit for lit in body if lit.get("neq")]
    return positives + negatives + neqs


def covers(rule: dict, project: dict) -> bool:
    """Проверяет тело правила на фактах проекта (закрытый мир для neg)."""
    env: dict[tuple[str, int], str] = {}

    def solve(lits: list[dict], i: int) -> bool:
        if i == len(lits):
            return True
        lit = lits[i]
        if lit.get("neq"):
            typ = lit["type"]
            v1 = env.get((typ, lit["left"]))
            v2 = env.get((typ, lit["right"]))
            if v1 is None or v2 is None or v1 == v2:
                return False
            return solve(lits, i + 1)
        pred = lit.get("predicate", "")
        typ = TYPE_BY_PREDICATE.get(pred, "")
        if lit.get("negated"):
            if not _known(project, pred):
                return False
            if "const" in lit:
                if lit["const"] in _values(project, pred):
                    return False
                return solve(lits, i + 1)
            var = lit.get("var")
            if var is None:
                return False
            bound = env.get((typ, var))
            if bound is None:
                return False
            if bound in _values(project, pred):
                return False
            return solve(lits, i + 1)
        if not _known(project, pred):
            return False
        if "const" in lit:
            if lit["const"] not in _values(project, pred):
                return False
            return solve(lits, i + 1)
        var = lit.get("var")
        if not isinstance(var, int):
            return False
        current = env.get((typ, var))
        for value in _values(project, pred):
            if current is not None and current != value:
                continue
            old = env.get((typ, var))
            env[(typ, var)] = value
            if solve(lits, i + 1):
                return True
            if old is None:
                env.pop((typ, var), None)
            else:
                env[(typ, var)] = old
        return False

    return solve(_ordered(rule.get("body", [])), 0)


def coverage(rule: dict, projects: list[dict]) -> dict:
    """TP/FP/FN для правила на размеченных проектах + списки id."""
    tp: list[str] = []
    fp: list[str] = []
    fn: list[str] = []
    for project in projects:
        labels = project.get("labels")
        if labels is None:
            continue
        positive = rule.get("target") in labels
        covered = covers(rule, project)
        if covered and positive:
            tp.append(project["id"])
        elif covered and not positive:
            fp.append(project["id"])
        elif (not covered) and positive:
            fn.append(project["id"])
    return {"tp": tp, "fp": fp, "fn": fn,
            "n_tp": len(tp), "n_fp": len(fp), "n_fn": len(fn)}


def all_rule_coverage(rules: list[dict], projects: list[dict]) -> list[dict]:
    return [{"rule": rule, **coverage(rule, projects)} for rule in rules]


def missing_evidence(rule: dict, project: dict) -> list[str]:
    """Возвращает первый невыполненный литерал с учётом переменных и неизвестных полей."""
    env: dict[tuple[str, int], str] = {}
    out = []
    for lit in _ordered(rule.get("body", [])):
        if lit.get("neq"):
            typ = lit.get("type")
            left = env.get((typ, lit.get("left")))
            right = env.get((typ, lit.get("right")))
            if left is None or right is None:
                out.append("переменные для условия \\= не связаны")
                return out
            if left == right:
                out.append(f"значения переменных равны ({left}); условие \\= не выполнено")
            continue
        pred = lit.get("predicate", "")
        if not _known(project, pred):
            out.append(f"поле «{PREDICATE_TO_FIELD.get(pred, pred)}» неизвестно")
            return out
        values = _values(project, pred)
        typ = TYPE_BY_PREDICATE.get(pred, "")
        if lit.get("negated"):
            if "const" in lit:
                if lit["const"] in values:
                    out.append(f"присутствует запрещённый факт {pred}({project['id']}, {lit['const']})")
            else:
                bound = env.get((typ, lit.get("var")))
                if bound is None:
                    out.append(f"переменная для отрицания {pred} не связана")
                    return out
                if bound in values:
                    out.append(f"присутствует запрещённый факт {pred}({project['id']}, {bound})")
            continue
        if "const" in lit:
            value = lit["const"]
            if value not in values:
                out.append(f"нет факта {pred}({project['id']}, {value})")
                return out
            bound_value = value
        else:
            var = lit.get("var")
            previous = env.get((typ, var))
            matching = next((value for value in values if previous is None or value == previous), None)
            if matching is None:
                if previous is not None:
                    out.append(f"нет общего значения для {pred} и связанных условий (ожидалось {previous})")
                else:
                    out.append(f"нет фактов {pred}({project['id']}, _)")
                return out
            bound_value = matching
        if "var" in lit:
            env[(typ, lit["var"])] = bound_value
    return out


def counterfactuals(match: dict) -> list[str]:
    """Простые контрфакты: какое одно подтверждение убрать, чтобы это доказательство исчезло."""
    out = []
    for proof in match.get("proofs", []):
        for fact in proof.get("facts", []):
            args = ", ".join(fact.get("args", []))
            out.append(f"Если убрать факт {fact['predicate']}({args}), это доказательство правила перестанет работать.")
    return out


def literal_text(lit: dict) -> str:
    if lit.get("neq"):
        return f"переменная {lit.get('left')} \\= переменная {lit.get('right')}"
    pred = lit.get("predicate", "?")
    arg = lit.get("const", lit.get("var", "?"))
    prefix = "\\+ " if lit.get("negated") else ""
    return f"{prefix}{pred}(Project, {arg})"


def rule_summary(rule: dict) -> str:
    return ", ".join(literal_text(literal) for literal in rule.get("body", []))
