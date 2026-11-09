"""Разбор ответа Prolog при диагностике и вспомогательные функции для интерфейса."""
from __future__ import annotations

from .dataset import ERROR_CLASSES, PREDICATE_TO_FIELD, unknown_fields

STATUS_RU = {
    "detected": "обнаружен",
    "not_detected": "не обнаружен обученными правилами",
    "insufficient_data": "недостаточно данных",
    "no_rules": "недостаточно данных / по этому классу правил нет",
}
NOT_FOUND_MESSAGE = "Обученные правила не обнаружили известных признаков ошибок"
INSUFFICIENT_MESSAGE = ("Недостаточно данных: неизвестны поля, от которых зависит правило. "
                        "«Неизвестно» не считается «отсутствует».")
NO_RULE_MESSAGE = "Недостаточно данных для построения надёжного правила"

_SUGGESTIONS = {
    "missing_dependency": "Установите пакет {value} (например, `pip install {value}`) "
                          "или добавьте его в зависимости проекта.",
    "incompatible_version": "Приведите версию пакета {value} к требуемому диапазону: "
                            "обновите зависимость или зафиксируйте совместимую версию.",
    "missing_config_key": "Задайте ключ конфигурации {value} в конфигурационном файле проекта.",
    "missing_environment_variable": "Определите переменную окружения {value} "
                                    "в окружении или в файле `.env`.",
}

_BINDING_NAME = {
    "missing_dependency": "Package",
    "incompatible_version": "Package",
    "missing_config_key": "Key",
    "missing_environment_variable": "Variable",
}


def model_rules(model_doc: dict) -> list[dict]:
    return model_doc["model"]["rules"]


def body_fields(rule: dict) -> set[str]:
    """Поля facts, от которых зависит тело правила (включая neg/const)."""
    fields = set()
    for lit in rule.get("body", []):
        pred = lit.get("predicate")
        fld = PREDICATE_TO_FIELD.get(pred)
        if fld:
            fields.add(fld)
    return fields


def parse_diagnosis(response: dict, rules: list[dict]) -> list[dict]:
    """Ответ Prolog -> список словарей по проектам, с текстами правил (rule_index 1-based)."""
    out = []
    for pr in response["results"]:
        classes = {}
        for cls, cr in pr["classes"].items():
            matches = []
            for m in cr["matches"]:
                rule = rules[m["rule_index"] - 1]
                match = {"rule_index": m["rule_index"], "rule": rule["rule"],
                         "precision": rule.get("precision"),
                         "precision_smoothed": rule.get("precision_smoothed"),
                         "proofs": m["proofs"]}
                match["suggestion"] = suggestion_text(cls, match)
                matches.append(match)
            classes[cls] = {"status": cr["status"], "matches": matches,
                            "missing_fields": []}
        out.append({"id": pr["id"], "classes": classes})
    return out


def mark_insufficient_data(project: dict, rules: list[dict], parsed: dict) -> dict:
    """Третий статус из п. 7.2 ТЗ: правило не сработало, но нужные поля неизвестны."""
    unknown = set(unknown_fields(project))
    if not unknown:
        return parsed
    for cls in ERROR_CLASSES:
        cr = parsed["classes"].get(cls)
        if cr is None or cr["status"] != "not_detected":
            continue
        need = set()
        for rule in rules:
            if rule.get("target") == cls:
                need |= body_fields(rule)
        missing = sorted(unknown & need)
        if missing:
            cr["status"] = "insufficient_data"
            cr["missing_fields"] = missing
    return parsed


def predicted_labels(parsed: list[dict], classes: list[str] | None = None) -> list[list[str]]:
    classes = classes or ERROR_CLASSES
    return [[c for c in classes if p["classes"].get(c, {}).get("status") == "detected"]
            for p in parsed]


def format_fact(fact: dict) -> str:
    pred = fact["predicate"]
    args = ", ".join(fact["args"])
    return f"{pred}({args})"


def explanation_text(match: dict) -> list[str]:
    """Человекочитаемое объяснение сработавшего правила."""
    lines = []
    for proof in match["proofs"]:
        binds = ", ".join(f"{k} = {v}" for k, v in sorted(proof["bindings"].items()))
        facts = "; ".join(format_fact(f) for f in proof["facts"])
        lines.append(f"Подстановка: {binds or '—'}. Подтверждающие факты BK: {facts}")
    return lines


def suggestion_text(class_name: str, match: dict) -> str:
    """Подсказка исправления по сработавшему правилу (без автоматической правки)."""
    binding = _BINDING_NAME.get(class_name)
    values: list[str] = []
    for proof in match.get("proofs", []):
        value = (proof.get("bindings") or {}).get(binding or "")
        if value and value not in values:
            values.append(value)
    template = _SUGGESTIONS.get(class_name)
    if not template:
        return ""
    if not values:
        return template.format(value="…")
    if len(values) == 1:
        return template.format(value=values[0])
    return "; ".join(template.format(value=v) for v in values)


def unknown_field_warnings(project: dict, rules: list[dict]) -> list[str]:
    """Предупреждения: правила используют предикаты, значения которых неизвестны."""
    unknown = set(unknown_fields(project))
    if not unknown:
        return []
    msgs = []
    for cls in ERROR_CLASSES:
        need = set()
        for r in rules:
            if r["target"] == cls:
                need |= body_fields(r)
        missing = sorted(unknown & need)
        if missing:
            msgs.append(f"Класс «{cls}»: для проверки нужны поля, значение которых неизвестно "
                        f"({', '.join(missing)}); «неизвестно» не считается «отсутствует», "
                        "поэтому результат «не обнаружен» может быть неполным.")
    return msgs
