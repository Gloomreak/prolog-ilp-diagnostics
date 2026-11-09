"""Формат набора данных: загрузка, проверка, нормализация, сводка.

Политики (зафиксированы в README):
  * labels: [] означает «проект проверен, ошибок нет» (отрицательный пример
    для всех классов). Отсутствующее/null поле labels в обучающем наборе — ошибка
    записи: неизвестная разметка не должна молча превращаться в «нет ошибок».
  * facts.<поле>: null или отсутствие поля = «неизвестно»; [] = «известно, что
    пусто». Для Prolog оба случая дают отсутствие фактов, но интерфейс их различает
    (предупреждения при диагностике).
  * Неизвестные поля игнорируются с предупреждением.
  * Некорректная запись исключается из набора и попадает в отчёт; если корректных
    записей нет, набор отклоняется целиком.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMAS = {"1.0"}

ERROR_CLASSES = [
    "missing_dependency",
    "incompatible_version",
    "missing_config_key",
    "missing_environment_variable",
]

CLASS_TITLES_RU = {
    "missing_dependency": "Отсутствующая зависимость",
    "incompatible_version": "Несовместимая версия",
    "missing_config_key": "Отсутствующий ключ конфигурации",
    "missing_environment_variable": "Отсутствующая переменная окружения",
}

# поле JSON -> предикат BK (должно совпадать с data_io:fact_field/2)
FACT_FIELDS = {
    "required_packages": "requires_package",
    "absent_packages": "package_absent",
    "version_mismatches": "version_mismatch",
    "required_config_keys": "requires_config_key",
    "absent_config_keys": "config_key_absent",
    "required_env_variables": "requires_env_variable",
    "absent_env_variables": "env_variable_absent",
    "build_tools": "uses_build_tool",
    "runtimes": "uses_runtime",
}
PREDICATE_TO_FIELD = {v: k for k, v in FACT_FIELDS.items()}

_PROJECT_KEYS = {"id", "labels", "facts"}


@dataclass
class Issue:
    level: str                 # "error" | "warning"
    message: str
    record: str | None = None  # id записи или «запись №N»
    field: str | None = None

    def __str__(self) -> str:
        where = []
        if self.record:
            where.append(f"запись {self.record}")
        if self.field:
            where.append(f"поле «{self.field}»")
        prefix = (", ".join(where) + ": ") if where else ""
        return prefix + self.message

    def as_dict(self) -> dict:
        return {"Уровень": "ошибка" if self.level == "error" else "предупреждение",
                "Запись": self.record or "", "Поле": self.field or "", "Сообщение": self.message}


@dataclass
class ValidationResult:
    projects: list[dict] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    fatal: bool = False
    n_records: int = 0          # сколько записей было во входном файле
    n_rejected: int = 0         # сколько отброшено как ошибочные
    meta: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.fatal and len(self.projects) > 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


def _norm_item(item: Any) -> tuple[str | None, str | None]:
    """Элемент списка признаков -> (значение, ошибка)."""
    if isinstance(item, str):
        s = item.strip()
        return (s, None) if s else (None, "пустая строка в списке")
    if isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip(), None
        return None, "объект должен содержать непустое строковое поле «name»"
    return None, f"ожидалась строка или объект {{\"name\": ...}}, получено {type(item).__name__}"


def validate_project(obj: Any, index: int, require_labels: bool = True
                     ) -> tuple[dict | None, list[Issue]]:
    """Проверяет одну запись. Возвращает нормализованный проект или None."""
    issues: list[Issue] = []
    rec = f"№{index + 1}"
    if not isinstance(obj, dict):
        return None, [Issue("error", "запись должна быть JSON-объектом", rec)]

    pid = obj.get("id")
    if not isinstance(pid, str) or not pid.strip():
        return None, [Issue("error", "id должен быть непустой строкой", rec, "id")]
    pid = pid.strip()
    rec = pid

    for k in obj:
        if k not in _PROJECT_KEYS:
            issues.append(Issue("warning", "неизвестное поле проигнорировано", rec, k))

    # --- labels
    labels_raw = obj.get("labels")
    labels: list[str] | None
    if labels_raw is None:
        if require_labels:
            issues.append(Issue("error", "обязательное поле отсутствует или равно null "
                                         "(для проекта без ошибок укажите пустой список [])", rec, "labels"))
            return None, issues
        labels = None
    elif not isinstance(labels_raw, list):
        issues.append(Issue("error", "labels должен быть массивом", rec, "labels"))
        return None, issues
    else:
        labels = []
        for lab in labels_raw:
            if not isinstance(lab, str) or lab not in ERROR_CLASSES:
                issues.append(Issue("error", f"недопустимый класс ошибки {lab!r}; допустимы: "
                                             f"{', '.join(ERROR_CLASSES)}", rec, "labels"))
                return None, issues
            if lab not in labels:
                labels.append(lab)
        if len(labels) != len(labels_raw):
            issues.append(Issue("warning", "дубликаты меток удалены", rec, "labels"))

    # --- facts
    facts_raw = obj.get("facts")
    if not isinstance(facts_raw, dict):
        issues.append(Issue("error", "facts должен быть объектом", rec, "facts"))
        return None, issues
    for k in facts_raw:
        if k not in FACT_FIELDS:
            issues.append(Issue("warning", "неизвестное поле признаков проигнорировано", rec, f"facts.{k}"))
    facts: dict[str, list[str] | None] = {}
    for fld in FACT_FIELDS:
        raw = facts_raw.get(fld)
        if raw is None:
            facts[fld] = None            # «неизвестно»
            continue
        if not isinstance(raw, list):
            issues.append(Issue("error", "ожидался массив (или null — «неизвестно»)", rec, f"facts.{fld}"))
            return None, issues
        values: list[str] = []
        for item in raw:
            val, problem = _norm_item(item)
            if problem:
                issues.append(Issue("error", problem, rec, f"facts.{fld}"))
                return None, issues
            if val not in values:
                values.append(val)
        if len(values) != len(raw):
            issues.append(Issue("warning", "дубликаты удалены", rec, f"facts.{fld}"))
        facts[fld] = values
    return {"id": pid, "labels": labels, "facts": facts}, issues


def validate_dataset(obj: Any, require_labels: bool = True) -> ValidationResult:
    res = ValidationResult()
    if not isinstance(obj, dict):
        res.fatal = True
        res.issues.append(Issue("error", "корень файла должен быть JSON-объектом с полями "
                                         "schema_version и projects"))
        return res
    ver = obj.get("schema_version")
    if ver is None:
        res.fatal = True
        res.issues.append(Issue("error", "обязательное поле отсутствует", None, "schema_version"))
        return res
    if ver not in SUPPORTED_SCHEMAS:
        res.fatal = True
        res.issues.append(Issue("error", f"неподдерживаемая версия схемы {ver!r}; "
                                         f"поддерживается: {', '.join(sorted(SUPPORTED_SCHEMAS))}",
                                None, "schema_version"))
        return res
    for k in obj:
        if k not in ("schema_version", "projects", "meta"):
            res.issues.append(Issue("warning", "неизвестное поле проигнорировано", None, k))
    meta = obj.get("meta")
    if isinstance(meta, dict):
        res.meta = meta
    elif meta is not None:
        res.issues.append(Issue("warning", "meta должен быть объектом; поле проигнорировано",
                                None, "meta"))
    projects = obj.get("projects")
    if not isinstance(projects, list) or not projects:
        res.fatal = True
        res.issues.append(Issue("error", "должен быть непустым массивом", None, "projects"))
        return res

    res.n_records = len(projects)
    seen: set[str] = set()
    for i, raw in enumerate(projects):
        proj, issues = validate_project(raw, i, require_labels)
        res.issues.extend(issues)
        if proj is None:
            res.n_rejected += 1
            continue
        if proj["id"] in seen:
            res.issues.append(Issue("error", "повторяющийся id; запись отброшена", proj["id"], "id"))
            res.n_rejected += 1
            continue
        seen.add(proj["id"])
        res.projects.append(proj)
    if not res.projects:
        res.fatal = True
        res.issues.append(Issue("error", "в файле нет ни одной корректной записи"))
    return res


def load_dataset_bytes(data: bytes, require_labels: bool = True) -> ValidationResult:
    """Разбор загруженного файла (UTF-8 JSON) с понятными сообщениями об ошибках."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        r = ValidationResult(fatal=True)
        r.issues.append(Issue("error", f"файл не в кодировке UTF-8: {exc}"))
        return r
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        r = ValidationResult(fatal=True)
        r.issues.append(Issue("error", f"некорректный JSON (строка {exc.lineno}, "
                                       f"столбец {exc.colno}): {exc.msg}"))
        return r
    return validate_dataset(obj, require_labels)


def load_dataset_file(path, require_labels: bool = True) -> ValidationResult:
    with open(path, "rb") as fh:
        return load_dataset_bytes(fh.read(), require_labels)


def dataset_to_json(projects: list[dict], meta: dict | None = None) -> dict:
    """Нормализованные проекты -> документ формата schema_version 1.0."""
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if meta:
        doc["meta"] = meta
    doc["projects"] = [
        {"id": p["id"], **({"labels": p["labels"]} if p["labels"] is not None else {}),
         "facts": p["facts"]} for p in projects]
    return doc


def dumps(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2)


def unknown_fields(project: dict) -> list[str]:
    """Поля признаков, значение которых «неизвестно» (null/отсутствует)."""
    return [f for f, v in project["facts"].items() if v is None]


def summary(projects: list[dict], validation: ValidationResult | None = None) -> dict:
    """Сводка набора (FR-05)."""
    n = len(projects)
    freq = Counter()
    combos = Counter()
    unlabeled = 0
    for p in projects:
        if p["labels"] is None:
            unlabeled += 1
            continue
        for lab in p["labels"]:
            freq[lab] += 1
        combos[", ".join(p["labels"]) if p["labels"] else "(без ошибок)"] += 1
    labeled = n - unlabeled
    return {
        "n_projects": n,
        "n_unlabeled": unlabeled,
        "class_counts": {c: freq.get(c, 0) for c in ERROR_CLASSES},
        "class_negatives": {c: labeled - freq.get(c, 0) for c in ERROR_CLASSES},
        "n_without_errors": combos.get("(без ошибок)", 0),
        "n_multi_label": sum(1 for p in projects if p["labels"] and len(p["labels"]) > 1),
        "label_combinations": dict(combos.most_common()),
        "n_with_unknown_fields": sum(1 for p in projects if unknown_fields(p)),
        "n_rejected_records": validation.n_rejected if validation else 0,
        "n_issues": len(validation.issues) if validation else 0,
    }


def labeled_only(projects: list[dict]) -> list[dict]:
    return [p for p in projects if p["labels"] is not None]
