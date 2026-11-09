"""Экспорт: модель (JSON + .pl), диагностика (JSON/CSV), эксперимент, датасет."""
from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

from .dataset import ERROR_CLASSES, PREDICATE_TO_FIELD
from .diagnosis import format_fact

MODEL_FORMAT = "prolog-ilp-diagnostics/model"
MODEL_FORMAT_VERSION = "2.0"
SUPPORTED_MODEL_VERSIONS = {"1.0", "2.0"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_model_document(train_response: dict, params: dict, dataset_info: dict,
                         environment: dict | None = None) -> dict:
    """Единый документ модели: правила, метаданные обучения, параметры, журнал."""
    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "created_utc": _now(),
        "params_requested": params,
        "dataset": dataset_info,
        "environment": environment or {},
        "model": train_response["model"],
        "stats": train_response["stats"],
        "log": train_response["log"],
        "warnings": train_response["warnings"],
    }


def load_model_document(data: str | bytes | dict) -> dict:
    """Загружает и проверяет ранее экспортированную модель (JSON).

    Поддерживает формат 1.0 (только положительные литералы-переменные)
    и 2.0 (константы, отрицания, \\=).
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig")
    if isinstance(data, str):
        try:
            doc = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Модель не является корректным JSON: {exc}") from None
    elif isinstance(data, dict):
        doc = data
    else:
        raise ValueError("Ожидался JSON-объект модели.")

    if not isinstance(doc, dict) or doc.get("format") != MODEL_FORMAT:
        raise ValueError("Это не файл модели prolog-ilp-diagnostics/model.")
    version = str(doc.get("format_version", ""))
    if version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(f"Неподдерживаемая версия модели {version!r}; "
                         f"поддерживаются: {', '.join(sorted(SUPPORTED_MODEL_VERSIONS))}.")
    model = doc.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("rules"), list):
        raise ValueError("В модели отсутствует список model.rules.")
    for i, rule in enumerate(model["rules"], 1):
        if not isinstance(rule, dict):
            raise ValueError(f"Правило №{i}: ожидался объект.")
        if rule.get("target") not in ERROR_CLASSES:
            raise ValueError(f"Правило №{i}: неизвестный класс {rule.get('target')!r}.")
        if not isinstance(rule.get("rule"), str) or not isinstance(rule.get("body"), list):
            raise ValueError(f"Правило №{i}: нужны поля rule (строка) и body (массив).")
        for j, lit in enumerate(rule["body"], 1):
            _validate_literal(lit, i, j)
    doc.setdefault("format_version", version)
    doc.setdefault("model", model)
    doc.setdefault("stats", {})
    doc.setdefault("log", {"levels": [], "rejected": []})
    doc.setdefault("warnings", [])
    doc.setdefault("dataset", {})
    doc.setdefault("params_requested", {})
    return doc


def _validate_literal(lit, rule_no: int, lit_no: int) -> None:
    if not isinstance(lit, dict):
        raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: ожидался объект.")
    if lit.get("neq"):
        types = {"package", "config_key", "env_variable", "build_tool", "runtime"}
        left, right = lit.get("left"), lit.get("right")
        if (lit.get("type") not in types or not isinstance(left, int) or isinstance(left, bool)
                or not isinstance(right, int) or isinstance(right, bool)
                or left < 1 or right <= left):
            raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: неверное \\=.")
        return
    predicate = lit.get("predicate")
    if predicate not in PREDICATE_TO_FIELD:
        raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: неизвестный предикат {predicate!r}.")
    if "const" in lit:
        if not isinstance(lit["const"], (str, int, float)) or isinstance(lit["const"], bool):
            raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: неверная константа.")
    elif not isinstance(lit.get("var"), int) or isinstance(lit.get("var"), bool) or lit["var"] < 1:
        raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: нужен var или const.")
    if "negated" in lit and not isinstance(lit["negated"], bool):
        raise ValueError(f"Правило №{rule_no}, литерал №{lit_no}: negated должен быть логическим.")


def model_to_json(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2)


def rules_to_prolog(doc: dict) -> str:
    """Читаемый .pl-файл с выученными правилами.

    Файл можно загрузить в SWI-Prolog: BK-предикаты объявлены dynamic,
    поэтому вызов правила без фактов просто завершится неудачей, а не
    синтаксической ошибкой.
    """
    st = doc.get("stats", {})
    lines = [
        "% Правила, выученные ILP-движком (SWI-Prolog).",
        f"% Создано (UTC): {doc.get('created_utc', '')}",
        f"% Параметры: max_body_length={st.get('max_body_length')}, "
        f"min_support={st.get('min_support')}, min_precision={st.get('min_precision')}, "
        f"max_rules_per_class={st.get('max_rules_per_class')}",
        f"% Обучающих проектов: {st.get('train_size')}",
        "% Правила применяются к фактам BK (requires_package/2, package_absent/2, ...).",
        "% Пример: ?- requires_package(p1, pandas), package_absent(p1, pandas), has_error(p1, E).",
        "",
        ":- discontiguous has_error/2.",
        ":- dynamic requires_package/2, package_absent/2, version_mismatch/2,",
        "           requires_config_key/2, config_key_absent/2,",
        "           requires_env_variable/2, env_variable_absent/2,",
        "           uses_build_tool/2, uses_runtime/2.",
        "",
    ]
    by_class = {c: [] for c in ERROR_CLASSES}
    for r in doc["model"]["rules"]:
        by_class.setdefault(r["target"], []).append(r)
    for cls, rules in by_class.items():
        if not rules:
            lines.append(f"% {cls}: надёжное правило не найдено")
            lines.append("")
            continue
        for r in rules:
            lines.append(f"% {cls}: TP={r['tp']} FP={r['fp']} FN={r['fn']} "
                         f"precision={r['precision']} recall={r['recall']} f1={r['f1']}")
            lines.append(r["rule"])
            lines.append("")
    return "\n".join(lines)


def diagnosis_to_json(parsed: list[dict], extra: dict | None = None) -> str:
    return json.dumps({"format": "prolog-ilp-diagnostics/diagnosis", "created_utc": _now(),
                       **(extra or {}), "results": parsed}, ensure_ascii=False, indent=2)


def diagnosis_to_csv(parsed: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["project_id", "error_class", "status", "rule", "substitution",
                "facts", "suggestion"])
    for p in parsed:
        for cls in ERROR_CLASSES:
            cr = p["classes"].get(cls)
            if not cr:
                continue
            if not cr["matches"]:
                w.writerow([p["id"], cls, cr["status"], "", "", "", ""])
            for m in cr["matches"]:
                rule = m["rule"].replace("\n", " ")
                suggestion = m.get("suggestion", "")
                if not m["proofs"]:
                    w.writerow([p["id"], cls, cr["status"], rule, "", "", suggestion])
                for pr in m["proofs"]:
                    w.writerow([p["id"], cls, cr["status"], rule,
                                "; ".join(f"{k}={v}" for k, v in sorted(pr["bindings"].items())),
                                "; ".join(format_fact(f) for f in pr["facts"]),
                                suggestion])
    return buf.getvalue()


def experiment_to_json(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2)


def comparison_rows(exp: dict) -> list[dict]:
    """Плоская таблица сравнения методов (для CSV и интерфейса)."""
    rows = []
    for name, m in exp["methods"].items():
        met = m["metrics"]
        row = {"Метод": name, "macro-F1": met["macro_f1"], "micro-F1": met["micro_f1"],
               "exact-match": met["exact_match"], "время обучения, с": m["train_seconds"]}
        for c in ERROR_CLASSES:
            row[f"F1 {c}"] = met["per_class"][c]["f1"]
        rows.append(row)
    return rows


def comparison_to_csv(exp: dict) -> str:
    rows = comparison_rows(exp)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
