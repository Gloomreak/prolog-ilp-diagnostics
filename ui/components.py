"""Переиспользуемые элементы интерфейса: правила, таблицы, графики.

Здесь нет алгоритма обучения — только отображение результатов.
"""
from __future__ import annotations

import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.dataset import CLASS_TITLES_RU, ERROR_CLASSES  # noqa: E402

REASON_RU = {
    "low_support": "поддержка (TP) ниже min_support",
    "low_precision": "точность ниже min_precision",
    "no_new_positives": "порог пройден, но новых положительных примеров правило не покрывает",
    "dominated": "порог пройден, но правило уступает выбранному по приросту покрытия",
    "rule_limit_reached": "порог пройден, но достигнут max_rules_per_class",
}

WARNING_RU = {
    "max_candidates_reached": "Достигнут лимит max_candidates ({d}): поиск остановлен, часть "
                              "кандидатов не проверена. Увеличьте лимит или уменьшите max_body_length.",
    "no_rule_found": "Для класса «{d}» не найдено правило, прошедшее пороги.",
    "no_positive_examples": "В обучающей выборке нет положительных примеров класса «{d}».",
}

_COLORS = {"ILP (Prolog)": "#1f77b4", "Decision Tree": "#ff7f0e",
           "Decision Tree + агрегаты": "#7f7f7f"}


def warning_text(w: dict) -> str:
    tpl = WARNING_RU.get(w["code"], "{c}: {d}")
    return tpl.format(d=w["detail"], c=w["code"])


def rule_html(rule: dict) -> str:
    """Правило с подсветкой целевого предиката и условий."""
    head = f"has_error(Project, {rule['target']})"
    lits = []
    for lit in rule["body"]:
        if lit.get("neq"):
            left = (lit.get("args") or ["", "?"])[0]
            right = (lit.get("args") or ["", "?"])[1] if len(lit.get("args") or []) > 1 else "?"
            lits.append(f"<span style='color:#0b7a3b;font-weight:600'>{html.escape(left)}</span>"
                        f" \\= <span style='color:#0b7a3b;font-weight:600'>{html.escape(str(right))}</span>")
            continue
        prefix = "\\+ " if lit.get("negated") else ""
        args = ", ".join(str(a) for a in lit.get("args", ["Project", lit.get("var", "?")]))
        lits.append(f"{prefix}<span style='color:#0b7a3b;font-weight:600'>"
                    f"{html.escape(str(lit.get('predicate', '?')))}</span>"
                    f"({html.escape(args)})")
    body = ",<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(lits)
    return ("<div style='font-family:monospace;background:rgba(127,127,127,0.12);padding:0.6rem 0.8rem;"
            "border-radius:6px;line-height:1.5'>"
            f"<span style='color:#b3261e;font-weight:700'>{html.escape(head)}</span> :-<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{body}.</div>")


def show_rule(rule: dict, index: int | None = None) -> None:
    """Блок правила: подсветка, метрики, текст для копирования."""
    st.markdown(rule_html(rule), unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("support (TP)", rule["tp"])
    c[1].metric("FP", rule["fp"])
    c[2].metric("FN", rule["fn"])
    c[3].metric("precision", f"{rule['precision']:.3f}")
    if rule.get("precision_smoothed") is not None:
        c[3].caption(f"сглаженная: {rule['precision_smoothed']:.3f}")
    c[4].metric("recall", f"{rule['recall']:.3f}")
    c[5].metric("F1", f"{rule['f1']:.3f}")
    with st.expander("Скопировать текст правила"):
        st.code(rule["rule"], language="prolog")


def rules_table(rules: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Класс": r["target"], "Длина": r["length"], "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
        "precision": r["precision"], "recall": r["recall"], "F1": r["f1"],
        "Правило": " ".join(r["rule"].split()),
    } for r in rules])


def projects_table(projects: list[dict]) -> pd.DataFrame:
    rows = []
    for p in projects:
        f = p["facts"]
        j = lambda k: ", ".join(f[k]) if f[k] else ("?" if f[k] is None else "")   # noqa: E731
        rows.append({
            "id": p["id"],
            "метки": "—" if p["labels"] is None else (", ".join(p["labels"]) or "(нет ошибок)"),
            "требуются пакеты": j("required_packages"), "нет пакетов": j("absent_packages"),
            "несоотв. версий": j("version_mismatches"),
            "требуются ключи": j("required_config_keys"), "нет ключей": j("absent_config_keys"),
            "требуются перем.": j("required_env_variables"), "нет перем.": j("absent_env_variables"),
            "сборка": j("build_tools"), "среда": j("runtimes"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ графики
def _finish(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def class_distribution_chart(summary: dict) -> None:
    pos = [summary["class_counts"][c] for c in ERROR_CLASSES]
    neg = [summary["class_negatives"][c] for c in ERROR_CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = range(len(ERROR_CLASSES))
    ax.bar(xs, pos, label="положительные (E+)", color="#d62728")
    ax.bar(xs, neg, bottom=pos, label="отрицательные (E−)", color="#c7c7c7")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c.replace("_", "\n") for c in ERROR_CLASSES], fontsize=8)
    ax.set_ylabel("проектов")
    ax.legend(fontsize=8)
    _finish(fig)


def f1_by_class_chart(exp: dict) -> None:
    methods = list(exp["methods"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = [exp["methods"][m]["metrics"]["per_class"][c]["f1"] for c in ERROR_CLASSES]
        ax.bar([i + k * w for i in range(len(ERROR_CLASSES))], vals, w, label=m,
               color=_COLORS.get(m))
    ax.set_xticks([i + w * (len(methods) - 1) / 2 for i in range(len(ERROR_CLASSES))])
    ax.set_xticklabels([c.replace("_", "\n") for c in ERROR_CLASSES], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 на тестовой выборке")
    ax.legend(fontsize=8)
    _finish(fig)


def overall_chart(exp: dict) -> None:
    methods = list(exp["methods"])
    keys = [("macro_f1", "macro-F1"), ("micro_f1", "micro-F1"), ("exact_match", "exact-match")]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = [exp["methods"][m]["metrics"][key] for key, _ in keys]
        ax.bar([i + k * w for i in range(len(keys))], vals, w, label=m, color=_COLORS.get(m))
    ax.set_xticks([i + w * (len(methods) - 1) / 2 for i in range(len(keys))])
    ax.set_xticklabels([n for _, n in keys])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _finish(fig)


def time_chart(exp: dict) -> None:
    methods = list(exp["methods"])
    fig, ax = plt.subplots(figsize=(6, 3.0))
    vals = [exp["methods"][m]["train_seconds"] for m in methods]
    ax.barh(methods, vals, color=[_COLORS.get(m) for m in methods])
    ax.set_xlabel("время обучения, с (ILP — вместе с запуском процесса SWI-Prolog)")
    _finish(fig)


def multiseed_chart(agg: dict) -> None:
    methods = list(agg)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = 0.8 / len(methods)
    keys = [("macro_f1", "macro-F1"), ("micro_f1", "micro-F1"), ("exact_match", "exact-match")]
    for k, m in enumerate(methods):
        ax.bar([i + k * w for i in range(len(keys))], [agg[m][key]["mean"] for key, _ in keys], w,
               yerr=[agg[m][key]["std"] for key, _ in keys], capsize=3, label=m,
               color=_COLORS.get(m))
    ax.set_xticks([i + w * (len(methods) - 1) / 2 for i in range(len(keys))])
    ax.set_xticklabels([n for _, n in keys])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _finish(fig)


def class_title(c: str) -> str:
    return f"{CLASS_TITLES_RU.get(c, c)} (`{c}`)"
