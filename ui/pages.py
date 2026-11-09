"""Страницы Streamlit. Здесь только интерфейс: обучение, диагностика и сравнение
выполняются модулями core/ и логическим ядром на Prolog."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core import exports
from core.dataset import (
    ERROR_CLASSES,
    FACT_FIELDS,
    dataset_to_json,
    dumps,
    labeled_only,
    load_dataset_bytes,
    load_dataset_file,
    summary,
    validate_project,
)
from core.diagnosis import (
    INSUFFICIENT_MESSAGE,
    NO_RULE_MESSAGE,
    NOT_FOUND_MESSAGE,
    STATUS_RU,
    explanation_text,
    mark_insufficient_data,
    parse_diagnosis,
    unknown_field_warnings,
)
from core.evaluation import (
    environment_info,
    multilabel_metrics,
    run_ablation,
    run_experiment,
    run_learning_curve,
    run_multi_seed,
    run_repeated_stratified_cv,
    split_projects,
)
from core.feature_encoder import label_matrix
from core.generator import generate_document
from core.project_loader import project_from_files
from core.prolog_runner import PrologEngine, PrologError, check_swipl
from core.rule_analysis import counterfactuals, missing_evidence
from core.rule_analysis import coverage as rule_coverage
from core.settings import DATA_DIR, DEMO_DATASET, RESULTS_DIR, swipl_path_setting

from . import components as ui

PAGES = ["Обзор", "Данные", "Обучение ILP", "Диагностика", "Пакетная диагностика",
         "Эксперимент", "Экспорт / О проекте"]

DEMO_VARIANTS = {
    "Лёгкий (500 проектов, seed 42)": DEMO_DATASET,
    "Трудный: ошибка зависит от контекста": DATA_DIR / "demo_hard.json",
    "Частично неизвестные поля": DATA_DIR / "demo_partial.json",
}

FIELD_LABELS = {
    "required_packages": "Требуемые пакеты",
    "absent_packages": "Отсутствующие пакеты",
    "version_mismatches": "Пакеты с несоответствием версии",
    "required_config_keys": "Требуемые ключи конфигурации",
    "absent_config_keys": "Отсутствующие ключи конфигурации",
    "required_env_variables": "Требуемые переменные окружения",
    "absent_env_variables": "Отсутствующие переменные окружения",
    "build_tools": "Системы сборки",
    "runtimes": "Среды выполнения",
}
EXAMPLE_FORM = {
    "required_packages": "numpy, pandas, scikit-learn", "absent_packages": "pandas, scipy",
    "version_mismatches": "", "required_config_keys": "DATABASE_URL, SECRET_KEY",
    "absent_config_keys": "", "required_env_variables": "API_TOKEN",
    "absent_env_variables": "", "build_tools": "pip", "runtimes": "python",
}


# ------------------------------------------------------------------ состояние
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("swipl_path", swipl_path_setting() or "")
    for key in ("dataset", "model", "diagnosis", "experiment", "split_info",
                "rule_comparison", "batch_results"):
        ss.setdefault(key, None)
    ss.setdefault("disabled_rules", set())


@st.cache_data(ttl=60, show_spinner=False)
def _check_swipl_cached(path: str):
    """Проверка swipl выполняется не чаще раза в минуту (иначе — при каждом клике)."""
    try:
        return check_swipl(path or None), None
    except PrologError as exc:
        return None, str(exc)


def engine_status():
    """(engine | None, сообщение об ошибке | None, EngineInfo | None)"""
    info, error = _check_swipl_cached(st.session_state.get("swipl_path") or "")
    if info is None:
        return None, error, None
    return PrologEngine(info.path), None, info


def _need_dataset():
    ds = st.session_state.dataset
    if not ds:
        st.info("Сначала выберите или сгенерируйте набор данных на странице «Данные».")
        return None
    return ds


def _need_engine():
    engine, error, _ = engine_status()
    if engine is None:
        st.error(error)
        return None
    return engine


def _set_dataset(projects, validation, source) -> None:
    st.session_state.dataset = {"projects": projects, "validation": validation, "source": source,
                               "meta": validation.meta}
    # новые данные -> прежние результаты неактуальны
    st.session_state.model = None
    st.session_state.diagnosis = None
    st.session_state.experiment = None
    st.session_state.split_info = None
    st.session_state.disabled_rules = set()
    st.session_state.rule_comparison = None
    st.session_state.batch_results = None


def _active_rules(doc: dict) -> list[dict]:
    disabled = set(st.session_state.get("disabled_rules") or set())
    return [r for i, r in enumerate(doc["model"]["rules"]) if i not in disabled]


def _set_rule_enabled(index: int, enabled: bool) -> None:
    disabled = set(st.session_state.get("disabled_rules") or set())
    if enabled:
        disabled.discard(index)
    else:
        disabled.add(index)
    st.session_state.disabled_rules = disabled


def _load_model_upload() -> None:
    up = st.file_uploader("Загрузить модель (JSON)", type=["json"], key="model_upload")
    if up is not None and st.button("Применить загруженную модель", key="apply_model"):
        try:
            doc = exports.load_model_document(up.getvalue())
        except ValueError as exc:
            st.error(f"Модель не загружена: {exc}")
        else:
            st.session_state.model = doc
            st.session_state.disabled_rules = set()
            st.session_state.diagnosis = None
            st.session_state.rule_comparison = None
            st.success("Модель загружена. Можно диагностировать без переобучения.")
            st.rerun()


# ------------------------------------------------------------------ Обзор
def page_overview() -> None:
    st.title("Диагностика ошибок конфигурации на основе ILP (Prolog)")
    st.markdown(
        "Прототип **индуктивного логического программирования**: SWI-Prolog *сам выводит* "
        "диагностические правила из положительных и отрицательных примеров и фоновых знаний, "
        "а затем применяет их к новым проектам и объясняет срабатывание. Заранее написанных "
        "правил в системе нет — есть только словарь допустимых предикатов и алгоритм поиска.")
    _, error, info = engine_status()
    c = st.columns(3)
    if info:
        c[0].metric("SWI-Prolog", "найден")
        c[0].caption(info.version_text)
        if not info.is_recommended:
            st.warning("Рекомендуется SWI-Prolog 9.x; на более старой версии возможны отличия.")
    else:
        c[0].metric("SWI-Prolog", "не найден")
        st.error(error)
    ds = st.session_state.dataset
    c[1].metric("Проектов загружено", len(ds["projects"]) if ds else 0)
    c[2].metric("Правил выучено", len(st.session_state.model["model"]["rules"])
                if st.session_state.model else 0)
    st.subheader("Как работать")
    st.markdown(
        "1. **Данные** — выберите демонстрационный набор, сгенерируйте новый или загрузите JSON.\n"
        "2. **Обучение ILP** — задайте параметры и запустите поиск правил в Prolog.\n"
        "3. **Диагностика** — проверьте новый проект: вы увидите сработавшие правила и факты-основания.\n"
        "4. **Эксперимент** — сравните ILP с решающим деревом на одной тестовой выборке.\n"
        "5. **Экспорт** — сохраните модель, правила (`.pl`), результаты и набор данных.")
    with st.expander("Как устроено обучение"):
        st.markdown(
            "Цель — `has_error(Project, ErrorType)`. Тело правила строится из словаря "
            "`requires_package/2`, `package_absent/2`, `version_mismatch/2`, "
            "`requires_config_key/2`, `config_key_absent/2`, `requires_env_variable/2`, "
            "`env_variable_absent/2`, `uses_build_tool/2`, `uses_runtime/2`. Prolog перебирает "
            "тела длиной 1…`max_body_length` (с отсечением по поддержке и удалением изоморфных "
            "дубликатов), проверяет покрытие логическим выводом, считает TP/FP/FN, "
            "precision/recall/F1 и жадно отбирает правила. Метки примеров в фоновые знания "
            "не попадают.")


# ------------------------------------------------------------------ Данные
def page_data() -> None:
    st.header("Данные")
    source = st.radio("Источник набора", ["Демонстрационный набор", "Сгенерировать", "Загрузить JSON"],
                      horizontal=True)
    if source == "Демонстрационный набор":
        variant = st.selectbox("Вариант демонстрационного набора", list(DEMO_VARIANTS))
        path = DEMO_VARIANTS[variant]
        st.caption("Лёгкий — базовые правила длины 2; трудный — ошибка зависит от контекста "
                   "(появляются константы и правила длины 3); частичный — часть полей неизвестна.")
        if st.button("Загрузить демонстрационный набор", key="load_demo_dataset"):
            if path.is_file():
                res = load_dataset_file(path)
            else:
                from core.dataset import validate_dataset
                mode = "hard" if "Трудный" in variant else "partial" if "Частично" in variant else "easy"
                res = validate_dataset(generate_document(500, 42, mode=mode))
            _apply_validation(res, f"демонстрационный набор «{variant}»")
    elif source == "Сгенерировать":
        c = st.columns(3)
        n = c[0].number_input("Число проектов", 50, 5000, 500, step=50)
        seed = c[1].number_input("Random seed", 0, 10**9, 42)
        rate = c[2].slider("Доля проектов с ошибкой каждого класса", 0.05, 0.6, 0.30, 0.05)
        if st.button("Сгенерировать", key="generate_dataset"):
            from core.dataset import validate_dataset
            res = validate_dataset(generate_document(int(n), int(seed), float(rate)))
            _apply_validation(res, f"синтетический набор (n={int(n)}, seed={int(seed)})")
    else:
        up = st.file_uploader("JSON-файл (UTF-8, schema_version 1.0)", type=["json"])
        if up is not None and st.button("Проверить и загрузить"):
            res = load_dataset_bytes(up.getvalue())
            _apply_validation(res, f"файл {up.name}")

    ds = st.session_state.dataset
    if not ds:
        return
    val, projects = ds["validation"], ds["projects"]
    st.success(f"Источник: {ds['source']}. Корректных проектов: {len(projects)}.")
    if val and val.issues:
        with st.expander(f"Сообщения проверки: ошибок {len(val.errors)}, предупреждений {len(val.warnings)}",
                         expanded=bool(val.errors)):
            st.dataframe(pd.DataFrame([i.as_dict() for i in val.issues]))
    s = summary(projects, val)
    c = st.columns(5)
    c[0].metric("Проектов", s["n_projects"])
    c[1].metric("Без ошибок", s["n_without_errors"])
    c[2].metric("С несколькими ошибками", s["n_multi_label"])
    c[3].metric("Ошибочных записей отброшено", s["n_rejected_records"])
    c[4].metric("С «неизвестными» полями", s["n_with_unknown_fields"])
    st.subheader("Распределение классов")
    ui.class_distribution_chart(s)
    st.dataframe(pd.DataFrame([{"Класс": c, "E+": s["class_counts"][c], "E−": s["class_negatives"][c]}
                               for c in ERROR_CLASSES]))
    with st.expander("Сочетания меток"):
        st.dataframe(pd.DataFrame(list(s["label_combinations"].items()), columns=["Метки", "Проектов"]))
    st.subheader("Проекты")
    st.dataframe(ui.projects_table(projects))
    st.download_button("Скачать набор (JSON)", dumps(dataset_to_json(projects)),
                       "dataset.json", "application/json")


def _apply_validation(res, source: str) -> None:
    if not res.ok:
        st.error("Набор не загружен:")
        for i in res.errors[:20]:
            st.error(str(i))
        return
    _set_dataset(res.projects, res, source)
    st.rerun()


# ------------------------------------------------------------------ Обучение
def page_train() -> None:
    st.header("Обучение ILP")
    st.caption("Модель можно обучить заново или загрузить ранее сохранённый JSON — "
               "диагностика работает без переобучения.")
    _load_model_upload()
    ds = st.session_state.dataset
    if not ds:
        st.info("Сначала выберите или сгенерируйте набор данных на странице «Данные».")
        if st.session_state.model:
            _show_model(st.session_state.model)
        return
    labeled = labeled_only(ds["projects"])
    if len(labeled) < 10:
        st.error("Для обучения нужно не менее 10 размеченных проектов.")
        return
    st.subheader("Параметры поиска правил")
    c = st.columns(5)
    max_len = c[0].selectbox("max_body_length", [1, 2, 3], index=2,
                             help="Максимальное число условий в теле правила.")
    min_supp = c[1].number_input("min_support", 1, 1000, 2,
                                 help="Минимум покрытых положительных обучающих примеров (TP).")
    min_prec = c[2].slider("min_precision", 0.0, 1.0, 0.80, 0.01,
                           help="Порог применяется к сглаженной точности (m-оценке).")
    max_rules = c[3].number_input("max_rules_per_class", 1, 20, 5)
    max_cand = c[4].number_input("max_candidates", 100, 500000, 20000, step=1000,
                                 help="Предохранитель: максимум проверяемых тел правил.")
    with st.expander("Дополнительно: устойчивость к шуму и MDL"):
        d = st.columns(3)
        smoothing = d[0].number_input("smoothing (m-оценка)", 0.0, 20.0, 1.0, 0.5,
                                      help="Сглаживание точности: (TP + m·prior) / (TP + FP + m). "
                                           "Больше m — осторожнее к редким правилам.")
        mdl_penalty = d[1].number_input("MDL-штраф за литерал", 0.0, 20.0, 1.0, 0.5,
                                        help="Штраф за сложность в MDL-оценке отбора правил.")
        max_const = d[2].number_input("max_constants", 0, 100, 20,
                                      help="Сколько значений-констант рассматривать на предикат.")
    c = st.columns(3)
    holdout = c[0].checkbox("Оставить отложенную выборку (рекомендуется)", True,
                            help="Обучение идёт только на train; test остаётся для диагностики "
                                 "и эксперимента.")
    seed = c[1].number_input("Random seed разбиения", 0, 10**9, 42)
    test_size = c[2].slider("Доля test", 0.1, 0.5, 0.2, 0.05, disabled=not holdout)

    if st.button("Запустить обучение", key="train_model"):
        engine = _need_engine()
        if engine:
            if holdout:
                train, test = split_projects(labeled, float(test_size), int(seed))
            else:
                train, test = labeled, []
            params = {"max_body_length": int(max_len), "min_support": int(min_supp),
                      "min_precision": float(min_prec), "max_rules_per_class": int(max_rules),
                      "max_candidates": int(max_cand), "smoothing": float(smoothing),
                      "mdl_penalty": float(mdl_penalty), "max_constants": int(max_const)}
            try:
                with st.spinner(f"SWI-Prolog ищет правила на {len(train)} проектах…"):
                    resp = engine.train(train, params)
            except PrologError as exc:
                st.error(str(exc))
            else:
                info = {"source": ds["source"], "n_total": len(labeled), "n_train": len(train),
                        "n_test": len(test), "seed": int(seed),
                        "test_size": float(test_size) if holdout else 0.0,
                        "holdout": holdout, "test_ids": [p["id"] for p in test]}
                st.session_state.model = exports.build_model_document(
                    resp, params, info, environment_info(engine))
                st.session_state.diagnosis = None
                st.session_state.split_info = info
                st.session_state.disabled_rules = set()
                st.session_state.rule_comparison = None
                st.session_state.experiment = None
                st.session_state.batch_results = None
    doc = st.session_state.model
    if doc:
        _show_model(doc)


def _show_model(doc: dict) -> None:
    st.divider()
    st.subheader("Выученные правила")
    di = doc.get("dataset") or {}
    if di.get("n_train"):
        st.caption(f"Обучено на {di['n_train']} проектах из {di.get('n_total', '?')}"
                   + (f"; отложено для проверки: {di['n_test']}." if di.get("holdout") else "."))
    for w in doc.get("warnings", []):
        st.warning(ui.warning_text(w))
    rules = doc["model"]["rules"]
    infos = doc["model"].get("classes", {})
    labeled = labeled_only(st.session_state.dataset["projects"]) if st.session_state.dataset else []
    for cls in ERROR_CLASSES:
        info = infos.get(cls)
        if info is None:
            continue
        st.markdown(f"#### {ui.class_title(cls)}")
        st.caption(f"E+: {info['n_pos']} , E−: {info['n_neg']} , правил: {info['n_rules']}  "
                   f"покрыто положительных: {info['covered_positives']}")
        cls_rules = [(i, r) for i, r in enumerate(rules) if r["target"] == cls]
        if not cls_rules:
            st.info(NO_RULE_MESSAGE)
        for index, rule in cls_rules:
            enabled = st.checkbox("Правило включено", value=index not in st.session_state.disabled_rules,
                                  key=f"rule_on_{index}",
                                  help="Отключите правило, чтобы увидеть результат диагностики без него.")
            _set_rule_enabled(index, enabled)
            if not enabled:
                st.caption("Правило отключено для диагностики.")
            ui.show_rule(rule, index)
            if enabled:
                _show_rule_errors(rule, labeled)
    st.divider()
    st.subheader("Журнал обучения")
    s = doc["stats"]
    c = st.columns(5)
    c[0].metric("Кандидатов сгенерировано", s["candidates_generated"])
    c[1].metric("Кандидатов проверено", s["candidates_tested"])
    c[2].metric("Гипотез оценено (тело × класс)", s["hypotheses_scored"])
    c[3].metric("Время обучения, с", f"{s['elapsed_seconds']:.2f}")
    c[4].metric("Обучающих примеров покрыто", f"{s['train_covered_share']:.0%}")
    lv = doc.get("log", {}).get("levels", [])
    if lv:
        st.markdown("**Поиск по уровням (длина тела правила)**")
        st.dataframe(pd.DataFrame([{"Длина тела": x["length"], "Сгенерировано": x["generated"],
                                    "Проверено": x["tested"], "Прошло отсечение по поддержке": x["extended"]}
                                   for x in lv]))
    rej = doc.get("log", {}).get("rejected", [])
    st.markdown("**Лучшие отклонённые кандидаты и причины**")
    if rej:
        st.dataframe(pd.DataFrame([{
            "Класс": r["target"], "TP": r["tp"], "FP": r["fp"], "precision": r["precision"],
            "recall": r["recall"], "F1": r["f1"],
            "Причина": "; ".join(ui.REASON_RU.get(x, x) for x in r["reasons"]),
            "Кандидат": " ".join(r["rule"].split())} for r in rej]))
    else:
        st.caption("Отклонённых кандидатов с ненулевым покрытием нет.")
    with st.expander("Все правила таблицей"):
        st.dataframe(ui.rules_table(rules) if rules else pd.DataFrame())
    st.caption(f"Включено правил: {len(_active_rules(doc))} из {len(rules)}.")
    c = st.columns(2)
    if c[0].button("Сохранить модель в results/"):
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"model_{stamp}.json"
        path.write_text(exports.model_to_json(doc), encoding="utf-8")
        st.success(f"Сохранено: {path}")
    c[1].download_button("Скачать модель (JSON)", exports.model_to_json(doc),
                         "model.json", "application/json")


def _show_rule_errors(rule: dict, projects: list[dict]) -> None:
    with st.expander("Разбор ошибок правила: FP/FN и чего не хватает"):
        if not projects:
            st.caption("Нет размеченных проектов для анализа.")
            return
        cov = rule_coverage(rule, projects)
        c = st.columns(4)
        c[0].metric("TP", cov["n_tp"])
        c[1].metric("FP", cov["n_fp"])
        c[2].metric("FN", cov["n_fn"])
        c[3].metric("Всего размечено", len(projects))
        by_id = {p["id"]: p for p in projects}
        if cov["fp"]:
            st.markdown("**FP — правило сработало, но метки класса нет**")
            st.dataframe(pd.DataFrame([{
                "id": pid,
                "метки": ", ".join(by_id[pid].get("labels") or []) or "(нет ошибок)",
                "факты": json.dumps(by_id[pid]["facts"], ensure_ascii=False)[:300],
            } for pid in cov["fp"]]))
        if cov["fn"]:
            st.markdown("**FN — метка класса есть, но правило не сработало**")
            st.dataframe(pd.DataFrame([{
                "id": pid,
                "чего не хватает": "; ".join(missing_evidence(rule, by_id[pid])) or "—",
            } for pid in cov["fn"]]))
        if not cov["fp"] and not cov["fn"]:
            st.success("На этом наборе правило не даёт FP/FN.")


# ------------------------------------------------------------------ Диагностика
def _parse_list(text: str) -> list[str]:
    parts = text.replace(",", "\n").splitlines()
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return out


def page_diagnosis() -> None:
    st.header("Диагностика нового проекта")
    doc = st.session_state.model
    if not doc:
        st.info("Сначала обучите модель на странице «Обучение ILP» или загрузите её JSON.")
        return
    active = _active_rules(doc)
    if not active:
        st.error("Все правила отключены. Включите хотя бы одно правило на странице обучения.")
        return
    st.caption(f"Активных правил: {len(active)} из {len(doc['model']['rules'])}.")
    mode = st.radio("Откуда взять проект",
                    ["Ввести вручную", "Загрузить JSON одного проекта",
                     "Из отложенной выборки", "Из файлов проекта"], horizontal=True)
    project = None
    if mode == "Ввести вручную":
        project = _manual_project_form()
    elif mode == "Загрузить JSON одного проекта":
        up = st.file_uploader("JSON с одним проектом (объект {id, facts[, labels]}) или набором",
                              type=["json"], key="single_project_json")
        if up is not None:
            try:
                obj = json.loads(up.getvalue().decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Некорректный JSON: {exc}")
                obj = None
            if isinstance(obj, dict) and "projects" in obj and obj["projects"]:
                obj = obj["projects"][0]
                st.caption("В файле набор проектов — берётся первый.")
            if obj is not None:
                project, issues = validate_project(obj, 0, require_labels=False)
                for i in issues:
                    (st.error if i.level == "error" else st.warning)(str(i))
    elif mode == "Из отложенной выборки":
        ds = st.session_state.dataset
        ids = doc.get("dataset", {}).get("test_ids", [])
        if not ids or not ds:
            st.info("Отложенной выборки нет: обучение шло на всём наборе.")
        else:
            pid = st.selectbox("Проект, не участвовавший в обучении", ids)
            project = next((p for p in ds["projects"] if p["id"] == pid), None)
    else:
        project = _project_from_files()

    if project is None:
        return
    if st.button("Запустить диагностику", key="run_diagnosis"):
        engine = _need_engine()
        if engine:
            try:
                with st.spinner("Prolog применяет выученные правила…"):
                    resp = engine.diagnose(active, [project], max_proofs=3)
                    parsed = parse_diagnosis(resp, active)[0]
                    parsed = mark_insufficient_data(project, active, parsed)
                    comparison = None
                    if st.session_state.disabled_rules:
                        full_rules = doc["model"]["rules"]
                        full = parse_diagnosis(engine.diagnose(full_rules, [project],
                                                               max_proofs=0), full_rules)[0]
                        full = mark_insufficient_data(project, full_rules, full)
                        comparison = _compare_statuses(parsed, full)
            except PrologError as exc:
                st.error(str(exc))
            else:
                st.session_state.diagnosis = {
                    "project": project, "parsed": parsed, "rules": active,
                    "warnings": unknown_field_warnings(project, active),
                    "fingerprint": _project_fingerprint(project),
                    "comparison": comparison,
                }
                st.session_state.rule_comparison = comparison
    d = st.session_state.diagnosis
    if d and d.get("fingerprint") == _project_fingerprint(project):
        _show_diagnosis(d)


def _project_fingerprint(project: dict) -> str:
    return project.get("id", "") + "|" + json.dumps(project.get("facts", {}), sort_keys=True,
                                                    ensure_ascii=False)


def _project_from_files() -> dict | None:
    st.caption("Загрузите файлы реального проекта: requirements.txt, pyproject.toml, "
               "poetry.lock/Pipfile.lock, .env, .env.example, config.example.json/toml. "
               "Значения .env не читаются — важны только имена переменных.")
    uploaded = st.file_uploader("Файлы проекта", accept_multiple_files=True,
                                key="project_files")
    if not uploaded:
        return None
    files: dict[str, str] = {}
    for f in uploaded:
        try:
            files[f.name] = f.getvalue().decode("utf-8-sig", "replace")
        except (UnicodeDecodeError, AttributeError):
            st.warning(f"Файл {f.name} пропущен: не удалось прочитать как UTF-8.")
    pid = st.text_input("Идентификатор проекта", "uploaded_project")
    parsed = project_from_files(files, pid)
    st.caption("Использованы: " + (", ".join(parsed.files_used) or "—"))
    for note in parsed.notes:
        st.info(note)
    project, issues = validate_project(parsed.project, 0, require_labels=False)
    for i in issues:
        (st.error if i.level == "error" else st.warning)(str(i))
    if project is not None:
        rows = [{"Поле": k, "Значение": "неизвестно" if v is None else ", ".join(v)}
                for k, v in project["facts"].items()]
        with st.expander("Разобранные факты"):
            st.dataframe(pd.DataFrame(rows))
    return project


def _compare_statuses(active_parsed: dict, full_parsed: dict) -> list[dict]:
    rows = []
    for cls in ERROR_CLASSES:
        a = active_parsed["classes"].get(cls, {}).get("status", "?")
        f = full_parsed["classes"].get(cls, {}).get("status", "?")
        if a != f:
            rows.append({"Класс": cls, "С отключёнными правилами": STATUS_RU.get(a, a),
                         "Со всеми правилами": STATUS_RU.get(f, f)})
    return rows


def _manual_project_form() -> dict | None:
    st.caption("Значения через запятую или с новой строки. Отметка «неизвестно» означает, что "
               "значение не известно (это не то же самое, что «пусто»).")
    pid = st.text_input("Идентификатор проекта", "new_project")
    values, unknown = {}, {}
    cols = st.columns(2)
    for k, fld in enumerate(FACT_FIELDS):
        with cols[k % 2]:
            unknown[fld] = st.checkbox(f"{FIELD_LABELS[fld]}: неизвестно", False, key=f"unk_{fld}")
            values[fld] = st.text_area(FIELD_LABELS[fld], EXAMPLE_FORM[fld], key=f"val_{fld}", height=68,
                                       disabled=unknown[fld])
    facts = {fld: (None if unknown[fld] else _parse_list(values[fld])) for fld in FACT_FIELDS}
    project, issues = validate_project({"id": pid, "facts": facts}, 0, require_labels=False)
    for i in issues:
        (st.error if i.level == "error" else st.warning)(str(i))
    return project


def _show_diagnosis(d: dict) -> None:
    st.divider()
    parsed = d["parsed"]
    st.subheader(f"Результат для проекта «{parsed['id']}»")
    for w in d.get("warnings", []):
        st.warning(w)
    detected = 0
    insufficient = 0
    for cls in ERROR_CLASSES:
        cr = parsed["classes"].get(cls)
        if cr is None:
            continue
        status = cr["status"]
        label = f"{ui.class_title(cls)} — {STATUS_RU.get(status, status)}"
        if status == "detected":
            detected += 1
            st.error(label)
            for m in cr["matches"]:
                with st.expander(f"Правило №{m['rule_index']} сработало", expanded=True):
                    st.code(m["rule"], language="prolog")
                    if m.get("suggestion"):
                        st.info("Что сделать: " + m["suggestion"])
                    for line in explanation_text(m):
                        st.markdown("- " + line)
                    for line in counterfactuals(m):
                        st.caption(line)
                    if m.get("precision_smoothed") is not None:
                        st.caption(f"Сглаженная точность правила: {m['precision_smoothed']:.3f}")
        elif status == "insufficient_data":
            insufficient += 1
            st.warning(label)
            fields = ", ".join(cr.get("missing_fields") or [])
            st.caption(INSUFFICIENT_MESSAGE + (f" Неизвестны: {fields}." if fields else ""))
        elif status == "no_rules":
            st.warning(label)
        else:
            st.info(label)
    if detected == 0 and insufficient == 0:
        st.info(NOT_FOUND_MESSAGE + ". Это не гарантирует, что конфигурация корректна.")
    if d.get("comparison"):
        st.subheader("Влияние отключённых правил")
        st.dataframe(pd.DataFrame(d["comparison"]))
    true_labels = d["project"].get("labels")
    if true_labels is not None:
        pred = [c for c in ERROR_CLASSES if parsed["classes"].get(c, {}).get("status") == "detected"]
        st.caption(f"Разметка в наборе: {', '.join(true_labels) or '(нет ошибок)'} , "
                   f"предсказано: {', '.join(pred) or '(ничего)'}")
    c = st.columns(2)
    c[0].download_button("Скачать диагностику (JSON)",
                         exports.diagnosis_to_json([parsed], {"warnings": d.get("warnings", [])}),
                         "diagnosis.json", "application/json")
    c[1].download_button("Скачать диагностику (CSV)", exports.diagnosis_to_csv([parsed]),
                         "diagnosis.csv", "text/csv")


# ------------------------------------------------------------------ Пакетная диагностика
def page_batch() -> None:
    st.header("Пакетная диагностика")
    doc = st.session_state.model
    if not doc:
        st.info("Сначала обучите модель или загрузите её JSON.")
        return
    active = _active_rules(doc)
    if not active:
        st.error("Все правила отключены.")
        return
    st.caption(f"Активных правил: {len(active)} из {len(doc['model']['rules'])}. "
               "Диагностика выполняется одним запросом в Prolog (без доказательств).")
    source = st.radio("Набор проектов", ["Текущий набор", "Загрузить JSON"],
                      horizontal=True, key="batch_source")
    projects: list[dict] = []
    if source == "Текущий набор":
        if not st.session_state.dataset:
            st.info("Сначала загрузите набор на странице «Данные».")
            return
        projects = st.session_state.dataset["projects"]
    else:
        up = st.file_uploader("JSON-набор (labels можно не указывать)", type=["json"],
                              key="batch_json")
        if up is not None:
            res = load_dataset_bytes(up.getvalue(), require_labels=False)
            if not res.ok:
                for i in res.errors[:20]:
                    st.error(str(i))
                return
            projects = res.projects
    if not projects:
        return
    st.caption(f"Проектов: {len(projects)}; размечено: {len(labeled_only(projects))}.")
    if st.button("Диагностировать пакет", key="run_batch_diagnosis"):
        engine = _need_engine()
        if engine:
            try:
                with st.spinner(f"Prolog проверяет {len(projects)} проектов…"):
                    resp = engine.diagnose(active, projects, max_proofs=0)
                    parsed = parse_diagnosis(resp, active)
                    by_id = {p["id"]: p for p in projects}
                    for item in parsed:
                        mark_insufficient_data(by_id[item["id"]], active, item)
            except PrologError as exc:
                st.error(str(exc))
            else:
                st.session_state.batch_results = {"projects": projects, "parsed": parsed,
                                                  "rules": active}
    res = st.session_state.batch_results
    if res:
        _show_batch(res)
        ids = [p["id"] for p in res["projects"]]
        selected_id = st.selectbox("Проект для разбора правила", ids, key="batch_detail_id")
        selected = next(p for p in res["projects"] if p["id"] == selected_id)
        st.subheader(f"Разбор проекта «{selected_id}»")
        item = next(p for p in res["parsed"] if p["id"] == selected_id)
        for cls in ERROR_CLASSES:
            class_result = item["classes"].get(cls, {})
            if class_result.get("status") == "detected":
                st.markdown(f"**{ui.class_title(cls)}:** правило сработало.")
                for rule in (r for r in res["rules"] if r["target"] == cls):
                    if rule_coverage(rule, [selected])["n_tp"] + rule_coverage(rule, [selected])["n_fp"]:
                        st.code(rule["rule"], language="prolog")
                        for line in counterfactuals(rule):
                            st.caption(line)
            elif class_result.get("status") == "not_detected":
                class_rules = [r for r in res["rules"] if r["target"] == cls]
                if class_rules:
                    st.markdown(f"**{ui.class_title(cls)}:**")
                    st.caption("Первое невыполненное условие: " +
                               (missing_evidence(class_rules[0], selected) or ["—"])[0])


def _show_batch(res: dict) -> None:
    st.divider()
    projects, parsed = res["projects"], res["parsed"]
    by_id = {p["id"]: p for p in projects}
    rows = []
    for item in parsed:
        labels = by_id[item["id"]].get("labels")
        predicted = [c for c in ERROR_CLASSES
                     if item["classes"].get(c, {}).get("status") == "detected"]
        unknown = [f for c in ERROR_CLASSES
                   for f in item["classes"].get(c, {}).get("missing_fields", [])]
        rows.append({"id": item["id"],
                     "истинные": ", ".join(labels) if labels is not None else "—",
                     "предсказанные": ", ".join(predicted) or "—",
                     "неизвестные поля": ", ".join(sorted(set(unknown))) or "—"})
    st.dataframe(pd.DataFrame(rows))
    labeled = [p for p in projects if p.get("labels") is not None]
    if labeled:
        parsed_by_id = {p["id"]: p for p in parsed}
        Y_true = label_matrix(labeled)
        Y_pred = []
        for p in labeled:
            item = parsed_by_id[p["id"]]
            Y_pred.append([1 if item["classes"].get(c, {}).get("status") == "detected" else 0
                           for c in ERROR_CLASSES])
        import numpy as np
        metrics = multilabel_metrics(Y_true, np.asarray(Y_pred, dtype=np.int8))
        st.subheader("Метрики на размеченной части")
        c = st.columns(3)
        c[0].metric("macro-F1", f"{metrics['macro_f1']:.3f}")
        c[1].metric("micro-F1", f"{metrics['micro_f1']:.3f}")
        c[2].metric("exact-match", f"{metrics['exact_match']:.3f}")
        st.dataframe(pd.DataFrame([{"Класс": cls, **{k: v for k, v in vals.items()
                                                     if k in ("precision", "recall", "f1", "tp", "fp", "fn")}}
                                   for cls, vals in metrics["per_class"].items()]))
    c = st.columns(2)
    c[0].download_button("Скачать пакетную диагностику (JSON)",
                         exports.diagnosis_to_json(parsed), "batch_diagnosis.json",
                         "application/json")
    c[1].download_button("Скачать пакетную диагностику (CSV)",
                         exports.diagnosis_to_csv(parsed), "batch_diagnosis.csv", "text/csv")


# ------------------------------------------------------------------ Эксперимент
def page_experiment() -> None:
    st.header("Эксперимент: ILP и решающее дерево")
    ds = _need_dataset()
    if not ds:
        return
    labeled = labeled_only(ds["projects"])
    protocol = st.selectbox(
        "Протокол оценки",
        ["Повторное разбиение", "Repeated stratified k-fold", "Кривая обучения", "Абляция параметра"],
        key="experiment_protocol")
    c = st.columns(4)
    seed = c[0].number_input("Random seed", 0, 10**9, 42)
    test_size = c[1].slider("Доля test", 0.1, 0.5, 0.2, 0.05)
    noise = c[2].selectbox("Шум меток в train", [0.0, 0.05, 0.10, 0.15, 0.20],
                           format_func=lambda x: f"{int(x * 100)}%")
    n_seeds = c[3].selectbox("Число запусков (seed, seed+1, …)", [1, 3, 5])
    with_join = st.checkbox("Добавить расширенное дерево с агрегатами «требуется ∩ отсутствует»", True,
                            help="Агрегаты вычисляются из фактов без меток, но вручную кодируют связь "
                                 "между фактами. Основной базовый метод — дерево на плоских признаках.")
    st.caption("Параметры ILP берутся со страницы «Обучение ILP» (или значения по умолчанию).")
    params = (st.session_state.model or {}).get("params_requested") or {
        "max_body_length": 3, "min_support": 2, "min_precision": 0.8,
        "max_rules_per_class": 5, "max_candidates": 20000,
        "smoothing": 1.0, "mdl_penalty": 1.0, "max_constants": 20}
    true_rules = (ds.get("meta") or {}).get("true_rules")
    st.json(params, expanded=False)

    cv_folds = 5
    cv_repeats = 3
    ablation_parameter = "max_body_length"
    ablation_values: list[float | int] = [1, 2, 3]
    if protocol == "Repeated stratified k-fold":
        d = st.columns(2)
        cv_folds = int(d[0].selectbox("Число фолдов", [3, 5, 10], index=1))
        cv_repeats = int(d[1].selectbox("Повторов", [2, 3, 5], index=1))
    elif protocol == "Кривая обучения":
        st.caption("Все точки используют один и тот же test; меняется только объём train.")
    elif protocol == "Абляция параметра":
        d = st.columns(2)
        ablation_parameter = d[0].selectbox(
            "Варьируемый параметр", ["max_body_length", "min_support", "min_precision"])
        ablation_values = {
            "max_body_length": [1, 2, 3],
            "min_support": [1, 2, 4, 8],
            "min_precision": [0.5, 0.65, 0.8, 0.9],
        }[ablation_parameter]
        d[1].caption("Значения: " + ", ".join(map(str, ablation_values)))

    if st.button("Запустить сравнение", key="run_experiment"):
        engine = _need_engine()
        if engine:
            try:
                bar = st.progress(0.0)
                with st.spinner("Обучение ILP и дерева, предсказание на test…"):
                    if protocol == "Repeated stratified k-fold":
                        result = run_repeated_stratified_cv(
                            engine, labeled, params, cv_folds, cv_repeats, int(seed),
                            float(noise), with_join,
                            progress=lambda i, n: bar.progress(i / n),
                            true_rules=true_rules)
                        st.session_state.experiment = {"analysis": result}
                    elif protocol == "Кривая обучения":
                        result = run_learning_curve(
                            engine, labeled, params, seed=int(seed), test_size=float(test_size),
                            noise=float(noise), with_join_tree=with_join,
                            progress=lambda i, n: bar.progress(i / n),
                            true_rules=true_rules)
                        st.session_state.experiment = {"analysis": result}
                    elif protocol == "Абляция параметра":
                        result = run_ablation(
                            engine, labeled, params, ablation_parameter, ablation_values,
                            int(seed), float(test_size), float(noise), with_join,
                            progress=lambda i, n: bar.progress(i / n),
                            true_rules=true_rules)
                        st.session_state.experiment = {"analysis": result}
                    else:
                        seeds = [int(seed) + i for i in range(int(n_seeds))]
                        if len(seeds) == 1:
                            exp = run_experiment(engine, labeled, params, seeds[0], float(test_size),
                                                 float(noise), with_join, true_rules=true_rules)
                            st.session_state.experiment = {"single": exp, "multi": None}
                        else:
                            multi = run_multi_seed(engine, labeled, params, seeds, float(test_size),
                                                   float(noise), with_join,
                                                   progress=lambda i, n: bar.progress(i / n),
                                                   true_rules=true_rules)
                            st.session_state.experiment = {"single": multi["runs"][0],
                                                           "multi": multi}
                bar.progress(1.0)
            except (PrologError, ValueError) as exc:
                st.error(str(exc))
    res = st.session_state.experiment
    if res:
        if "analysis" in res:
            _show_research_experiment(res["analysis"])
        else:
            _show_experiment(res)


def _show_research_experiment(result: dict) -> None:
    st.divider()
    protocol = result["protocol"]
    st.subheader(protocol.replace("_", " ").capitalize())
    if protocol == "repeated_stratified_k_fold":
        st.caption(f"{result['n_splits']} фолдов × {result['n_repeats']} повторов; "
                   f"шум меток в train: {result['noise']:.0%}.")
        rows = []
        for method, metrics in result["aggregate"].items():
            for name in ("macro_f1", "micro_f1", "exact_match"):
                ci = metrics[name]["bootstrap_ci"]
                rows.append({"Метод": method, "Метрика": name,
                             "среднее": metrics[name]["mean"], "std": metrics[name]["std"],
                             "95% CI": f"[{ci['low']:.3f}, {ci['high']:.3f}]"})
        st.dataframe(pd.DataFrame(rows))
        recovery_runs = [run for run in result["runs"] if run.get("rule_recovery")]
        if recovery_runs:
            st.markdown("**Восстановление порождающих правил (exact match)**")
            st.dataframe(pd.DataFrame([
                {"Повтор": run["meta"]["repeat"], "Фолд": run["meta"]["fold"],
                 "precision": run["rule_recovery"]["precision"],
                 "recall": run["rule_recovery"]["recall"],
                 "восстановлено": run["rule_recovery"]["recovered_rules"]}
                for run in recovery_runs
            ]))
    elif protocol == "learning_curve":
        rows = []
        for point in result["runs"]:
            for method, record in point["result"]["methods"].items():
                rows.append({"train, проектов": point["n_train"], "доля train": point["fraction"],
                             "Метод": method,
                             "macro-F1": record["metrics"]["macro_f1"],
                             "micro-F1": record["metrics"]["micro_f1"]})
        st.dataframe(pd.DataFrame(rows))
        st.caption("Общая тестовая выборка: " + ", ".join(result["test_ids"][:8]) +
                   (" …" if len(result["test_ids"]) > 8 else ""))
    else:
        rows = []
        for point in result["runs"]:
            for method, record in point["result"]["methods"].items():
                row = {"Параметр": point["parameter"], "Значение": point["value"],
                       "Метод": method, "macro-F1": record["metrics"]["macro_f1"],
                       "Правил ILP": len(point["result"]["ilp_rules"])}
                if point["result"].get("rule_recovery"):
                    row["recall порождающих правил"] = point["result"]["rule_recovery"]["recall"]
                rows.append(row)
        st.dataframe(pd.DataFrame(rows))
    st.download_button("Скачать результаты исследования (JSON)",
                       exports.experiment_to_json(result), "research_experiment.json",
                       "application/json")


def _show_experiment(res: dict) -> None:
    exp, multi = res["single"], res["multi"]
    st.divider()
    m = exp["meta"]
    st.caption(f"seed={m['seed']} , train={m['n_train']} , test={m['n_test']} , "
               f"шум меток={int(m['label_noise'] * 100)}%")
    st.subheader("Сравнение на тестовой выборке" + (f" (seed {m['seed']})" if multi else ""))
    st.dataframe(pd.DataFrame(exports.comparison_rows(exp)))
    ci_rows = []
    for method, record in exp["methods"].items():
        for metric, bounds in record.get("bootstrap_ci", {}).items():
            ci_rows.append({"Метод": method, "Метрика": metric,
                            "95% bootstrap CI": f"[{bounds['low']:.3f}, {bounds['high']:.3f}]"})
    with st.expander("Bootstrap-интервалы и парные тесты"):
        if ci_rows:
            st.markdown("**95% bootstrap confidence intervals**")
            st.dataframe(pd.DataFrame(ci_rows))
        if exp.get("paired_tests"):
            paired_rows = []
            for name, result in exp["paired_tests"].items():
                paired_rows.append({"Сравнение": name, "exact match b": result["a_only_correct"],
                                    "exact match c": result["b_only_correct"],
                                    "McNemar p": result["p_value"]})
                for cls, values in result["per_class"].items():
                    paired_rows.append({"Сравнение": f"{name} · {cls}",
                                        "exact match b": values["a_only_correct"],
                                        "exact match c": values["b_only_correct"],
                                        "McNemar p": values["p_value"]})
            st.markdown("**Точный тест Мак-Немара (парные предсказания на одинаковом test)**")
            st.dataframe(pd.DataFrame(paired_rows))
    c = st.columns(2)
    with c[0]:
        ui.overall_chart(exp)
    with c[1]:
        ui.time_chart(exp)
    st.markdown("**F1 по классам**")
    ui.f1_by_class_chart(exp)
    rows = []
    for name, meth in exp["methods"].items():
        for cls, v in meth["metrics"]["per_class"].items():
            rows.append({"Метод": name, "Класс": cls, "precision": v["precision"],
                         "recall": v["recall"], "F1": v["f1"], "support": v["support"]})
    with st.expander("Precision / recall / F1 по классам"):
        st.dataframe(pd.DataFrame(rows))
    st.markdown("**Сложность моделей**")
    ilp_c = exp["methods"]["ILP (Prolog)"]["complexity"]
    c = st.columns(4)
    c[0].metric("ILP: правил", ilp_c["n_rules"])
    c[1].metric("ILP: ср. условий в правиле", f"{ilp_c['avg_body_length']:.2f}")
    c[2].metric("ILP: всего литералов", ilp_c["total_literals"])
    c[3].metric("ILP: покрыто train", f"{ilp_c['train_covered_share']:.0%}")
    dt_c = exp["methods"]["Decision Tree"]["complexity"]
    c = st.columns(4)
    c[0].metric("Дерево: листьев (все классы)", dt_c["n_leaves"])
    c[1].metric("Дерево: узлов", dt_c["n_nodes"])
    c[2].metric("Дерево: макс. глубина", dt_c["max_depth"])
    c[3].metric("Дерево: признаков", dt_c["n_features"])
    st.caption("Время ILP включает запуск процесса SWI-Prolog; время обучения внутри Prolog: "
               f"{exp['methods']['ILP (Prolog)'].get('train_seconds_prolog', 0):.2f} с.")
    with st.expander("Правила ILP этого запуска"):
        for r in exp["ilp_rules"]:
            st.code(r["rule"], language="prolog")
    if multi:
        st.subheader(f"Несколько запусков: seed {multi['seeds'][0]}…{multi['seeds'][-1]}")
        agg = multi["aggregate"]
        st.dataframe(pd.DataFrame([{
            "Метод": n, "macro-F1": f"{a['macro_f1']['mean']:.3f} ± {a['macro_f1']['std']:.3f}",
            "micro-F1": f"{a['micro_f1']['mean']:.3f} ± {a['micro_f1']['std']:.3f}",
            "exact-match": f"{a['exact_match']['mean']:.3f} ± {a['exact_match']['std']:.3f}",
            "время обучения, с": f"{a['train_seconds']['mean']:.3f} ± {a['train_seconds']['std']:.3f}"}
            for n, a in agg.items()]))
        ui.multiseed_chart(agg)
    payload = multi if multi else exp
    c = st.columns(3)
    c[0].download_button("Скачать результаты (JSON)", exports.experiment_to_json(payload),
                         "experiment.json", "application/json")
    c[1].download_button("Скачать сравнение (CSV)", exports.comparison_to_csv(exp),
                         "comparison.csv", "text/csv")
    if c[2].button("Сохранить в папку results/"):
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = RESULTS_DIR / f"experiment_{stamp}.json"
        p.write_text(exports.experiment_to_json(payload), encoding="utf-8")
        (RESULTS_DIR / f"comparison_{stamp}.csv").write_text(exports.comparison_to_csv(exp), encoding="utf-8")
        st.success(f"Сохранено: {p}")


# ------------------------------------------------------------------ Экспорт / О проекте
def page_export() -> None:
    st.header("Экспорт / О проекте")
    _load_model_upload()
    doc, ds = st.session_state.model, st.session_state.dataset
    c = st.columns(3)
    if doc:
        c[0].download_button("Модель (JSON)", exports.model_to_json(doc), "model.json", "application/json")
        c[1].download_button("Правила (.pl)", exports.rules_to_prolog(doc), "learned_rules.pl", "text/plain")
        if c[2].button("Сохранить модель в results/"):
            RESULTS_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"model_{stamp}.json"
            path.write_text(exports.model_to_json(doc), encoding="utf-8")
            st.success(f"Сохранено: {path}")
    else:
        c[0].caption("Модель пока не обучена.")
    if ds:
        st.download_button("Набор данных (JSON)", dumps(dataset_to_json(ds["projects"])),
                           "dataset.json", "application/json")
    d = st.session_state.diagnosis
    if d:
        c = st.columns(2)
        c[0].download_button("Последняя диагностика (JSON)",
                             exports.diagnosis_to_json([d["parsed"]],
                                                       {"warnings": d.get("warnings", [])}),
                             "diagnosis.json", "application/json")
        c[1].download_button("Последняя диагностика (CSV)",
                             exports.diagnosis_to_csv([d["parsed"]]),
                             "diagnosis.csv", "text/csv")
    b = st.session_state.batch_results
    if b:
        c = st.columns(2)
        c[0].download_button("Пакетная диагностика (JSON)",
                             exports.diagnosis_to_json(b["parsed"]), "batch_diagnosis.json",
                             "application/json")
        c[1].download_button("Пакетная диагностика (CSV)",
                             exports.diagnosis_to_csv(b["parsed"]), "batch_diagnosis.csv",
                             "text/csv")
    st.subheader("Версии программного обеспечения")
    engine, error, _ = engine_status()
    st.json(environment_info(engine), expanded=True)
    st.subheader("Ограничения прототипа")
    st.markdown(
        "- Язык гипотез: положительные и отрицательные литералы из заданного словаря, "
        "константы для `uses_build_tool`/`uses_runtime`, `\\=` между переменными одного типа, "
        "длина тела до 3.\n"
        "- Отбор правил использует сглаженную точность (m-оценку) и MDL-штраф за длину; "
        "порог `min_precision` применяется к сглаженной точности.\n"
        "- Метки считаются полными: пустой список = «ошибок нет» (отрицательный пример).\n"
        "- «Неизвестно» (`null`) не равно «отсутствует»: при нехватке данных диагностика "
        "показывает отдельный статус.\n"
        "- Реальные проекты разбираются по `requirements.txt`, `pyproject.toml`, lock-файлам, "
        "`.env`/`.env.example` и config-примерам; отсутствие lock-файла означает «неизвестно», "
        "а не «пакет отсутствует».\n"
        "- Правила применяются как есть: система только обнаруживает известные закономерности "
        "и не гарантирует корректность проекта. Автоматическое исправление файлов не реализовано.")
    if error:
        st.caption(f"SWI-Prolog: {error}")
