"""Интеграционные тесты: Python + настоящий SWI-Prolog.
Без установленного swipl эти тесты пропускаются (фикстура engine)."""
import re
import subprocess
from pathlib import Path

import pytest

from core.dataset import ERROR_CLASSES
from core.diagnosis import parse_diagnosis
from core.evaluation import run_experiment, split_projects
from core.exports import build_model_document, load_model_document, model_to_json
from core.generator import generate_projects
from core.prolog_runner import (
    PrologEngine,
    PrologEngineError,
    PrologNotFoundError,
    check_swipl,
    find_swipl,
)
from tests.helpers import counts, rule_covers

ROOT = Path(__file__).resolve().parent.parent
PARAMS = {"max_body_length": 3, "min_support": 2, "min_precision": 0.8, "max_rules_per_class": 5}


@pytest.fixture(scope="module")
def trained(engine, small_dataset):
    train, test = split_projects(small_dataset, 0.2, 1)
    return train, test, engine.train(train, PARAMS)


def test_swipl_detected(engine):
    info = engine.info()
    assert info.version is not None and info.version[0] >= 8


def test_missing_swipl_gives_clear_error():
    with pytest.raises(PrologNotFoundError):
        check_swipl("/definitely/not/here/swipl")
    assert find_swipl("/definitely/not/here/swipl") is None


def test_plunit_suite_passes():
    path = find_swipl()
    if not path:
        pytest.skip("swipl не найден")
    proc = subprocess.run([path, "-g", "run_tests", "-t", "halt", str(ROOT / "prolog" / "tests.pl")],
                          capture_output=True, text=True, cwd=ROOT, timeout=600,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ilp_learns_rules_for_every_class_with_joined_literals(trained):
    train, _, resp = trained
    rules = resp["model"]["rules"]
    assert resp["status"] == "ok"
    for cls in ERROR_CLASSES:
        cls_rules = [r for r in rules if r["target"] == cls]
        assert cls_rules, f"нет правила для {cls}"
        best = cls_rules[0]
        # осмысленное правило связывает «требуется» и «нарушено» ОДНОЙ переменной
        assert best["precision"] >= 0.95 and best["recall"] >= 0.9
        assert len({lit["var"] for lit in best["body"]}) == 1 and len(best["body"]) == 2


def test_prolog_metrics_equal_independent_python_count(trained):
    train, _, resp = trained
    for r in resp["model"]["rules"]:
        assert counts(r, train) == (r["tp"], r["fp"], r["fn"]), r["rule"]


def test_diagnosis_agrees_with_independent_rule_checker(engine, trained):
    train, test, resp = trained
    rules = resp["model"]["rules"]
    parsed = parse_diagnosis(engine.diagnose(rules, test, max_proofs=2), rules)
    for proj, res in zip(test, parsed):
        assert proj["id"] == res["id"]
        for cls in ERROR_CLASSES:
            expected = any(rule_covers(r["body"], proj) for r in rules if r["target"] == cls)
            got = res["classes"][cls]["status"] == "detected"
            assert got == expected, (proj["id"], cls)


def test_diagnosis_explains_with_substitution_and_facts(engine, trained):
    _, _, resp = trained
    rules = resp["model"]["rules"]
    proj = {"id": "brand_new", "labels": None, "facts": {
        "required_packages": ["zzz_pkg", "numpy"], "absent_packages": ["zzz_pkg"],
        "version_mismatches": [], "required_config_keys": [], "absent_config_keys": [],
        "required_env_variables": [], "absent_env_variables": [], "build_tools": [], "runtimes": []}}
    parsed = parse_diagnosis(engine.diagnose(rules, [proj], max_proofs=3), rules)[0]
    cr = parsed["classes"]["missing_dependency"]
    assert cr["status"] == "detected"
    proof = cr["matches"][0]["proofs"][0]
    assert proof["bindings"] == {"Package": "zzz_pkg"}       # правило обобщилось на ранее не виденный пакет
    facts = {(f["predicate"], tuple(f["args"])) for f in proof["facts"]}
    assert ("requires_package", ("brand_new", "zzz_pkg")) in facts
    assert ("package_absent", ("brand_new", "zzz_pkg")) in facts
    assert parsed["classes"]["missing_config_key"]["status"] == "not_detected"


def test_model_json_roundtrip_preserves_predictions(engine, trained):
    train, test, response = trained
    params = {**PARAMS, "smoothing": 1.0, "mdl_penalty": 1.0, "max_constants": 20}
    document = build_model_document(response, params, {"n_train": len(train)})
    loaded = load_model_document(model_to_json(document))
    project = test[0]
    original = parse_diagnosis(engine.diagnose(response["model"]["rules"], [project]), 
                               response["model"]["rules"])
    restored_rules = loaded["model"]["rules"]
    restored = parse_diagnosis(engine.diagnose(restored_rules, [project]), restored_rules)
    assert original == restored


def test_hard_generator_learns_context_constant(engine):
    projects = generate_projects(140, seed=51, mode="hard")
    response = engine.train(projects, {**PARAMS, "max_candidates": 20000})
    rules = response["model"]["rules"]
    contextual = [
        r for r in rules if r["target"] in {"missing_dependency", "incompatible_version"}
        and any(lit.get("const") in {"poetry", "python"} for lit in r["body"])
    ]
    assert contextual
    assert all(r["length"] >= 3 for r in contextual)


def test_changing_training_examples_changes_the_model(engine, small_dataset):
    train, _ = split_projects(small_dataset, 0.2, 1)
    base = engine.train(train, PARAMS)["model"]["rules"]
    # снимаем все метки missing_dependency -> правило для этого класса не должно появиться
    stripped = [{**p, "labels": [c for c in p["labels"] if c != "missing_dependency"]} for p in train]
    changed = engine.train(stripped, PARAMS)
    assert any(r["target"] == "missing_dependency" for r in base)
    assert not any(r["target"] == "missing_dependency" for r in changed["model"]["rules"])
    assert changed["model"]["classes"]["missing_dependency"]["status"] == "no_positive_examples"
    # метки зависимостей приписаны другому классу (а его прежние метки сняты):
    # правило для этого класса должно «переехать» за примерами — теперь оно про пакеты
    moved = [{**p, "labels": ["missing_config_key" if c == "missing_dependency" else c
                             for c in p["labels"] if c != "missing_config_key"]} for p in train]
    rules = [r for r in engine.train(moved, PARAMS)["model"]["rules"] if r["target"] == "missing_config_key"]
    assert rules and {literal["predicate"] for literal in rules[0]["body"]} == {
        "requires_package", "package_absent"}
    old = [r for r in base if r["target"] == "missing_config_key"]
    assert {literal["predicate"] for literal in old[0]["body"]} == {
        "requires_config_key", "config_key_absent"}


def test_unrelated_labels_give_no_rule_instead_of_a_made_up_one(engine, small_dataset):
    import random
    rng = random.Random(0)
    shuffled = [{**p, "labels": ["missing_dependency"] if rng.random() < 0.3 else []}
                for p in small_dataset]
    resp = engine.train(shuffled, {**PARAMS, "min_precision": 0.9})
    assert not any(r["target"] == "missing_dependency" for r in resp["model"]["rules"])
    assert resp["model"]["classes"]["missing_dependency"]["status"] == "no_rule_found"
    assert any(w["code"] == "no_rule_found" for w in resp["warnings"])


def test_max_candidates_warning(engine, small_dataset):
    resp = engine.train(small_dataset[:60], {**PARAMS, "max_candidates": 5})
    assert resp["stats"]["truncated"] is True
    assert any(w["code"] == "max_candidates_reached" for w in resp["warnings"])


def test_training_log_has_required_fields(trained):
    _, _, resp = trained
    s = resp["stats"]
    assert s["candidates_generated"] >= s["candidates_tested"] > 0 and s["elapsed_seconds"] >= 0
    assert resp["log"]["levels"][0]["length"] == 1 and "rejected" in resp["log"]
    for r in resp["log"]["rejected"]:
        assert r["reasons"]


def test_labels_do_not_leak_into_background_knowledge(engine, small_dataset):
    """Две разметки с ОДИНАКОВЫМИ фактами: правила для класса без положительных
    примеров обязаны отсутствовать, значит факты сами по себе класс не выдают."""
    nolabel = [{**p, "labels": []} for p in small_dataset]
    resp = engine.train(nolabel, PARAMS)
    assert resp["model"]["rules"] == []


def test_invalid_rule_is_reported_not_crashing(engine, small_dataset):
    bad = [{"target": "missing_dependency", "body": [{"predicate": "no_such_pred", "var": 1}]}]
    with pytest.raises(PrologEngineError):
        engine.diagnose(bad, small_dataset[:1])


def test_engine_with_missing_engine_file_reports_error(engine, tmp_path):
    broken = PrologEngine(engine.info().path, engine_file=tmp_path / "nope.pl")
    with pytest.raises(PrologEngineError):
        broken.ping()


def test_experiment_same_test_set_and_reproducible(engine):
    ps = generate_projects(120, 21)
    a = run_experiment(engine, ps, PARAMS, seed=5)
    b = run_experiment(engine, ps, PARAMS, seed=5)
    n_test = a["meta"]["n_test"]
    for m in a["methods"].values():
        assert m["metrics"]["n_samples"] == n_test
    for name in a["methods"]:
        assert a["methods"][name]["metrics"]["per_class"] == b["methods"][name]["metrics"]["per_class"]
    for key in ("macro_f1", "micro_f1", "exact_match"):
        assert key in a["methods"]["ILP (Prolog)"]["metrics"]
    assert a["meta"]["environment"]["python"] and a["meta"]["ilp_params"] == PARAMS
    assert a["methods"]["ILP (Prolog)"]["metrics"]["macro_f1"] > 0.9


def test_no_prebuilt_rules_in_sources():
    """Эвристика AC-04: в исходниках нет готовых правил has_error(_, <класс>) :- ..."""
    pattern = re.compile(r"has_error\(\s*\w+\s*,\s*(missing_|incompatible_)\w*\s*\)\s*:-")
    for path in list((ROOT / "core").glob("*.py")) + [ROOT / "prolog" / "ilp_engine.pl",
                                                       ROOT / "prolog" / "data_io.pl", ROOT / "app.py"] \
            + list((ROOT / "ui").glob("*.py")):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
