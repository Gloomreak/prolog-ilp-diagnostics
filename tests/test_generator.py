import json

from core.dataset import ERROR_CLASSES, FACT_FIELDS, validate_dataset
from core.generator import generate_document, generate_projects, true_rule_specs
from tests.helpers import ground_truth


def test_same_seed_same_dataset_different_seed_differs():
    a = json.dumps(generate_projects(120, 5))
    assert a == json.dumps(generate_projects(120, 5))
    assert a != json.dumps(generate_projects(120, 6))


def test_generated_document_is_valid_and_sizes_work():
    for n in (1, 50, 500):
        res = validate_dataset(generate_document(n, 1))
        assert res.ok and len(res.projects) == n and not res.errors


def test_labels_agree_with_independent_ground_truth():
    for seed in range(5):
        for p in generate_projects(200, seed):
            assert set(p["labels"]) == ground_truth(p), p["id"]


def test_each_class_has_positive_and_negative_cases_and_multilabel():
    ps = generate_projects(300, 42)
    for c in ERROR_CLASSES:
        pos = sum(c in p["labels"] for p in ps)
        assert 0 < pos < len(ps)
    assert any(len(p["labels"]) > 1 for p in ps)
    assert any(not p["labels"] for p in ps)


def test_distractors_exist():
    ps = generate_projects(300, 42)
    # пакет отсутствует, но не требуется; версия не совпадает у не требуемого пакета
    assert any(set(p["facts"]["absent_packages"]) - set(p["facts"]["required_packages"]) for p in ps)
    assert any(set(p["facts"]["version_mismatches"]) - set(p["facts"]["required_packages"]) for p in ps)
    assert any(set(p["facts"]["absent_config_keys"]) - set(p["facts"]["required_config_keys"]) for p in ps)
    # контрпримеры для голого package_absent: есть проекты с absent, но без ошибки
    assert any(p["facts"]["absent_packages"] and "missing_dependency" not in p["labels"] for p in ps)


def test_labels_are_not_in_facts():
    for p in generate_projects(50, 1):
        assert set(p["facts"]) == set(FACT_FIELDS)
        blob = json.dumps(p["facts"])
        assert not any(c in blob for c in ERROR_CLASSES)


def test_invalid_arguments():
    import pytest
    with pytest.raises(ValueError):
        generate_projects(0, 1)
    with pytest.raises(ValueError):
        generate_projects(10, 1, error_rate=1.5)


def test_hard_generator_requires_context_in_ground_truth_rules():
    rules = {r["target"]: r["body"] for r in true_rule_specs("hard")}
    assert len(rules["missing_dependency"]) == 3
    assert {"predicate": "uses_build_tool", "const": "poetry"} in rules["missing_dependency"]
    assert {"predicate": "uses_runtime", "const": "python"} in rules["incompatible_version"]
    doc = generate_document(20, 2, mode="hard")
    assert doc["meta"]["true_rules"] == true_rule_specs("hard")
