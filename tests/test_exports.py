import json

import pytest

from core import exports
from core.diagnosis import parse_diagnosis, unknown_field_warnings

RULE = {"target": "missing_dependency",
        "rule": "has_error(Project, missing_dependency) :-\n    requires_package(Project, Package),\n    package_absent(Project, Package).",
        "body": [{"predicate": "requires_package", "var": 1, "args": ["Project", "Package"]},
                 {"predicate": "package_absent", "var": 1, "args": ["Project", "Package"]}],
        "tp": 10, "fp": 1, "fn": 2, "precision": 0.9091, "recall": 0.8333, "f1": 0.8696, "length": 2}
RESPONSE = {"status": "ok", "model": {"rules": [RULE], "classes": {}},
            "stats": {"max_body_length": 3, "min_support": 2, "min_precision": 0.8,
                      "max_rules_per_class": 5, "train_size": 100},
            "log": {"levels": [], "rejected": []}, "warnings": []}


def test_model_json_and_prolog_export():
    doc = exports.build_model_document(RESPONSE, {"min_support": 2}, {"n_train": 100}, {"python": "3"})
    loaded = exports.load_model_document(exports.model_to_json(doc))
    assert loaded["model"]["rules"][0]["tp"] == 10
    assert loaded["params_requested"] == {"min_support": 2}
    pl = exports.rules_to_prolog(doc)
    assert RULE["rule"] in pl and "% missing_config_key: надёжное правило не найдено" in pl
    assert pl.count(":- discontiguous has_error/2.") == 1


def test_model_import_rejects_bad_format_and_unknown_literals():
    with pytest.raises(ValueError, match="не файл модели"):
        exports.load_model_document('{"format":"other"}')
    doc = exports.build_model_document(RESPONSE, {}, {})
    doc["model"]["rules"][0]["body"][0]["predicate"] = "unknown_predicate"
    with pytest.raises(ValueError, match="неизвестный предикат"):
        exports.load_model_document(doc)


def test_diagnosis_parse_and_exports():
    resp = {"results": [{"id": "p1", "classes": {
        "missing_dependency": {"status": "detected", "matches": [{"rule_index": 1, "proofs": [
            {"bindings": {"Package": "pandas"},
             "facts": [{"predicate": "requires_package", "args": ["p1", "pandas"]},
                       {"predicate": "package_absent", "args": ["p1", "pandas"]}]}]}]},
        "incompatible_version": {"status": "no_rules", "matches": []}}}]}
    parsed = parse_diagnosis(resp, [RULE])
    assert parsed[0]["classes"]["missing_dependency"]["matches"][0]["rule"] == RULE["rule"]
    assert "Package=pandas" in exports.diagnosis_to_csv(parsed)
    assert json.loads(exports.diagnosis_to_json(parsed))["results"][0]["id"] == "p1"


def test_unknown_field_warning_only_when_rule_needs_it():
    proj = {"id": "x", "labels": None, "facts": {"required_packages": ["a"], "absent_packages": None}}
    assert unknown_field_warnings(proj, [RULE])
    proj["facts"]["absent_packages"] = []
    assert not unknown_field_warnings(proj, [RULE])
