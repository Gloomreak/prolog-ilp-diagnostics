from core.dataset import ERROR_CLASSES
from core.diagnosis import mark_insufficient_data
from core.rule_analysis import counterfactuals, covers, missing_evidence

RULE = {
    "target": "missing_dependency",
    "body": [
        {"predicate": "requires_package", "var": 1},
        {"predicate": "package_absent", "var": 1},
    ],
}


def _parsed():
    return {"id": "p1", "classes": {
        cls: {"status": "not_detected", "matches": [], "missing_fields": []}
        for cls in ERROR_CLASSES
    }}


def test_unknown_required_field_has_distinct_insufficient_status():
    project = {"id": "p1", "facts": {"required_packages": ["pandas"], "absent_packages": None}}
    parsed = mark_insufficient_data(project, [RULE], _parsed())
    assert parsed["classes"]["missing_dependency"]["status"] == "insufficient_data"
    assert parsed["classes"]["missing_dependency"]["missing_fields"] == ["absent_packages"]
    assert parsed["classes"]["missing_config_key"]["status"] == "not_detected"


def test_rule_analysis_does_not_treat_unknown_as_known_empty():
    project = {"id": "p1", "labels": ["missing_dependency"], "facts": {
        "required_packages": ["pandas"], "absent_packages": None}}
    assert not covers(RULE, project)
    assert missing_evidence(RULE, project) == ["поле «absent_packages» неизвестно"]


def test_rule_analysis_reports_the_first_failed_literal_and_counterfactual():
    project = {"id": "p1", "facts": {
        "required_packages": ["pandas"], "absent_packages": ["scipy"]}}
    assert missing_evidence(RULE, project) == [
        "нет общего значения для package_absent и связанных условий (ожидалось pandas)"]
    match = {"proofs": [{"facts": [
        {"predicate": "requires_package", "args": ["p1", "pandas"]},
        {"predicate": "package_absent", "args": ["p1", "pandas"]},
    ]}]}
    assert len(counterfactuals(match)) == 2
