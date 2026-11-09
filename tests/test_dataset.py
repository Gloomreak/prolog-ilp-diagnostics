import json
from pathlib import Path

from core.dataset import (
    ERROR_CLASSES,
    dataset_to_json,
    load_dataset_bytes,
    load_dataset_file,
    summary,
    unknown_fields,
    validate_dataset,
    validate_project,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def _doc(*projects, **extra):
    return {"schema_version": "1.0", "projects": list(projects), **extra}


def _proj(pid="p1", labels=(), **facts):
    return {"id": pid, "labels": list(labels), "facts": facts}


def test_demo_dataset_is_valid():
    res = load_dataset_file(DATA / "demo_dataset.json")
    assert res.ok and len(res.projects) == 500 and not res.errors


def test_example_project_for_diagnosis_is_valid():
    obj = json.loads((DATA / "example_project_missing_dependency.json").read_text(encoding="utf-8"))
    proj, issues = validate_project(obj, 0, require_labels=False)
    assert proj is not None and proj["labels"] is None and not issues


def test_invalid_example_reports_record_and_field():
    res = load_dataset_file(DATA / "invalid_example.json")
    text = "\n".join(str(i) for i in res.errors)
    assert "bad_label" in text and "labels" in text
    assert "bad_type" in text and "facts.required_packages" in text
    assert "повторяющийся id" in text
    assert res.n_rejected == 5 and len(res.projects) == 2
    assert any("comment" in str(w) for w in res.warnings)       # неизвестное поле -> предупреждение


def test_schema_version_required_and_checked():
    r = validate_dataset({"projects": []})
    assert r.fatal and "schema_version" in str(r.issues[0])
    r = validate_dataset({"schema_version": "9.9", "projects": [_proj()]})
    assert r.fatal and "неподдерживаемая версия" in str(r.issues[0])


def test_projects_must_be_nonempty_array():
    assert validate_dataset({"schema_version": "1.0", "projects": []}).fatal
    assert validate_dataset({"schema_version": "1.0", "projects": "x"}).fatal
    assert validate_dataset([]).fatal


def test_bad_json_and_encoding_give_readable_errors():
    r = load_dataset_bytes(b"{not json")
    assert r.fatal and "некорректный JSON" in str(r.issues[0])
    r = load_dataset_bytes("{}".encode("utf-16"))
    assert r.fatal
    r = load_dataset_bytes(b"\xef\xbb\xbf" + json.dumps(_doc(_proj())).encode())   # BOM допустим
    assert r.ok


def test_empty_labels_is_negative_example_but_missing_labels_is_error():
    p, issues = validate_project({"id": "a", "labels": [], "facts": {}}, 0)
    assert p["labels"] == [] and not issues
    p, issues = validate_project({"id": "a", "facts": {}}, 0)
    assert p is None and issues[0].field == "labels"


def test_unknown_differs_from_absent():
    p, _ = validate_project({"id": "a", "labels": [], "facts": {"absent_packages": [],
                                                                 "required_packages": None}}, 0)
    assert p["facts"]["absent_packages"] == []          # известно: пусто
    assert p["facts"]["required_packages"] is None      # неизвестно
    assert "required_packages" in unknown_fields(p) and "absent_packages" not in unknown_fields(p)


def test_duplicates_are_normalised_and_objects_accepted():
    p, issues = validate_project({"id": "a", "labels": ["missing_dependency", "missing_dependency"],
                                  "facts": {"required_packages": ["numpy", " numpy ", {"name": "pandas"}]}}, 0)
    assert p["labels"] == ["missing_dependency"]
    assert p["facts"]["required_packages"] == ["numpy", "pandas"]
    assert any(i.level == "warning" for i in issues)


def test_bad_values_rejected():
    for bad in ([1], [""], [{"x": 1}], [["a"]]):
        p, issues = validate_project({"id": "a", "labels": [], "facts": {"required_packages": bad}}, 0)
        assert p is None and issues[0].level == "error"


def test_summary_counts():
    ps = [_proj("a", ["missing_dependency"]), _proj("b", []),
          _proj("c", ["missing_dependency", "missing_config_key"])]
    res = validate_dataset(_doc(*ps))
    s = summary(res.projects, res)
    assert s["n_projects"] == 3 and s["class_counts"]["missing_dependency"] == 2
    assert s["class_negatives"]["missing_dependency"] == 1
    assert s["n_without_errors"] == 1 and s["n_multi_label"] == 1


def test_roundtrip():
    res = load_dataset_file(DATA / "demo_dataset.json")
    again = validate_dataset(dataset_to_json(res.projects))
    assert again.projects == res.projects
    assert set(ERROR_CLASSES) == set(summary(res.projects)["class_counts"])
