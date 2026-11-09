from pathlib import Path

from core.project_loader import (
    load_project_dir,
    parse_env_keys,
    parse_lock_text,
    parse_pyproject_text,
    parse_requirements_text,
    project_from_files,
)


def test_requirements_and_poetry_lock_give_absent_and_mismatch():
    files = {
        "requirements.txt": """
# comment
numpy>=2.0
pandas==3.0.0
scipy==1.0
requests[security]>=2.0 ; python_version >= "3.10"
""",
        "poetry.lock": """
[[package]]
name = "numpy"
version = "2.5.4"

[[package]]
name = "scipy"
version = "1.5.0"

[[package]]
name = "requests"
version = "2.32.0"
""",
    }
    parsed = project_from_files(files, "p1")
    facts = parsed.project["facts"]
    assert facts["required_packages"] == ["numpy", "pandas", "requests", "scipy"]
    assert facts["absent_packages"] == ["pandas"]
    assert facts["version_mismatches"] == ["scipy"]
    assert facts["build_tools"] == ["pip", "poetry"]
    assert facts["runtimes"] == ["python"]


def test_pyproject_pep621_and_optional_dependencies():
    text = """
[project]
name = "demo"
dependencies = ["numpy>=2", "pandas"]

[project.optional-dependencies]
dev = ["pytest>=8"]
"""
    reqs = parse_pyproject_text(text)
    names = sorted({r.name for r in reqs})
    assert names == ["numpy", "pandas", "pytest"]


def test_no_lock_means_unknown_not_empty():
    parsed = project_from_files({"requirements.txt": "numpy\n"}, "p2")
    facts = parsed.project["facts"]
    assert facts["required_packages"] == ["numpy"]
    assert facts["absent_packages"] is None
    assert facts["version_mismatches"] is None
    assert any("lock" in n.lower() for n in parsed.notes)


def test_env_and_config_keys_are_distinguished_from_unknown():
    files = {
        ".env.example": "API_TOKEN=\nDATABASE_URL=\n",
        ".env": "API_TOKEN=secret\n",
        "config.example.toml": 'database = "x"\nport = 8000\n',
        "config.toml": 'database = "x"\n',
    }
    facts = project_from_files(files, "p3").project["facts"]
    assert facts["required_env_variables"] == ["API_TOKEN", "DATABASE_URL"]
    assert facts["absent_env_variables"] == ["DATABASE_URL"]
    assert facts["required_config_keys"] == ["database", "port"]
    assert facts["absent_config_keys"] == ["port"]


def test_missing_manifest_and_config_are_unknown():
    facts = project_from_files({}, "p4").project["facts"]
    assert facts["required_packages"] is None
    assert facts["required_config_keys"] is None
    assert facts["required_env_variables"] is None


def test_parse_requirements_skips_unsupported_lines():
    reqs = parse_requirements_text("""
-r other.txt
-e .
--index-url https://example.com/simple
numpy==2.0  # inline
bad requirement !!!
""")
    assert [r.name for r in reqs] == ["numpy"]


def test_pipfile_lock_and_uv_lock():
    pip = '{"default": {"numpy": {"version": "==2.0.0"}}, "develop": {}}'
    assert parse_lock_text("Pipfile.lock", pip) == {"numpy": "2.0.0"}
    uv = '[[package]]\nname = "numpy"\nversion = "2.1.0"\n'
    assert parse_lock_text("uv.lock", uv) == {"numpy": "2.1.0"}


def test_env_parser_handles_export_and_comments():
    assert parse_env_keys("# x\nexport API_TOKEN=1\nDB_HOST: localhost\nnot a key\n") == {
        "API_TOKEN", "DB_HOST"}


def test_load_project_dir(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==2.0\n", encoding="utf-8")
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n', encoding="utf-8")
    (tmp_path / ".env.example").write_text("API_TOKEN=\n", encoding="utf-8")
    parsed = load_project_dir(tmp_path, "real")
    assert parsed.project["id"] == "real"
    assert parsed.project["facts"]["absent_packages"] == []
    assert parsed.project["facts"]["version_mismatches"] == []
    assert parsed.project["facts"]["required_env_variables"] == ["API_TOKEN"]
