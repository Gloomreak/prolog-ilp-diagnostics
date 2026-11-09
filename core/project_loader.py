"""Разбор реального проекта в факты схемы диагностики.

Поддерживаются:
  * requirements.txt (включая -r/-c, extras, environment markers, комментарии);
  * pyproject.toml ([project].dependencies, [tool.poetry.dependencies], optional);
  * lock-файлы: poetry.lock, Pipfile.lock, requirements.lock / *freeze*.txt, uv.lock;
  * .env / .env.example (наличие переменных, значения не важны);
  * config.example.json/toml/yaml и config.json/toml/yaml (ключи).

Модуль только читает файлы и формирует facts. Он не содержит диагностических
правил: правила по-прежнему выводит Prolog из размеченных примеров.
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

FACT_FIELDS = (
    "required_packages", "absent_packages", "version_mismatches",
    "required_config_keys", "absent_config_keys",
    "required_env_variables", "absent_env_variables",
    "build_tools", "runtimes",
)

_ENV_EXAMPLE = (".env.example", ".env.sample", ".env.template", "env.example")
_ENV_ACTUAL = (".env", ".env.local", "env")
_CONFIG_EXAMPLE = ("config.example.json", "config.example.toml",
                   "config.example.yaml", "config.example.yml",
                   "config.sample.json", "config.sample.toml")
_CONFIG_ACTUAL = ("config.json", "config.toml", "config.yaml", "config.yml")
_LOCK_NAMES = ("poetry.lock", "Pipfile.lock", "requirements.lock",
               "requirements-freeze.txt", "pip-freeze.txt", "uv.lock")


@dataclass
class ParsedProject:
    project: dict
    files_used: list[str]
    notes: list[str]


def _empty_facts() -> dict[str, list[str] | None]:
    return {name: None for name in FACT_FIELDS}


def parse_requirements_text(text: str) -> list[Requirement]:
    """Строки requirements.txt -> Requirement. Неразобранные строки пропускаются."""
    out: list[Requirement] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "-c ", "--requirement ", "--constraint ")):
            continue
        if line.startswith(("-e ", "--editable ")):
            continue
        line = line.split(" #", 1)[0].strip()
        try:
            out.append(Requirement(line))
        except InvalidRequirement:
            continue
    return out


def parse_pyproject_text(text: str) -> list[Requirement]:
    """Зависимости из pyproject.toml: PEP 621 и Poetry (main + optional, без dev)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    specs: list[str] = []
    project = data.get("project") or {}
    specs.extend(str(x) for x in (project.get("dependencies") or []))
    for group in (project.get("optional-dependencies") or {}).values():
        specs.extend(str(x) for x in (group or []))
    poetry = ((data.get("tool") or {}).get("poetry") or {})
    for name, value in (poetry.get("dependencies") or {}).items():
        if name.lower() == "python":
            continue
        if isinstance(value, str):
            specs.append(f"{name}{value}" if value and value[0] in "<>=!~" else f"{name}{value}")
        elif isinstance(value, dict):
            version = value.get("version")
            if version:
                specs.append(f"{name}{version}")
            else:
                specs.append(name)
    out: list[Requirement] = []
    for spec in specs:
        try:
            out.append(Requirement(spec))
        except InvalidRequirement:
            continue
    return out


def parse_lock_text(name: str, text: str) -> dict[str, str]:
    """Возвращает {canonical_name: version}; пустой словарь — не удалось разобрать."""
    name = name.lower()
    if name == "poetry.lock" or name == "uv.lock":
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return {}
        result = {}
        for pkg in data.get("package") or []:
            if isinstance(pkg, dict) and pkg.get("name"):
                result[canonicalize_name(pkg["name"])] = str(pkg.get("version") or "")
        return result
    if name == "pipfile.lock":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        result = {}
        for section in ("default", "develop"):
            for pkg, info in (data.get(section) or {}).items():
                version = ""
                if isinstance(info, dict):
                    version = str(info.get("version") or "").lstrip("=")
                result[canonicalize_name(pkg)] = version
        return result
    # requirements.lock / pip freeze
    result = {}
    for req in parse_requirements_text(text):
        version = ""
        for spec in req.specifier:
            if spec.operator == "==":
                version = spec.version
                break
        result[canonicalize_name(req.name)] = version
    return result


def parse_env_keys(text: str) -> set[str]:
    keys = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key = re.split(r"[=:]", line, maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            keys.add(key)
    return keys


def parse_config_keys(name: str, text: str) -> set[str]:
    name = name.lower()
    try:
        if name.endswith(".json"):
            data = json.loads(text)
        elif name.endswith(".toml"):
            data = tomllib.loads(text)
        else:
            return {m.group(1) for m in re.finditer(r"^([A-Za-z_][\w.-]*)\s*:", text, re.M)}
    except (json.JSONDecodeError, tomllib.TOMLDecodeError):
        return set()
    if isinstance(data, dict):
        return {str(k) for k in data}
    return set()


def _pick(files: dict[str, str], names: tuple[str, ...]) -> tuple[str | None, str | None]:
    lower = {Path(k).name.lower(): (Path(k).name, v) for k, v in files.items()}
    for name in names:
        if name in lower:
            return lower[name]
    return None, None


def _merge_requirements(groups: list[list[Requirement]]) -> dict[str, Requirement]:
    merged: dict[str, Requirement] = {}
    for group in groups:
        for req in group:
            key = canonicalize_name(req.name)
            if key not in merged:
                merged[key] = req
    return merged


def project_from_files(files: dict[str, str], project_id: str = "project",
                       labels: list[str] | None = None) -> ParsedProject:
    """Собирает один проект из словаря {имя_файла: текст}."""
    notes: list[str] = []
    used: list[str] = []
    facts = _empty_facts()

    req_specs: list[list[Requirement]] = []
    req_name, req_text = _pick(files, ("requirements.txt",))
    if req_text is not None:
        req_specs.append(parse_requirements_text(req_text))
        used.append(req_name or "requirements.txt")
    py_name, py_text = _pick(files, ("pyproject.toml",))
    if py_text is not None:
        req_specs.append(parse_pyproject_text(py_text))
        used.append(py_name or "pyproject.toml")
    required = _merge_requirements(req_specs)
    facts["required_packages"] = sorted(required) if req_specs else None
    if not req_specs:
        notes.append("Не найден requirements.txt/pyproject.toml: зависимости неизвестны.")

    lock_name, lock_text = _pick(files, _LOCK_NAMES)
    installed: dict[str, str] | None = None
    if lock_name and lock_text is not None:
        installed = parse_lock_text(lock_name, lock_text)
        used.append(lock_name)
    if installed is None or facts["required_packages"] is None:
        facts["absent_packages"] = None
        facts["version_mismatches"] = None
        notes.append("Нет lock-файла или манифеста: отсутствующие пакеты и версии неизвестны.")
    else:
        absent = sorted(name for name in required if name not in installed)
        mismatches: list[str] = []
        for name, req in required.items():
            version = installed.get(name)
            if not version or not req.specifier:
                continue
            try:
                ver = Version(version)
            except InvalidVersion:
                continue
            try:
                spec = SpecifierSet(str(req.specifier))
            except InvalidSpecifier:
                continue
            if not spec.contains(ver, prereleases=True):
                mismatches.append(name)
        facts["absent_packages"] = absent
        facts["version_mismatches"] = sorted(set(mismatches))

    env_name, env_text = _pick(files, _ENV_EXAMPLE)
    if env_text is not None:
        required_env = parse_env_keys(env_text)
        facts["required_env_variables"] = sorted(required_env)
        used.append(env_name or ".env.example")
        actual_name, actual_text = _pick(files, _ENV_ACTUAL)
        if actual_text is not None:
            present = parse_env_keys(actual_text)
            facts["absent_env_variables"] = sorted(required_env - present)
            used.append(actual_name or ".env")
        else:
            facts["absent_env_variables"] = None
            notes.append("Нет .env: отсутствие переменных окружения неизвестно.")
    else:
        facts["required_env_variables"] = None
        facts["absent_env_variables"] = None

    cfg_name, cfg_text = _pick(files, _CONFIG_EXAMPLE)
    if cfg_text is not None:
        required_cfg = parse_config_keys(cfg_name or "", cfg_text)
        facts["required_config_keys"] = sorted(required_cfg)
        used.append(cfg_name or "config.example.json")
        actual_name, actual_text = _pick(files, _CONFIG_ACTUAL)
        if actual_text is not None:
            present_cfg = parse_config_keys(actual_name or "", actual_text)
            facts["absent_config_keys"] = sorted(required_cfg - present_cfg)
            used.append(actual_name or "config.json")
        else:
            facts["absent_config_keys"] = None
            notes.append("Нет фактического config-файла: отсутствие ключей неизвестно.")
    else:
        facts["required_config_keys"] = None
        facts["absent_config_keys"] = None

    names = {Path(k).name.lower() for k in files}
    tools = set()
    if "pyproject.toml" in names or "setup.py" in names or "setup.cfg" in names:
        tools.add("setuptools")
    if "poetry.lock" in names or "poetry" in (py_text or ""):
        tools.add("poetry")
    if "Pipfile" in names:
        tools.add("pipenv")
    if "requirements.txt" in names:
        tools.add("pip")
    if "environment.yml" in names or "environment.yaml" in names:
        tools.add("conda")
    if "package.json" in names:
        tools.add("npm")
    if "pom.xml" in names:
        tools.add("maven")
    if "build.gradle" in names:
        tools.add("gradle")
    if "Makefile" in names:
        tools.add("make")
    facts["build_tools"] = sorted(tools) if tools else []

    runtimes = set()
    if any(n in names for n in ("pyproject.toml", "requirements.txt", "setup.py", "poetry.lock",
                                "Pipfile", "environment.yml", ".python-version")):
        runtimes.add("python")
    if "package.json" in names:
        runtimes.add("node")
    if "go.mod" in names:
        runtimes.add("go")
    if "pom.xml" in names or "build.gradle" in names:
        runtimes.add("java")
    facts["runtimes"] = sorted(runtimes) if runtimes else []

    project = {"id": project_id, "labels": labels, "facts": facts}
    return ParsedProject(project=project, files_used=used, notes=notes)


def load_project_dir(path: str | Path, project_id: str | None = None,
                     labels: list[str] | None = None) -> ParsedProject:
    root = Path(path)
    files: dict[str, str] = {}
    for name in ("requirements.txt", "pyproject.toml", "poetry.lock", "Pipfile.lock",
                 "requirements.lock", "uv.lock", ".env", ".env.example", ".env.sample",
                 "config.example.json", "config.example.toml", "config.json", "config.toml",
                 "setup.py", ".python-version", "environment.yml", "package.json",
                 "pom.xml", "build.gradle", "Makefile", "Pipfile", "go.mod"):
        p = root / name
        if p.is_file():
            try:
                files[name] = p.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                continue
    return project_from_files(files, project_id or root.name or "project", labels)
