import os, sys, tempfile, uuid, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def mkdtemp(suffix=None, prefix=None, dir=None):
    base = dir or tempfile.gettempdir()
    prefix = prefix or "tmp"
    for _ in range(100):
        p = os.path.join(base, f"{prefix}{uuid.uuid4().hex[:10]}{suffix or ''}")
        try:
            os.makedirs(p, exist_ok=False)
            return p
        except FileExistsError:
            continue
    raise RuntimeError


tempfile.mkdtemp = mkdtemp

from core.prolog_runner import PrologEngine, check_swipl

def proj(i, labels, facts):
    f = {k: None for k in ("required_packages", "absent_packages", "version_mismatches",
                           "required_config_keys", "absent_config_keys",
                           "required_env_variables", "absent_env_variables",
                           "build_tools", "runtimes")}
    for pred, vals in facts.items():
        f[pred] = vals
    return {"id": f"p{i}", "labels": labels, "facts": f}

projects = []
rows = [
    (["missing_dependency"], {"required_packages": ["a"], "absent_packages": ["a"], "build_tools": ["poetry"]}),
    (["missing_dependency"], {"required_packages": ["b"], "absent_packages": ["b"], "build_tools": ["poetry"]}),
    (["missing_dependency"], {"required_packages": ["c"], "absent_packages": ["c"], "build_tools": ["poetry"]}),
    ([], {"required_packages": ["a"], "absent_packages": ["a"], "build_tools": ["make"]}),
    ([], {"required_packages": ["b"], "absent_packages": ["b"], "build_tools": ["pip"]}),
    ([], {"required_packages": ["c"], "build_tools": ["poetry"]}),
    ([], {"required_packages": ["d"], "build_tools": ["poetry"]}),
]
for i, (labels, facts) in enumerate(rows, 1):
    projects.append(proj(i, labels, facts))

eng = PrologEngine(check_swipl().path)
resp = eng.train(projects, {"min_support": 2, "min_precision": 0.8, "max_body_length": 3})
for r in resp["model"]["rules"]:
    print(r["target"], r["tp"], r["fp"], "|", " ".join(r["rule"].split()))
    print(json.dumps(r["body"], ensure_ascii=False))
print("rejected:")
for r in resp["log"]["rejected"]:
    print(r["target"], r["tp"], r["fp"], r["reasons"], " ".join(r["rule"].split()))
print("stats", resp["stats"])
