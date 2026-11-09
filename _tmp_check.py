import os, sys, tempfile, uuid
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

from core.dataset import load_dataset_file, labeled_only
from core.evaluation import split_projects
from core.prolog_runner import PrologEngine, check_swipl

eng = PrologEngine(check_swipl().path)
ps = labeled_only(load_dataset_file(ROOT / "data" / "demo_dataset.json").projects)
train, test = split_projects(ps, 0.2, 42)
params = {"max_body_length": 3, "min_support": 2, "min_precision": 0.8,
          "max_rules_per_class": 5, "max_candidates": 20000}
resp = eng.train(train, params)
print("stats", resp["stats"])
for r in resp["model"]["rules"]:
    print(r["target"], r["tp"], r["fp"], r["precision"], r.get("precision_smoothed"), "|", " ".join(r["rule"].split()))
    print("  body", r["body"])
print("rejected")
for r in resp["log"]["rejected"][:5]:
    print(" ", r["target"], r["tp"], r["fp"], r["reasons"], " ".join(r["rule"].split())[:120])
