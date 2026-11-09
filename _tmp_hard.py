import os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def mkdtemp(suffix=None, prefix=None, dir=None):
    base = dir or tempfile.gettempdir(); prefix = prefix or "tmp"
    for _ in range(100):
        p = os.path.join(base, f"{prefix}{uuid.uuid4().hex[:10]}{suffix or ''}")
        try:
            os.makedirs(p, exist_ok=False); return p
        except FileExistsError:
            continue
    raise RuntimeError

tempfile.mkdtemp = mkdtemp

from core.dataset import load_dataset_file, labeled_only
from core.evaluation import split_projects
from core.prolog_runner import PrologEngine, check_swipl

eng = PrologEngine(check_swipl().path)
for name in ("demo_hard.json", "demo_partial.json"):
    print("==", name)
    d = load_dataset_file(ROOT / "data" / name)
    ps = labeled_only(d.projects)
    print("ok", d.ok, "n", len(ps), "unknown fields", sum(1 for p in ps if any(v is None for v in p["facts"].values())))
    train, test = split_projects(ps, 0.2, 42)
    resp = eng.train(train, {"max_body_length": 3, "min_support": 2, "min_precision": 0.8,
                             "max_rules_per_class": 5, "max_candidates": 20000,
                             "smoothing": 1.0, "mdl_penalty": 1.0, "max_constants": 20})
    print("stats", resp["stats"]["candidates_tested"], resp["stats"]["elapsed_seconds"], resp["stats"]["truncated"])
    for r in resp["model"]["rules"]:
        print(r["target"], r["tp"], r["fp"], r.get("precision_smoothed"), "|", " ".join(r["rule"].split()))
