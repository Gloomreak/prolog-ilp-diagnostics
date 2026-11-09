"""Создаёт примеры экспорта в папке examples/: модель, правила .pl, диагностику
и результаты эксперимента. Требует установленный SWI-Prolog.

Запуск из корня проекта:
    python scripts/make_examples.py [--swipl "C:\\Program Files\\swipl\\bin\\swipl.exe"]
Это же — быстрая проверка всего сквозного сценария без браузера.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import exports  # noqa: E402
from core.dataset import labeled_only, load_dataset_file, validate_project  # noqa: E402
from core.diagnosis import parse_diagnosis, unknown_field_warnings  # noqa: E402
from core.evaluation import environment_info, run_multi_seed, split_projects  # noqa: E402
from core.prolog_runner import PrologEngine, PrologError, check_swipl  # noqa: E402
from core.settings import DATA_DIR, DEMO_DATASET  # noqa: E402

PARAMS = {"max_body_length": 3, "min_support": 2, "min_precision": 0.8,
          "max_rules_per_class": 5, "max_candidates": 20000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swipl", default=None, help="путь к swipl (если его нет в PATH)")
    ap.add_argument("--out", default=str(ROOT / "examples"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        info = check_swipl(args.swipl)
    except PrologError as exc:
        print("ОШИБКА:", exc)
        return 1
    engine = PrologEngine(info.path)
    print("SWI-Prolog:", info.version_text)

    data = load_dataset_file(DEMO_DATASET)
    projects = labeled_only(data.projects)
    train, test = split_projects(projects, 0.2, 42)
    print(f"Набор: {len(projects)} проектов, train={len(train)}, test={len(test)}")

    resp = engine.train(train, PARAMS)
    doc = exports.build_model_document(
        resp, PARAMS, {"source": "data/demo_dataset.json", "n_train": len(train), "n_test": len(test),
                       "seed": 42, "test_size": 0.2, "test_ids": [p["id"] for p in test]},
        environment_info(engine))
    (out / "model.json").write_text(exports.model_to_json(doc), encoding="utf-8")
    (out / "learned_rules.pl").write_text(exports.rules_to_prolog(doc), encoding="utf-8")
    rules = resp["model"]["rules"]
    print(f"Выучено правил: {len(rules)}; время обучения в Prolog: {resp['stats']['elapsed_seconds']} с")
    for r in rules:
        print(f"  [{r['target']}] TP={r['tp']} FP={r['fp']} FN={r['fn']}  P={r['precision']} R={r['recall']}")

    obj = json.loads((DATA_DIR / "example_project_missing_dependency.json").read_text(encoding="utf-8"))
    project, _ = validate_project(obj, 0, require_labels=False)
    parsed = parse_diagnosis(engine.diagnose(rules, [project], max_proofs=3), rules)
    (out / "diagnosis.json").write_text(
        exports.diagnosis_to_json(parsed, {"warnings": unknown_field_warnings(project, rules)}),
        encoding="utf-8")
    (out / "diagnosis.csv").write_text(exports.diagnosis_to_csv(parsed), encoding="utf-8")
    print("Диагностика:", {c: v["status"] for c, v in parsed[0]["classes"].items()})

    multi = run_multi_seed(engine, projects, PARAMS, [42, 43, 44])
    (out / "experiment.json").write_text(exports.experiment_to_json(multi), encoding="utf-8")
    (out / "comparison.csv").write_text(exports.comparison_to_csv(multi["runs"][0]), encoding="utf-8")
    for name, a in multi["aggregate"].items():
        print(f"  {name:28s} macro-F1 {a['macro_f1']['mean']:.3f}±{a['macro_f1']['std']:.3f}  "
              f"exact {a['exact_match']['mean']:.3f}  t={a['train_seconds']['mean']:.2f}s")
    print("Готово ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
