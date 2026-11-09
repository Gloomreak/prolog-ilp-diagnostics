import numpy as np

from core.dataset import ERROR_CLASSES
from core.evaluation import (
    _bootstrap_ci,
    _mcnemar,
    add_label_noise,
    multilabel_metrics,
    repeated_stratified_folds,
    rule_recovery_metrics,
    run_decision_tree,
    split_projects,
)
from core.feature_encoder import FactEncoder, label_matrix
from core.generator import generate_projects


def test_metrics_known_values():
    Y = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
    P = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
    m = multilabel_metrics(Y, P)
    c0, c1 = m["per_class"][ERROR_CLASSES[0]], m["per_class"][ERROR_CLASSES[1]]
    assert (c0["tp"], c0["fp"], c0["fn"]) == (2, 0, 0) and c0["f1"] == 1.0
    assert (c1["tp"], c1["fp"], c1["fn"]) == (1, 1, 1)
    assert abs(c1["precision"] - 0.5) < 1e-9 and abs(c1["recall"] - 0.5) < 1e-9
    # классы 3 и 4 пусты: безопасное деление, F1 = 0
    assert m["per_class"][ERROR_CLASSES[2]]["f1"] == 0.0
    assert abs(m["micro_f1"] - 2 * 3 / (2 * 3 + 1 + 1)) < 1e-9
    assert abs(m["macro_f1"] - (1.0 + 0.5 + 0 + 0) / 4) < 1e-9
    assert m["exact_match"] == 0.5          # строки 1 и 4 совпали полностью


def test_split_is_reproducible_disjoint_and_complete():
    ps = generate_projects(300, 1)
    tr1, te1 = split_projects(ps, 0.2, 7)
    tr2, te2 = split_projects(ps, 0.2, 7)
    assert [p["id"] for p in te1] == [p["id"] for p in te2]
    ids_tr, ids_te = {p["id"] for p in tr1}, {p["id"] for p in te1}
    assert not ids_tr & ids_te and len(ids_tr | ids_te) == 300
    assert len(te1) == 60
    assert [p["id"] for p in split_projects(ps, 0.2, 8)[1]] != [p["id"] for p in te1]


def test_split_small_dataset_does_not_crash():
    ps = generate_projects(12, 1)
    tr, te = split_projects(ps, 0.25, 1)
    assert len(tr) + len(te) == 12


def test_repeated_stratified_folds_are_reproducible_complete_and_balanced():
    projects = generate_projects(300, 11)
    folds_a = repeated_stratified_folds(projects, n_splits=5, n_repeats=2, seed=7)
    folds_b = repeated_stratified_folds(projects, n_splits=5, n_repeats=2, seed=7)
    assert [(r, f, [p["id"] for p in te]) for r, f, _, te in folds_a] == [
        (r, f, [p["id"] for p in te]) for r, f, _, te in folds_b]
    assert len(folds_a) == 10
    for repeat in range(2):
        test_sets = [set(p["id"] for p in te) for r, _, _, te in folds_a if r == repeat]
        assert not any(test_sets[i] & test_sets[j] for i in range(5) for j in range(i + 1, 5))
        assert set.union(*test_sets) == {p["id"] for p in projects}
        sizes = [len(te) for r, _, _, te in folds_a if r == repeat]
        assert max(sizes) - min(sizes) <= 1


def test_bootstrap_intervals_and_exact_mcnemar():
    y = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=np.int8)
    pred = np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.int8)
    ci = _bootstrap_ci(y, pred, seed=3, n_resamples=100)
    assert all(0 <= b["low"] <= b["high"] <= 1 for b in ci.values())
    assert _bootstrap_ci(y, pred, seed=3, n_resamples=100) == ci
    result = _mcnemar(np.array([1, 1, 0, 0], dtype=bool),
                      np.array([0, 1, 1, 0], dtype=bool))
    assert (result["a_only_correct"], result["b_only_correct"]) == (1, 1)
    assert result["p_value"] == 1


def test_rule_recovery_uses_exact_generator_bodies():
    truth = [{"target": "missing_dependency", "body": [
        {"predicate": "requires_package", "var": 1},
        {"predicate": "package_absent", "var": 1},
        {"predicate": "uses_build_tool", "const": "poetry"},
    ]}]
    learned = [{"target": "missing_dependency", "body": list(reversed(truth[0]["body"]))}]
    metrics = rule_recovery_metrics(learned, truth)
    assert metrics["recovered_rules"] == 1 and metrics["precision"] == metrics["recall"] == 1


def test_noise_is_reproducible_and_changes_about_rate_of_labels():
    ps = generate_projects(400, 2)
    n1 = add_label_noise(ps, 0.10, 3)
    n2 = add_label_noise(ps, 0.10, 3)
    assert n1 == n2 and n1 != ps
    flips = sum(len(set(a["labels"]) ^ set(b["labels"])) for a, b in zip(ps, n1))
    assert 0.05 * 400 * 4 < flips < 0.15 * 400 * 4
    assert add_label_noise(ps, 0.0, 3) is ps
    assert ps[0]["labels"] == generate_projects(400, 2)[0]["labels"]   # исходные данные не изменены


def test_encoder_fits_on_train_only_and_ignores_unseen_values():
    train = [{"id": "a", "labels": [], "facts": {"required_packages": ["numpy"], "absent_packages": None}},
             {"id": "b", "labels": [], "facts": {"required_packages": ["pandas"]}}]
    # поля, которых нет в facts, трактуются как «неизвестно»
    for p in train:
        for f in ("absent_packages", "version_mismatches", "required_config_keys", "absent_config_keys",
                  "required_env_variables", "absent_env_variables", "build_tools", "runtimes"):
            p["facts"].setdefault(f, None)
    enc = FactEncoder().fit(train)
    assert "required_packages=numpy" in enc.feature_names_
    test = [{"id": "c", "labels": [], "facts": {**train[0]["facts"], "required_packages": ["numpy", "NEW"]}}]
    X = enc.transform(test)
    assert X.shape == (1, len(enc.feature_names_)) and X.sum() == 1      # NEW проигнорирован


def test_join_features_count_intersections():
    f = {k: [] for k in ("required_packages", "absent_packages", "version_mismatches",
                         "required_config_keys", "absent_config_keys", "required_env_variables",
                         "absent_env_variables", "build_tools", "runtimes")}
    f.update(required_packages=["a", "b"], absent_packages=["b", "z"])
    p = {"id": "x", "labels": [], "facts": f}
    enc = FactEncoder(include_join=True).fit([p])
    X = enc.transform([p])
    names = enc.feature_names_
    assert X[0, names.index("join:required_packages&absent_packages")] == 1


def test_decision_tree_baseline_runs_and_is_reproducible():
    ps = generate_projects(200, 3)
    tr, te = split_projects(ps, 0.2, 1)
    r1 = run_decision_tree(tr, te, 1)
    r2 = run_decision_tree(tr, te, 1)
    assert r1["Y_pred"].shape == (len(te), 4) and (r1["Y_pred"] == r2["Y_pred"]).all()
    rj = run_decision_tree(tr, te, 1, include_join=True)
    m = multilabel_metrics(label_matrix(te), rj["Y_pred"])
    assert m["macro_f1"] > 0.9          # агрегаты «требуется ∩ отсутствует» решают задачу почти целиком
