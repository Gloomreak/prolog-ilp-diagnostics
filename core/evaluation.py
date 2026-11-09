"""Эксперимент: единое разбиение train/test, ILP (Prolog) против решающего дерева.

Протокол (ТЗ, раздел 9.2):
  1. фиксированный seed -> разбиение train/test (по умолчанию 80/20);
  2. ILP обучается только на train (Prolog видит только обучающие проекты);
  3. кодировщик признаков и дерево обучаются только на train и тех же проектах;
  4. оба метода предсказывают одну и ту же test-выборку;
  5. метрики по классам, macro/micro, exact-match; время обучения; сложность.
"""
from __future__ import annotations

import platform
import random
import sys
import time
from collections import Counter
from math import comb

import numpy as np

from .dataset import ERROR_CLASSES
from .diagnosis import parse_diagnosis, predicted_labels
from .feature_encoder import FactEncoder, label_matrix
from .prolog_runner import PrologEngine


def rule_recovery_metrics(rules: list[dict], true_rules: list[dict]) -> dict:
    """Exact body recovery for synthetic datasets that publish generator ground truth."""
    def body_key(body: list[dict]) -> tuple:
        literals = []
        for lit in body:
            if lit.get("neq"):
                literals.append(("neq", lit.get("type"), lit.get("left"), lit.get("right")))
            else:
                literals.append((lit.get("predicate"), lit.get("var"), lit.get("const"),
                                 bool(lit.get("negated"))))
        return tuple(sorted(literals, key=repr))

    learned = {(rule["target"], body_key(rule.get("body", []))) for rule in rules}
    gold = {(rule["target"], body_key(rule.get("body", []))) for rule in true_rules}
    matched = learned & gold
    return {
        "true_rules": len(gold),
        "learned_rules": len(learned),
        "recovered_rules": len(matched),
        "precision": len(matched) / len(learned) if learned else 0.0,
        "recall": len(matched) / len(gold) if gold else 0.0,
        "exact_matches": [{"target": target, "body": list(body)}
                          for target, body in sorted(matched, key=repr)],
    }


# --------------------------------------------------------------------- разбиение
def split_projects(projects: list[dict], test_size: float = 0.2, seed: int = 42
                   ) -> tuple[list[dict], list[dict]]:
    """Воспроизводимое разбиение. Стратификация по сочетанию меток, если в каждой
    группе хотя бы 2 проекта; иначе — обычное случайное разбиение."""
    from sklearn.model_selection import train_test_split
    idx = list(range(len(projects)))
    combo = ["|".join(sorted(p["labels"] or [])) for p in projects]
    counts = Counter(combo)
    strat = combo if all(c >= 2 for c in counts.values()) and len(counts) > 1 else None
    if strat is not None:
        n_test = int(round(len(projects) * test_size))
        if n_test < len(counts) or len(projects) - n_test < len(counts):
            strat = None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True,
                              stratify=strat)
    return [projects[i] for i in sorted(tr)], [projects[i] for i in sorted(te)]


def repeated_stratified_folds(projects: list[dict], n_splits: int = 5,
                              n_repeats: int = 3, seed: int = 42
                              ) -> list[tuple[int, int, list[dict], list[dict]]]:
    """Многометочная repeated stratified k-fold с балансировкой каждого класса."""
    if n_splits < 2 or n_splits > len(projects):
        raise ValueError("n_splits должен быть между 2 и числом проектов.")
    if n_repeats < 1:
        raise ValueError("n_repeats должен быть положительным.")
    Y = label_matrix(projects).astype(bool)
    class_totals = Y.sum(axis=0).astype(float)
    desired = class_totals / n_splits
    folds: list[tuple[int, int, list[dict], list[dict]]] = []
    indices = np.arange(len(projects))
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        tie_break = rng.random(len(projects))
        rarity = np.divide(1.0, class_totals, out=np.zeros_like(class_totals),
                           where=class_totals > 0)
        priority = Y @ rarity
        order = sorted(indices, key=lambda i: (-priority[i], tie_break[i]))
        assignments = np.full(len(projects), -1, dtype=int)
        fold_counts = np.zeros((n_splits, Y.shape[1]), dtype=float)
        fold_sizes = np.zeros(n_splits, dtype=int)
        for i in order:
            active = Y[i]
            costs = []
            for fold in range(n_splits):
                label_cost = float(np.sum(
                    fold_counts[fold][active] / np.maximum(desired[active], 1)))
                size_cost = fold_sizes[fold] / (len(projects) / n_splits)
                costs.append((label_cost + size_cost * 0.25, fold_sizes[fold], fold))
            chosen = min(costs)[2]
            assignments[i] = chosen
            fold_sizes[chosen] += 1
            fold_counts[chosen] += Y[i]
        for fold in range(n_splits):
            test_idx = indices[assignments == fold]
            train_idx = indices[assignments != fold]
            folds.append((repeat, fold,
                          [projects[i] for i in train_idx],
                          [projects[i] for i in test_idx]))
    return folds


def _bootstrap_ci(Y_true: np.ndarray, Y_pred: np.ndarray, seed: int = 42,
                  n_resamples: int = 1000) -> dict:
    if n_resamples < 1:
        raise ValueError("n_resamples должен быть положительным.")
    if len(Y_true) == 0:
        return {}
    rng = np.random.default_rng(seed)
    values = {"macro_f1": [], "micro_f1": [], "exact_match": []}
    for _ in range(n_resamples):
        sample = rng.integers(0, len(Y_true), size=len(Y_true))
        metrics = multilabel_metrics(Y_true[sample], Y_pred[sample])
        for key in values:
            values[key].append(metrics[key])
    return {key: {"low": float(np.quantile(scores, 0.025)),
                  "high": float(np.quantile(scores, 0.975))}
            for key, scores in values.items()}


def _mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Точный двусторонний тест Мак-Немара для парных бинарных исходов."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p_value)}


def _paired_comparisons(Y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    names = list(predictions)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            correct_a = np.all(predictions[a] == Y_true, axis=1)
            correct_b = np.all(predictions[b] == Y_true, axis=1)
            exact = _mcnemar(correct_a, correct_b)
            exact["per_class"] = {
                cls: _mcnemar(predictions[a][:, j] == Y_true[:, j],
                              predictions[b][:, j] == Y_true[:, j])
                for j, cls in enumerate(ERROR_CLASSES)
            }
            out[f"{a} vs {b}"] = exact
    return out


def add_label_noise(projects: list[dict], rate: float, seed: int) -> list[dict]:
    """Сценарий шума меток: каждая пара (проект, класс) с вероятностью rate меняет
    значение на противоположное. Воспроизводимо по seed. Применяется только к train."""
    if rate <= 0:
        return projects
    rng = random.Random(seed * 1000003 + 17)
    noisy = []
    for p in projects:
        labels = set(p["labels"] or [])
        for c in ERROR_CLASSES:
            if rng.random() < rate:
                labels.symmetric_difference_update({c})
        noisy.append({**p, "labels": [c for c in ERROR_CLASSES if c in labels]})
    return noisy


# --------------------------------------------------------------------- метрики
def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return p, r, _div(2 * p * r, p + r)


def multilabel_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                       classes: list[str] | None = None) -> dict:
    classes = classes or (ERROR_CLASSES if Y_true.shape[1] == len(ERROR_CLASSES)
                          else [str(i) for i in range(Y_true.shape[1])])
    per_class = {}
    TP = FP = FN = 0
    for j, c in enumerate(classes):
        t, p = Y_true[:, j].astype(bool), Y_pred[:, j].astype(bool)
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        pr, rc, f1 = _prf(tp, fp, fn)
        per_class[c] = {"precision": pr, "recall": rc, "f1": f1, "support": int(t.sum()),
                        "tp": tp, "fp": fp, "fn": fn}
        TP += tp
        FP += fp
        FN += fn
    mp, mr, mf = _prf(TP, FP, FN)
    return {
        "per_class": per_class,
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "exact_match": float((Y_true == Y_pred).all(axis=1).mean()) if len(Y_true) else 0.0,
        "n_samples": int(len(Y_true)),
    }


# --------------------------------------------------------------------- базовый метод
def run_decision_tree(train: list[dict], test: list[dict], seed: int,
                      include_join: bool = False) -> dict:
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.tree import DecisionTreeClassifier
    t0 = time.perf_counter()
    enc = FactEncoder(include_join=include_join)
    X_tr = enc.fit_transform(train)
    Y_tr = label_matrix(train)
    clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=seed))
    clf.fit(X_tr, Y_tr)
    elapsed = time.perf_counter() - t0
    X_te = enc.transform(test)
    Y_pred = np.asarray(clf.predict(X_te), dtype=np.int8)
    trees = clf.estimators_
    return {
        "Y_pred": Y_pred,
        "train_seconds": elapsed,
        "n_features": int(X_tr.shape[1]),
        "n_leaves": int(sum(t.get_n_leaves() for t in trees)),
        "n_nodes": int(sum(t.tree_.node_count for t in trees)),
        "max_depth": int(max(t.get_depth() for t in trees)),
    }


# --------------------------------------------------------------------- ILP
def rule_complexity(rules: list[dict]) -> dict:
    lengths = [len(r["body"]) for r in rules]
    return {"n_rules": len(rules),
            "avg_body_length": float(np.mean(lengths)) if lengths else 0.0,
            "total_literals": int(sum(lengths))}


def run_ilp(engine: PrologEngine, train: list[dict], test: list[dict], params: dict) -> dict:
    t0 = time.perf_counter()
    train_resp = engine.train(train, params)
    wall = time.perf_counter() - t0
    rules = train_resp["model"]["rules"]
    diag = engine.diagnose(rules, test, max_proofs=0)
    parsed = parse_diagnosis(diag, rules)
    by_id = {p["id"]: p for p in parsed}
    ordered = [by_id[p["id"]] for p in test]
    pred = predicted_labels(ordered)
    Y_pred = np.zeros((len(test), len(ERROR_CLASSES)), dtype=np.int8)
    for i, labs in enumerate(pred):
        for c in labs:
            Y_pred[i, ERROR_CLASSES.index(c)] = 1
    stats = train_resp["stats"]
    return {"Y_pred": Y_pred, "train_seconds_wall": wall,
            "train_seconds_prolog": stats["elapsed_seconds"], "rules": rules,
            "stats": stats, "train_response": train_resp,
            "complexity": {**rule_complexity(rules),
                           "train_covered_share": stats.get("train_covered_share", 0.0)}}


def environment_info(engine: PrologEngine | None = None) -> dict:
    import pandas
    import sklearn
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__}
    try:
        import streamlit
        info["streamlit"] = streamlit.__version__
    except Exception:   # noqa: BLE001
        pass
    if engine is not None:
        try:
            info["swi-prolog"] = engine.info().version_text
        except Exception as exc:    # noqa: BLE001
            info["swi-prolog"] = f"недоступен: {exc}"
    return info


def run_experiment(engine: PrologEngine, projects: list[dict], params: dict, seed: int = 42,
                   test_size: float = 0.2, noise: float = 0.0,
                   with_join_tree: bool = True,
                   split: tuple[list[dict], list[dict]] | None = None,
                   true_rules: list[dict] | None = None) -> dict:
    """Один запуск сравнения на одном разбиении. Возвращает сериализуемый документ
    (без массивов numpy), пригодный для экспорта."""
    labeled = [p for p in projects if p["labels"] is not None]
    if len(labeled) < 10:
        raise ValueError("Для эксперимента нужно не менее 10 размеченных проектов.")
    train, test = split if split is not None else split_projects(labeled, test_size, seed)
    test_size = len(test) / (len(train) + len(test))
    train_used = add_label_noise(train, noise, seed)     # шум — только в train
    Y_test = label_matrix(test)

    ilp = run_ilp(engine, train_used, test, params)
    dt = run_decision_tree(train_used, test, seed, include_join=False)
    methods = {
        "ILP (Prolog)": {
            "metrics": multilabel_metrics(Y_test, ilp["Y_pred"]),
            "train_seconds": ilp["train_seconds_wall"],
            "train_seconds_prolog": ilp["train_seconds_prolog"],
            "complexity": ilp["complexity"],
        },
        "Decision Tree": {
            "metrics": multilabel_metrics(Y_test, dt["Y_pred"]),
            "train_seconds": dt["train_seconds"],
            "complexity": {k: dt[k] for k in ("n_features", "n_leaves", "n_nodes", "max_depth")},
        },
    }
    if with_join_tree:
        dtj = run_decision_tree(train_used, test, seed, include_join=True)
        methods["Decision Tree + агрегаты"] = {
            "metrics": multilabel_metrics(Y_test, dtj["Y_pred"]),
            "train_seconds": dtj["train_seconds"],
            "complexity": {k: dtj[k] for k in ("n_features", "n_leaves", "n_nodes", "max_depth")},
        }
    predictions = {
        "ILP (Prolog)": ilp["Y_pred"],
        "Decision Tree": dt["Y_pred"],
    }
    if with_join_tree:
        predictions["Decision Tree + агрегаты"] = dtj["Y_pred"]
    for i, (name, prediction) in enumerate(predictions.items()):
        methods[name]["bootstrap_ci"] = _bootstrap_ci(Y_test, prediction, seed + i)
    result = {
        "meta": {"seed": seed, "test_size": test_size, "label_noise": noise,
                 "n_train": len(train), "n_test": len(test), "ilp_params": params,
                 "train_class_counts": {c: sum(1 for p in train_used if c in p["labels"])
                                        for c in ERROR_CLASSES},
                 "test_class_counts": {c: sum(1 for p in test if c in p["labels"])
                                       for c in ERROR_CLASSES},
                 "environment": environment_info(engine)},
        "methods": methods,
        "paired_tests": _paired_comparisons(Y_test, predictions),
        "ilp_rules": ilp["rules"],
        "ilp_stats": ilp["stats"],
        "evaluation": {
            "y_true": Y_test.astype(int).tolist(),
            "predictions": {name: pred.astype(int).tolist()
                            for name, pred in predictions.items()},
        },
    }
    if true_rules is not None:
        result["rule_recovery"] = rule_recovery_metrics(ilp["rules"], true_rules)
    return result


def run_multi_seed(engine: PrologEngine, projects: list[dict], params: dict, seeds: list[int],
                   test_size: float = 0.2, noise: float = 0.0, with_join_tree: bool = True,
                   progress=None, true_rules: list[dict] | None = None) -> dict:
    runs = []
    for i, s in enumerate(seeds):
        runs.append(run_experiment(engine, projects, params, s, test_size, noise, with_join_tree,
                                   true_rules=true_rules))
        if progress:
            progress(i + 1, len(seeds))
    names = list(runs[0]["methods"])
    agg = {}
    for m in names:
        vals = lambda key: [r["methods"][m]["metrics"][key] for r in runs]   # noqa: E731
        agg[m] = {k: {"mean": float(np.mean(vals(k))), "std": float(np.std(vals(k)))}
                  for k in ("macro_f1", "micro_f1", "exact_match")}
        ts = [r["methods"][m]["train_seconds"] for r in runs]
        agg[m]["train_seconds"] = {"mean": float(np.mean(ts)), "std": float(np.std(ts))}
    return {"seeds": seeds, "runs": runs, "aggregate": agg}


def run_repeated_stratified_cv(engine: PrologEngine, projects: list[dict], params: dict,
                               n_splits: int = 5, n_repeats: int = 3, seed: int = 42,
                               noise: float = 0.0, with_join_tree: bool = True,
                               progress=None, true_rules: list[dict] | None = None) -> dict:
    """Сравнение на каждом фолде repeated stratified k-fold."""
    labeled = [p for p in projects if p.get("labels") is not None]
    if len(labeled) < n_splits:
        raise ValueError(f"Для {n_splits}-fold CV нужно не менее {n_splits} размеченных проектов.")
    folds = repeated_stratified_folds(labeled, n_splits, n_repeats, seed)
    runs = []
    for index, (repeat, fold, train, test) in enumerate(folds):
        run = run_experiment(engine, labeled, params, seed + repeat * n_splits + fold,
                             noise=noise, with_join_tree=with_join_tree,
                             split=(train, test), true_rules=true_rules)
        run["meta"]["repeat"] = repeat + 1
        run["meta"]["fold"] = fold + 1
        runs.append(run)
        if progress:
            progress(index + 1, len(folds))
    methods = list(runs[0]["methods"])
    aggregate = {}
    for name in methods:
        aggregate[name] = {}
        for metric in ("macro_f1", "micro_f1", "exact_match"):
            scores = [run["methods"][name]["metrics"][metric] for run in runs]
            rng = np.random.default_rng(seed + len(aggregate) * 31 + len(aggregate[name]))
            bootstrap_means = np.mean(
                rng.choice(scores, size=(1000, len(scores)), replace=True), axis=1)
            aggregate[name][metric] = {
                "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                "bootstrap_ci": {
                    "low": float(np.quantile(bootstrap_means, 0.025)),
                    "high": float(np.quantile(bootstrap_means, 0.975)),
                },
            }
    return {"protocol": "repeated_stratified_k_fold", "n_splits": n_splits,
            "n_repeats": n_repeats, "seed": seed, "noise": noise,
            "runs": runs, "aggregate": aggregate}


def run_ablation(engine: PrologEngine, projects: list[dict], params: dict,
                 parameter: str, values: list[float | int], seed: int = 42,
                 test_size: float = 0.2, noise: float = 0.0,
                 with_join_tree: bool = True, progress=None,
                 true_rules: list[dict] | None = None) -> dict:
    """Однофакторная абляция на одном фиксированном разбиении."""
    supported = {"max_body_length", "min_support", "min_precision"}
    if parameter not in supported:
        raise ValueError(f"Параметр абляции должен быть одним из: {', '.join(sorted(supported))}.")
    if not values:
        raise ValueError("Для абляции нужно указать хотя бы одно значение.")
    labeled = [p for p in projects if p.get("labels") is not None]
    train, test = split_projects(labeled, test_size, seed)
    runs = []
    for index, value in enumerate(values):
        current = {**params, parameter: value}
        run = run_experiment(engine, labeled, current, seed, noise=noise,
                             with_join_tree=with_join_tree, split=(train, test),
                             true_rules=true_rules)
        runs.append({"parameter": parameter, "value": value, "result": run})
        if progress:
            progress(index + 1, len(values))
    return {"protocol": "single_parameter_ablation", "parameter": parameter,
            "values": values, "seed": seed, "test_size": test_size, "noise": noise,
            "runs": runs}


def run_learning_curve(engine: PrologEngine, projects: list[dict], params: dict,
                       fractions: list[float] | None = None, seed: int = 42,
                       test_size: float = 0.2, noise: float = 0.0,
                       with_join_tree: bool = True, progress=None,
                       true_rules: list[dict] | None = None) -> dict:
    """Learning curve при фиксированной тестовой выборке."""
    fractions = fractions or [0.2, 0.4, 0.6, 0.8, 1.0]
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("Доли обучающих данных должны быть в интервале (0, 1].")
    labeled = [p for p in projects if p.get("labels") is not None]
    train, test = split_projects(labeled, test_size, seed)
    runs = []
    for index, fraction in enumerate(fractions):
        if fraction == 1:
            subset = train
        else:
            _, subset = split_projects(train, 1.0 - fraction, seed + index + 1)
            if not subset:
                subset = [train[0]]
        result = run_experiment(engine, labeled, params, seed, noise=noise,
                                with_join_tree=with_join_tree, split=(subset, test),
                                true_rules=true_rules)
        runs.append({"fraction": fraction, "n_train": len(subset), "result": result})
        if progress:
            progress(index + 1, len(fractions))
    return {"protocol": "learning_curve", "seed": seed, "test_size": test_size,
            "test_ids": [p["id"] for p in test], "runs": runs}
