"""Кодирование фактов проекта в числовые признаки для базового классификатора.

Те же самые факты, что получает ILP, но в «плоском» виде: по одному бинарному
признаку на каждое значение (например, «пакет numpy требуется»,
«пакет numpy отсутствует»). Словарь значений строится ТОЛЬКО по обучающей выборке;
значения, не встречавшиеся при обучении, при преобразовании игнорируются.

Режим include_join: добавляет 4 агрегата, вычисляемых из фактов без участия меток
(сколько элементов одновременно «требуется» и «отсутствует» и т. п.). Они вручную
вносят реляционную связь в плоские признаки, поэтому этот вариант показывается
отдельно как расширенный базовый, а не как основной.
"""
from __future__ import annotations

import numpy as np

from .dataset import ERROR_CLASSES, FACT_FIELDS

JOIN_PAIRS = [
    ("required_packages", "absent_packages"),
    ("required_packages", "version_mismatches"),
    ("required_config_keys", "absent_config_keys"),
    ("required_env_variables", "absent_env_variables"),
]


class FactEncoder:
    def __init__(self, include_join: bool = False):
        self.include_join = include_join
        self.feature_names_: list[str] = []
        self._index: dict[tuple[str, str], int] = {}
        self._fitted = False

    def fit(self, projects: list[dict]) -> FactEncoder:
        vocab: set[tuple[str, str]] = set()
        for p in projects:
            for fld in FACT_FIELDS:
                for v in (p["facts"].get(fld) or []):
                    vocab.add((fld, v))
        ordered = sorted(vocab)
        self._index = {k: i for i, k in enumerate(ordered)}
        self.feature_names_ = [f"{f}={v}" for f, v in ordered]
        if self.include_join:
            self.feature_names_ += [f"join:{a}&{b}" for a, b in JOIN_PAIRS]
        self._fitted = True
        return self

    def transform(self, projects: list[dict]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FactEncoder не обучен")
        n_flat = len(self._index)
        X = np.zeros((len(projects), len(self.feature_names_)), dtype=np.int16)
        for r, p in enumerate(projects):
            for fld in FACT_FIELDS:
                for v in (p["facts"].get(fld) or []):
                    j = self._index.get((fld, v))
                    if j is not None:
                        X[r, j] = 1
            if self.include_join:
                for k, (a, b) in enumerate(JOIN_PAIRS):
                    X[r, n_flat + k] = len(set(p["facts"].get(a) or []) & set(p["facts"].get(b) or []))
        return X

    def fit_transform(self, projects: list[dict]) -> np.ndarray:
        return self.fit(projects).transform(projects)


def label_matrix(projects: list[dict], classes: list[str] | None = None) -> np.ndarray:
    classes = classes or ERROR_CLASSES
    Y = np.zeros((len(projects), len(classes)), dtype=np.int8)
    for r, p in enumerate(projects):
        for j, c in enumerate(classes):
            if p["labels"] and c in p["labels"]:
                Y[r, j] = 1
    return Y
