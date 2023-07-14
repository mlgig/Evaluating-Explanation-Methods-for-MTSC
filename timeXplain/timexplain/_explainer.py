from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, Optional, Sequence, List

import numpy as np
from scipy import sparse

from timexplain._explanation import Explanation, TabularExplanation
from timexplain._utils import classname, is_iterable

E = TypeVar("E", bound=Explanation)


class Explainer(Generic[E], ABC):

    def explain(self, X_specimens) -> Union[E, List[E]]:
        if classname(X_specimens) != "pandas.core.frame.DataFrame" and not is_iterable(X_specimens[0]):
            if isinstance(X_specimens, np.ndarray):
                X_specimens = X_specimens[np.newaxis]
            else:
                X_specimens = [X_specimens]
            return self._explain(X_specimens)[0]
        else:
            return self._explain(X_specimens)

    @abstractmethod
    def _explain(self, X_specimens) -> List[E]:
        raise NotImplementedError

    def __rshift__(self, child: Explainer[TabularExplanation]) -> SuperposExplainer[E]:
        return SuperposExplainer(self, child, divide_spread=False)

    def __ge__(self, child: Explainer[TabularExplanation]) -> SuperposExplainer[E]:
        return SuperposExplainer(self, child, divide_spread=True)


class SuperposExplainer(Generic[E], Explainer[E]):
    parent_explainer: Explainer[E]
    child_explainer: Explainer[TabularExplanation]
    divide_spread: bool

    def __init__(self, parent_explainer: Explainer[E], child_explainer: Explainer[TabularExplanation],
                 *, divide_spread=False):
        self.parent_explainer = parent_explainer
        self.child_explainer = child_explainer
        self.divide_spread = divide_spread

    def _explain(self, X_specimens):
        parent_expls = self.parent_explainer.explain(X_specimens)

        # TODO: Support sktime (i.e., DataFrame) estimator inputs?
        child_X_specimens = [expl.y_pred for expl in parent_expls]
        if all(isinstance(arr, np.ndarray) for arr in child_X_specimens):
            child_X_specimens = np.array(child_X_specimens)
        elif all(isinstance(arr, sparse.spmatrix) for arr in child_X_specimens):
            child_X_specimens = sparse.vstack(child_X_specimens)
        else:
            raise ValueError(f"Intermediate model output not supported by superposition: {type(child_X_specimens[0])}")

        child_expls = self.child_explainer.explain(child_X_specimens)

        # If requested, normalize the rows of the parent impact matrix.
        # This leads to each superimposed time series from the parent contributing exactly as much as the
        # corresponding child impact indicates.
        if self.divide_spread:
            from sklearn.preprocessing import normalize
            parent_expls = [parent_expl.map(lambda parent_impacts: normalize(parent_impacts, norm="l1", copy=False))
                            for parent_expl in parent_expls]

        # Note that the matrix multiplication produces a dense array if either the parent or the child impacts
        # are dense. It produces a sparse matrix if both impacts are sparse.
        return [parent_expl.map(lambda parent_impacts: child_expl.impacts @ parent_impacts)
                for child_expl, parent_expl in zip(child_expls, parent_expls)]


class MeanExplainer(Generic[E], Explainer[E]):
    base_explainers: Sequence[Explainer[E]]
    weights: Optional[Sequence[float]]

    def __init__(self, base_explainers: Sequence[Explainer[E]], *, weights: Sequence[float] = None):
        self.base_explainers = list(base_explainers)
        self.weights = weights

    def _explain(self, X_specimens):
        all_base_expls = []
        for base_explainer in self.base_explainers:
            all_base_expls.append(base_explainer.explain(X_specimens))
        all_base_expls = np.transpose(all_base_expls)

        # Note that the following produces a dense array if all base impacts are dense. It produces a
        # sparse matrix if all base impacts are sparse. Mixing dense and sparse leads to an error.
        weights = 1 if self.weights is None else np.asarray(self.weights).reshape((-1, 1, 1))
        return [Explanation.reduce(lambda arrs: np.sum(weights * arrs, axis=0) / np.sum(weights),
                                   specimen_expls)
                for specimen_expls in all_base_expls]
