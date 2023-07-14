from __future__ import annotations

from typing import Any, Dict

import numpy as np
import shap

from timexplain._explainer import Explainer
from timexplain._explanation import TabularExplanation


class LinearShapExplainer(Explainer[TabularExplanation]):
    model: Any
    X_bg: Any
    constr_kwargs: Dict[str, Any]

    def __init__(self, model, X_bg=None, **constr_kwargs):
        self.model = model
        self.X_bg = X_bg
        self.constr_kwargs = constr_kwargs

    def _explain(self, X_specimens):
        X_bg = _get_X_bg(self.X_bg, X_specimens)
        impacts = np.asarray(shap.LinearExplainer(self.model, X_bg, **self.constr_kwargs)
                             .shap_values(X_specimens))
        return [TabularExplanation(X_specimens[idx], impacts[..., idx, :]) for idx in range(X_specimens.shape[0])]


class TreeShapExplainer(Explainer[TabularExplanation]):
    model: Any
    X_bg: Any
    constr_kwargs: Dict[str, Any]
    shapva_kwargs: Dict[str, Any]

    def __init__(self, model, X_bg=None, **kwargs):
        self.model = model
        self.X_bg = X_bg

        shapva_keys = ["tree_limit", "approximate", "check_additivity"]
        self.constr_kwargs = {k: v for k, v in kwargs.items() if k not in shapva_keys}
        self.shapva_kwargs = {k: v for k, v in kwargs.items() if k in shapva_keys}

    def _explain(self, X_specimens):
        # Support for: https://github.com/joshloyal/RotationForest
        if type(self.model).__name__ == "RotationTreeClassifier":
            impacts = self._explain_rot_tree(self.model, X_specimens)
        elif type(self.model).__name__ == "RotationForestClassifier":
            # Because SHAP values are additive, the SHAP values of the tree ensemble are equal to the mean of
            # the SHAP values of each tree.
            impacts = np.mean([self._explain_rot_tree(rot_tree, X_specimens)
                               for rot_tree in self.model.estimators_], axis=0)
        else:
            X_bg = _get_X_bg(self.X_bg, X_specimens)
            impacts = np.asarray(shap.TreeExplainer(self.model, X_bg, **self.constr_kwargs)
                                 .shap_values(X_specimens, **self.shapva_kwargs))

        return [TabularExplanation(X_specimens[idx], impacts[..., idx, :]) for idx in range(X_specimens.shape[0])]

    def _explain_rot_tree(self, rot_tree, X_specimens):
        X_bg = _get_X_bg(self.X_bg, X_specimens)
        rot_impacts = np.asarray(shap.TreeExplainer(rot_tree, X_bg, **self.constr_kwargs)
                                 .shap_values(X_specimens, **self.shapva_kwargs))
        return rot_impacts @ rot_tree.rotation_matrix.T  # Rotate the impacts back.


def _get_X_bg(X_bg, X_specimens):
    if X_bg is None and X_specimens.shape[0] >= 20:
        return X_specimens
    else:
        return X_bg
