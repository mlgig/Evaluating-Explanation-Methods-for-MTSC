from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from timexplain._explainer import Explainer
from timexplain._explanation import TimeExplanation
from timexplain._utils import optional_njit, optional_numba_list

if TYPE_CHECKING:
    from pandas import DataFrame as TC_DataFrame
    from sktime.transformers.shapelets import ShapeletTransform as TC_ShapeletTransform


# Much of this code is copied from ShapeletTransform.transform().
def align_shapelets(model: TC_ShapeletTransform, X: TC_DataFrame):
    if model.shapelets is None:
        raise Exception("Fit not called yet or no shapelets were generated")
    shapelets = optional_numba_list(shapelet.data for shapelet in model.shapelets)
    X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])
    dists, starts = _align_njit(shapelets, X)
    return dists, starts


# Externalized to allow for @njit.
@optional_njit
def _align_njit(shapelets, X):
    dists = np.zeros((len(X), len(shapelets)), dtype=np.float32)
    starts = np.zeros_like(dists, dtype=np.int32)

    # for the i^th series to transform
    for i in range(len(X)):
        this_series = X[i]

        # get the s^th shapelet
        for s in range(len(shapelets)):
            # find distance between this series and each shapelet
            min_dist = np.inf
            this_shapelet_length = shapelets[s].shape[-1]

            for start_pos in range(len(this_series[0]) - this_shapelet_length + 1):
                comparison = _align_zscore(this_series[:, start_pos:start_pos + this_shapelet_length])

                dist = np.linalg.norm(shapelets[s] - comparison)
                dist = dist * dist
                dist = 1.0 / this_shapelet_length * dist

                if dist < min_dist:
                    min_dist = dist
                    dists[i, s] = dist
                    starts[i, s] = start_pos

    return dists, starts


# Same as ShapeletTransform.zscore(), but adjusted to allow for @njit.
@optional_njit
def _align_zscore(a):
    zscored = np.empty(a.shape)
    for i, j in enumerate(a):
        sstd = j.std()
        if sstd == 0:
            zscored[i] = np.zeros(len(j))
        else:
            mns = j.mean()
            zscored[i] = (j - mns) / sstd
    return zscored


class ShapeletTransformExplainer(Explainer[TimeExplanation]):
    model: TC_ShapeletTransform

    def __init__(self, model: TC_ShapeletTransform):
        self.model = model

    def _explain(self, X_specimens):
        shapelet_dists, shapelet_starts = align_shapelets(self.model, X_specimens)
        n_specimens, n_model_outputs = shapelet_dists.shape

        explanations = []

        for specimen_idx in range(n_specimens):
            x_specimen = X_specimens.iloc[specimen_idx, 0]
            specimen_impacts = np.zeros((n_model_outputs, len(x_specimen)))

            for shapelet_idx, shapelet in enumerate(self.model.shapelets):
                start = shapelet_starts[specimen_idx, shapelet_idx]
                specimen_impacts[shapelet_idx, start:start + shapelet.length] += 1

            explanations.append(TimeExplanation(x_specimen, specimen_impacts, y_pred=shapelet_dists[specimen_idx]))

        return explanations
