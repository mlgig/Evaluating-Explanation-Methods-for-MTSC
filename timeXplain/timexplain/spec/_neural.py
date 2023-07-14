from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from timexplain._explainer import Explainer
from timexplain._explanation import TimeExplanation

if TYPE_CHECKING:
    from keras.engine.network import Network as TC_Network


class NeuralCamExplainer(Explainer[TimeExplanation]):
    model: TC_Network

    def __init__(self, model: TC_Network):
        self.model = model

    def _explain(self, X_specimens):
        from keras.layers import Dense, GlobalAveragePooling1D
        from keras.backend import function

        X_specimens = np.asarray(X_specimens)

        layers = self.model.layers
        if isinstance(layers[-1], Dense) and isinstance(layers[-2], GlobalAveragePooling1D):
            final_weights = layers[-1].get_weights()[0]
            beforegap_and_pred = function(self.model.inputs, [layers[-3].output, layers[-1].output])
            beforegap, pred = beforegap_and_pred([X_specimens])
        else:
            raise ValueError("CAM for keras networks is only available when the last two layers are "
                             "GlobalAveragePooling1D and Dense.")

        # TODO: Remove this once multivariate time series are supported.
        if X_specimens.ndim == 3:
            if X_specimens.shape[2] != 1:
                raise ValueError("Multivariate time series are not yet supported.")
            X_specimens = X_specimens[..., 0]

        cams = beforegap @ final_weights
        return [TimeExplanation(x_specimen, cam.T, y_pred=pr) for x_specimen, pr, cam in zip(X_specimens, pred, cams)]
