from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence
from warnings import warn

import numpy as np

# X references to the space of the original time series.
# Z references to the space of the simplified inputs.
from timexplain._explanation import Explanation


class Omitter(ABC):
    deterministic: bool
    requires_X_bg: bool
    size_z: int

    def refine(self, **kwargs) -> Omitter:
        return type(self)(**{**self.constr_args, **kwargs})

    def omit(self, X, Z) -> np.ndarray:
        X = np.asarray(X)
        Z = np.asarray(Z)

        if X.ndim == 1 and Z.ndim == 1:
            if Z.shape[0] != self.size_z:
                raise ValueError(f"Length of Z must match size_z of {type(self).__name__}.")
            return self._omit(X[np.newaxis], Z[np.newaxis])[0, :]

        if X.ndim == 1 and Z.ndim == 2:
            X = np.tile(X, (Z.shape[0], 1))
        elif X.ndim == 2 and Z.ndim == 1:
            Z = np.tile(Z, (X.shape[0], 1))

        if X.ndim == 2 and Z.ndim == 2:
            if X.shape[0] != Z.shape[0]:
                raise ValueError("Number of samples in X must match number of samples in Z.")
            if Z.shape[1] != self.size_z:
                raise ValueError(f"Length of each sample in Z must match size_z of {type(self).__name__}.")
            return self._omit(X, Z)
        else:
            raise ValueError("X.ndim must be either 1 or 2 and Z.ndim must be either 1 or 2; "
                             f"X.ndim={X.ndim}, Z.ndim={Z.ndim} is thus illegal.")

    @abstractmethod
    def _omit(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def create_explanation(self, x_specimen: Sequence[float], y_pred: Optional, impacts: np.ndarray) \
            -> Explanation:
        raise NotImplementedError

    def z_specimen(self, dims=1) -> np.ndarray:
        return np.ones((*np.ones(dims - 1, dtype=int), self.size_z), dtype=int)

    def z_empty(self, dims=1) -> np.ndarray:
        return np.zeros((*np.ones(dims - 1, dtype=int), self.size_z), dtype=int)

    # Utility for subclasses.
    def _clamp_size_z(self, size_z: int, max_size_z: int) -> int:
        if size_z > max_size_z:
            warn(f"{type(self).__name__} automatically shrank z vector size to {size_z}.", stacklevel=4)
            return max_size_z
        else:
            return size_z
