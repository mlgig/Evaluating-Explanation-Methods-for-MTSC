from __future__ import annotations

from typing import Optional, Sequence, List, Tuple

import numpy as np

from timexplain._explanation import EnsembleExplanation
from timexplain._utils import unpublished
from timexplain.om._base import Omitter


@unpublished
class EnsembleOmitter(Omitter):
    sub_omitters: Sequence[Omitter]
    sub_z_indices: List[Tuple[int, int]]

    def __init__(self, sub_omitters: Sequence[Omitter], X_bg=None):
        self.constr_args = {"sub_omitters": sub_omitters}
        self.deterministic = any(sub_omitter.deterministic for sub_omitter in sub_omitters)
        self.requires_X_bg = any(sub_omitter.requires_X_bg for sub_omitter in sub_omitters)

        sub_omitters = [sub_omitter.refine(X_bg=X_bg) if sub_omitter.requires_X_bg else sub_omitter
                        for sub_omitter in sub_omitters]
        self.sub_omitters = sub_omitters
        self.size_z = sum(sub_omitter.size_z for sub_omitter in sub_omitters)

        sub_z_indices = []
        start_idx = 0
        for sub_omitter in sub_omitters:
            sub_size_z = sub_omitter.size_z
            sub_z_indices.append((start_idx, start_idx + sub_size_z))
            start_idx += sub_size_z
        self.sub_z_indices = sub_z_indices

    def _omit(self, X, Z):
        for sub_omitter, (start, end) in zip(self.sub_omitters, self.sub_z_indices):
            X = sub_omitter.omit(X, Z[:, start:end])
        return X

    def create_explanation(self, x_specimen: Sequence[float], y_pred: Optional, impacts: np.ndarray) \
            -> EnsembleExplanation:
        assert impacts.shape[-1] == self.size_z, f"Number of impacts ({impacts.shape[-1]}) must match " \
                                                 f"size_z of EnsembleOmitter ({self.size_z})."
        return EnsembleExplanation(x_specimen, [
            sub_omitter.create_explanation(x_specimen, y_pred, impacts[:, start:end])
            for sub_omitter, (start, end) in zip(self.sub_omitters, self.sub_z_indices)
        ], y_pred=y_pred)
