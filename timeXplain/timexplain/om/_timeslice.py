from __future__ import annotations

from numbers import Integral
from typing import Union, Optional, Callable, Sequence, List

import numpy as np

from timexplain._explanation import TimeExplanation, Slicing
from timexplain._utils import replace_zero_slices
from timexplain.om._base import Omitter
from timexplain.om._surrogate import x_sample

XRepl = Sequence[float]
ReplStrat = Callable[[np.ndarray, np.ndarray], np.ndarray]


class TimeSliceOmitter(Omitter):
    size_x: int
    time_slicing: Slicing
    x_repl: Optional[Sequence[float]]
    repl_slices: List[Sequence[float]]
    repl_strat: ReplStrat

    def __init__(self, size_x: int, time_slicing: Union[int, Slicing],
                 x_repl: Union[XRepl, Callable[[np.ndarray], XRepl]] = x_sample,
                 repl_strat: ReplStrat = lambda a, b: b,
                 sample_rate=1.0, X_bg=None):
        self.constr_args = {"size_x": size_x, "time_slicing": time_slicing, "x_repl": x_repl,
                            "repl_strat": repl_strat, "sample_rate": sample_rate, "X_bg": X_bg}
        self.deterministic = x_repl != x_sample
        self.requires_X_bg = callable(x_repl)

        self.size_x = size_x
        self.repl_strat = repl_strat

        if isinstance(time_slicing, Integral):
            time_slicing = Slicing(n_slices=time_slicing)
        time_slicing = time_slicing.refine(bin_rate=sample_rate, bin_interval=(0, size_x))
        if time_slicing.bin_slices is None:
            raise ValueError("TimeSliceOmitter requires a slicing with bin slices.")
        if np.any(time_slicing.bin_edges < 0) or np.any(time_slicing.bin_edges > size_x):
            raise ValueError(f"Time slice bin edges must not exceed [0, size_x]: {time_slicing.bin_slices}")
        self.time_slicing = time_slicing

        if callable(x_repl):
            x_repl = None if X_bg is None else x_repl(np.asarray(X_bg))
        self.x_repl = x_repl

        if x_repl is not None:
            if size_x != len(x_repl):
                raise ValueError(f"Length of x_repl {len(x_repl)} must match "
                                 f"size_x of TimeSliceOmitter ({size_x}).")
            self.repl_slices = [x_repl[start:stop] for start, stop in time_slicing.bin_slices]

    @property
    def size_z(self) -> int:
        return self.time_slicing.n_slices

    def _omit(self, X, Z):
        if self.x_repl is None:
            raise ValueError("This TimeSliceOmitter has no concrete x_repl. "
                             "Have you maybe forgotten to pass in X_bg?")
        if X.shape[1] != self.size_x:
            raise ValueError("Length of each sample in X must match size_x of TimeSliceOmitter.")

        X = X.copy()
        replace_zero_slices(self.time_slicing, self.repl_slices, self.repl_strat, X, Z)
        return X

    def create_explanation(self, x_specimen: Sequence[float], y_pred: Optional, impacts: np.ndarray) -> TimeExplanation:
        return TimeExplanation(x_specimen, impacts, y_pred=y_pred, time_slicing=self.time_slicing)
