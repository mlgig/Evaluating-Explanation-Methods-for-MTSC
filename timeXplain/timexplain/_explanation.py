from __future__ import annotations

import operator
from abc import abstractmethod
from dataclasses import dataclass
from itertools import repeat
from numbers import Integral
from typing import TYPE_CHECKING, Union, Optional, Sequence, List, Tuple, Callable

import numpy as np
from scipy import sparse

from timexplain._utils import DenseOrSparse, classname, take_if_deep

if TYPE_CHECKING:
    from pandas import Series as TC_Series

    XSpecimen = Union[np.array, TC_Series, sparse.csr_matrix]


@dataclass(init=False)
class Explanation:
    x_specimen: XSpecimen
    y_pred: Optional

    def __init__(self, x_specimen: XSpecimen, y_pred=None):
        if not isinstance(x_specimen, (np.ndarray, sparse.csr_matrix)) and \
                not classname(x_specimen) == "pandas.core.series.Series":
            raise ValueError("Explanation x_specimen must be either a numpy array, a pandas series, "
                             "or a scipy sparse CSR matrix.")
        self.x_specimen = x_specimen
        self.y_pred = y_pred

    @abstractmethod
    def map(self, func: Callable[[DenseOrSparse], DenseOrSparse]) -> Explanation:
        raise NotImplementedError

    @staticmethod
    def reduce(func: Callable[[List[DenseOrSparse]], DenseOrSparse], explanations: Sequence[Explanation]) \
            -> Explanation:
        return explanations[0]._reduce(func, explanations)

    @staticmethod
    @abstractmethod
    def _reduce(func, expls):
        raise NotImplementedError


@dataclass(init=False)
class EnsembleExplanation(Explanation):
    sub_explanations: List[Explanation]

    def __init__(self, x_specimen: XSpecimen, sub_explanations: List[Explanation], *, y_pred=None):
        super().__init__(x_specimen, y_pred)
        self.sub_explanations = sub_explanations

    def map(self, func: Callable[[DenseOrSparse], DenseOrSparse]) -> EnsembleExplanation:
        return EnsembleExplanation(self.x_specimen, [sub.map(func) for sub in self.sub_explanations],
                                   y_pred=self.y_pred)

    @staticmethod
    def _reduce(func, expls):
        return EnsembleExplanation(expls[0].x_specimen,
                                   [Explanation.reduce(func, [expl.sub_explanations[sub_idx] for expl in expls])
                                    for sub_idx in range(len(expls[0].sub_explanations))],
                                   y_pred=expls[0].y_pred)


@dataclass(init=False)
class TimeExplanation(Explanation):
    impacts: DenseOrSparse  # Shape: n_model_outputs x n_slices
    time_slicing: Slicing

    def __init__(self, x_specimen: XSpecimen, impacts: np.ndarray, *, y_pred=None, time_slicing: Slicing = None):
        super().__init__(x_specimen, y_pred)

        # Slicing
        size_x = x_specimen.shape[-1]
        if time_slicing is None:
            self.time_slicing = Slicing(bin_rate=1, n_slices=size_x, bin_interval=(0, size_x))
        else:
            if np.any(time_slicing.cont_edges < 0) or np.any(time_slicing.cont_edges > size_x - 1):
                raise ValueError("Continuous time slice edges must not exceed [0, size_x-1].")
            self.time_slicing = time_slicing

        # Impacts
        if not isinstance(impacts, (np.ndarray, sparse.csr_matrix)):
            raise ValueError("Time impacts must be either numpy array or scipy sparse CSR matrix.")
        if impacts.shape[-1] != self.time_slicing.n_slices:
            raise ValueError(f"Number of time impacts ({impacts.shape[-1]}) must match "
                             f"number of time slices ({self.time_slicing.n_slices}).")
        self.impacts = impacts

    def explode(self, divide_spread=False) -> TimeExplanation:
        if self.time_slicing.bin_rate is None:
            raise ValueError("Cannot explode TimeExplanation when bin rate of time slicing is unknown.")

        size_x = self.x_specimen.shape[-1]
        n_model_outputs = self.impacts.shape[0]
        time_slices = self.time_slicing.cont_proper_vol_slices * self.time_slicing.bin_rate

        exploded_impacts = np.zeros((n_model_outputs, size_x))
        for t_slice, (t_start, t_stop) in enumerate(time_slices):
            contrib = self.impacts[:, t_slice]
            if divide_spread:
                slice_vol = (t_stop - t_start)
                contrib = contrib / slice_vol
            _softidx(exploded_impacts, np.s_[:, t_start + 0.5:t_stop + 0.5], operator.add, contrib[:, np.newaxis])

        exploded_slicing = Slicing(bin_rate=self.time_slicing.bin_rate, n_slices=size_x, bin_interval=(0, size_x))
        return TimeExplanation(self.x_specimen, exploded_impacts, y_pred=self.y_pred, time_slicing=exploded_slicing)

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> TimeExplanation:
        return TimeExplanation(self.x_specimen, func(self.impacts), y_pred=self.y_pred, time_slicing=self.time_slicing)

    @staticmethod
    def _reduce(func, expls):
        return TimeExplanation(expls[0].x_specimen, func([o.impacts for o in expls]),
                               y_pred=expls[0].y_pred, time_slicing=expls[0].time_slicing)


TabularExplanation = TimeExplanation


@dataclass(init=False)
class FreqExplanation(Explanation):
    impacts: np.ndarray  # Shape: n_model_outputs x n_time_slices*n_freq_slices
    time_slicing: Slicing
    freq_slicing: Slicing

    def __init__(self, x_specimen: XSpecimen, impacts: np.ndarray,
                 *, y_pred=None, time_slicing: Slicing = None, freq_slicing: Slicing = None):
        super().__init__(x_specimen, y_pred)

        # Slicing
        size_x = x_specimen.shape[-1]
        if time_slicing is None:
            self.time_slicing = Slicing(bin_rate=1, n_slices=size_x, bin_interval=(0, size_x))
        else:
            if np.any(time_slicing.cont_edges < 0) or np.any(time_slicing.cont_edges > size_x - 1):
                raise ValueError("Continuous time slice edges must not exceed [0, size_x-1].")
            self.time_slicing = time_slicing

        if freq_slicing is None:
            self.freq_slicing = Slicing(bin_rate=size_x / self.time_slicing.bin_rate, n_slices=size_x // 2 + 1,
                                        bin_interval=(0, size_x // 2 + 1))
        else:
            if np.any(freq_slicing.cont_edges < 0) or np.any(freq_slicing.cont_edges > size_x // 2):
                raise ValueError("Freq slice edges must not exceed [0, size_x//2].")
            self.freq_slicing = freq_slicing

        # Impacts
        if not isinstance(impacts, (np.ndarray, sparse.csr_matrix)):
            raise ValueError("Frequency impacts must be either numpy array or scipy sparse CSR matrix.")
        n_dices = self.time_slicing.n_slices * self.freq_slicing.n_slices
        if impacts.shape[-1] != n_dices:
            raise ValueError(f"Number of frequency impacts ({impacts.shape[-1]}) must match "
                             f"number of time slices by time slice edge array ({self.time_slicing.n_slices}) times "
                             f"number of frequency slices by frequency slice edge array ({self.freq_slicing.n_slices})"
                             f", totaling {n_dices}.")
        self.impacts = impacts

    def reshaped_impacts(self, model_output: int = None):
        n_model_outputs = self.impacts.shape[0]
        if model_output is None and isinstance(self.impacts, np.ndarray) and self.impacts.ndim == 2:
            return self.impacts.reshape((n_model_outputs, self.time_slicing.n_slices, -1))
        else:
            return take_if_deep(self.impacts, model_output, 2,
                                "Must supply 'model_output' when impacts are sparse and are available for "
                                "multiple model outputs.") \
                .reshape((self.time_slicing.n_slices, -1))

    def explode(self, divide_spread=False) -> FreqExplanation:
        if self.time_slicing.bin_rate is None or self.freq_slicing.bin_rate is None:
            raise ValueError("Cannot explode FreqExplanation when bin rate of either time or freq slicing is unknown.")

        size_x = self.x_specimen.shape[-1]
        n_model_outputs = self.impacts.shape[0]
        time_slices = self.time_slicing.cont_proper_vol_slices * self.time_slicing.bin_rate
        freq_slices = self.freq_slicing.cont_proper_vol_slices * self.freq_slicing.bin_rate

        reshaped_impacts = self.reshaped_impacts()
        exploded_impacts = np.zeros((n_model_outputs, size_x, size_x // 2 + 1))
        for t_slice, (t_start, t_stop) in enumerate(time_slices):
            for f_slice, (f_start, f_stop) in enumerate(freq_slices):
                contrib = reshaped_impacts[:, t_slice, f_slice]
                if divide_spread:
                    dice_vol = (t_stop - t_start) * (f_stop - f_start)
                    contrib = contrib / dice_vol
                _softidx(exploded_impacts, np.s_[:, t_start + 0.5:t_stop + 0.5, f_start + 0.5:f_stop + 0.5],
                         operator.add, contrib[:, np.newaxis, np.newaxis])
        exploded_impacts = exploded_impacts.reshape((n_model_outputs, -1))

        exploded_time_slicing = Slicing(bin_rate=self.time_slicing.bin_rate, n_slices=size_x,
                                        bin_interval=(0, size_x))
        exploded_freq_slicing = Slicing(bin_rate=self.freq_slicing.bin_rate, n_slices=size_x // 2 + 1,
                                        bin_interval=(0, size_x // 2 + 1))
        return FreqExplanation(self.x_specimen, exploded_impacts,
                               y_pred=self.y_pred, time_slicing=exploded_time_slicing,
                               freq_slicing=exploded_freq_slicing)

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> FreqExplanation:
        return FreqExplanation(self.x_specimen, func(self.impacts),
                               y_pred=self.y_pred, time_slicing=self.time_slicing,
                               freq_slicing=self.freq_slicing)

    @staticmethod
    def _reduce(func, expls):
        return FreqExplanation(expls[0].x_specimen, func([o.impacts for o in expls]),
                               y_pred=expls[0].y_pred, time_slicing=expls[0].time_slicing,
                               freq_slicing=expls[0].freq_slicing)


@dataclass(init=False)
class StatisticExplanation(Explanation):
    impacts: np.ndarray
    mean_impacts: Sequence[float]
    stddev_impacts: Sequence[float]

    def __init__(self, x_specimen: XSpecimen, impacts: np.ndarray, *, y_pred: Optional):
        super().__init__(x_specimen, y_pred)

        if not isinstance(impacts, np.ndarray):
            raise ValueError("Statistic impacts must be numpy array.")
        if impacts.shape[-1] != 2:
            raise ValueError(f"Number of statistic impacts ({impacts.shape[-1]}) must be 2.")

        self.impacts = impacts
        self.mean_impacts = impacts[..., 0]
        self.stddev_impacts = impacts[..., 1]

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> StatisticExplanation:
        return StatisticExplanation(self.x_specimen, func(self.impacts), y_pred=self.y_pred)

    @staticmethod
    def _reduce(func, expls):
        return StatisticExplanation(expls[0].x_specimen, func([o.impacts for o in expls]), y_pred=expls[0].y_pred)


class Slicing:
    n_slices: Optional[int]
    cont_slices: Optional[np.ndarray]  # DType: float, Shape: n x 2
    cont_proper_vol_slices: Optional[np.ndarray]  # DType: float, Shape: n x 2
    bin_rate: Optional[float]
    bin_slices: Optional[np.ndarray]  # DType: int, Shape: n x 2

    # Spacing: linear, quadratic
    # min_slice_vol is only relevant when spacing="quadratic"
    def __init__(self, *,
                 bin_rate: float = None,
                 n_slices: int = None, spacing="linear", first_slice_vol=1.0,
                 cont_interval: Tuple[float, float] = None, bin_interval: Tuple[int, int] = None,
                 cont_slices=None, cont_edges: Sequence[float] = None,
                 bin_slices=None, bin_edges: Sequence[int] = None):
        self.constr_args = {"bin_rate": bin_rate, "n_slices": n_slices, "spacing": spacing,
                            "first_slice_vol": first_slice_vol, "cont_interval": cont_interval,
                            "bin_interval": bin_interval, "cont_slices": cont_slices, "cont_edges": cont_edges,
                            "bin_slices": bin_slices, "bin_edges": bin_edges}

        if bin_rate is not None and bin_rate <= 0:
            raise ValueError(f"Bin rate must be > 0, not '{bin_rate}'.")
        if spacing not in ("linear", "quadratic"):
            raise ValueError(f"Spacing must be either 'linear' or 'quadratic', not '{spacing}'.")

        self.bin_rate = bin_rate
        self.n_slices = n_slices

        self.cont_slices = None
        self.cont_proper_vol_slices = None
        self.bin_slices = None

        # If n_slices and an interval are input, use those to construct cont_edges resp. bin_edges.
        if n_slices is not None:
            if cont_interval is not None:
                if spacing == "linear":
                    cont_edges = np.linspace(cont_interval[0], cont_interval[1], n_slices + 1)
                elif spacing == "quadratic":
                    cont_edges = _quadspace(cont_interval[0], cont_interval[1], n_slices + 1, first_slice_vol)
            elif bin_interval is not None:
                if spacing == "linear":
                    bin_edges = np.linspace(bin_interval[0], bin_interval[1], n_slices + 1, dtype=int)
                elif spacing == "quadratic":
                    bin_edges = _quadspace(bin_interval[0], bin_interval[1], n_slices + 1, first_slice_vol).astype(int)

        # If cont_slices or bin_slices are input, use those.
        # If cont_edges or bin_edges are input or generated from n_slices, use those to create slices.
        if cont_slices is not None:
            self.cont_slices = np.asarray(cont_slices, dtype=float)
        elif cont_edges is not None:
            cont_edges = np.asarray(cont_edges, dtype=float)
            self.cont_slices = np.column_stack([cont_edges[:-1], cont_edges[1:]])
        elif bin_slices is not None:
            self.bin_slices = np.asarray(bin_slices, dtype=int)
        elif bin_edges is not None:
            bin_edges = np.asarray(bin_edges, dtype=int)
            self.bin_slices = np.column_stack([bin_edges[:-1], bin_edges[1:]])

        # Determine n_slices from cont_slices or bin_slices.
        if self.cont_slices is not None:
            self.n_slices = self.cont_slices.shape[0]
        elif self.bin_slices is not None:
            self.n_slices = self.bin_slices.shape[0]

        # If bin_rate is input, convert bin_slices to cont_slices, compute cont_proper_vol_slices,
        # and convert cont_slices to bin_slices, depending on what is available.
        if bin_rate is not None:
            if self.bin_slices is not None and self.cont_slices is None:
                self.cont_slices = (self.bin_slices - 0.5) / bin_rate
                for s in range(self.n_slices):
                    if s == 0 or self.bin_slices[s - 1, 1] != self.bin_slices[s, 0]:
                        self.cont_slices[s, 0] += 0.5 / bin_rate
                    if s == self.n_slices - 1 or self.bin_slices[s, 1] != self.bin_slices[s + 1, 0]:
                        self.cont_slices[s, 1] -= 0.5 / bin_rate

            if self.cont_slices is not None:
                self.cont_proper_vol_slices = np.copy(self.cont_slices)
                for s in range(self.n_slices):
                    if s == 0 or not np.isclose(self.cont_slices[s - 1, 1], self.cont_slices[s, 0]):
                        self.cont_proper_vol_slices[s, 0] -= 0.5 / bin_rate
                    if s == self.n_slices - 1 or not np.isclose(self.cont_slices[s, 1], self.cont_slices[s + 1, 0]):
                        self.cont_proper_vol_slices[s, 1] += 0.5 / bin_rate

                potential_bin_slices = self.cont_proper_vol_slices * bin_rate + 0.5
                if self.bin_slices is None and np.allclose(potential_bin_slices, np.round(potential_bin_slices)):
                    self.bin_slices = np.round(potential_bin_slices).astype(int)

        # Check that cont_slices and bin_slices have positive volume.
        if self.cont_slices is not None and np.any(np.diff(self.cont_slices) <= 0):
            raise ValueError("Continuous slices must have volume > 0.")
        if self.bin_slices is not None and np.any(np.diff(self.bin_slices) <= 0):
            raise ValueError("Bin slices must have volume >= 1.")

    def refine(self, **kwargs) -> Slicing:
        return Slicing(**{**self.constr_args, **kwargs})

    @property
    def cont_edges(self) -> Optional[np.ndarray]:
        return None if self.cont_slices is None else np.unique(self.cont_slices)

    @property
    def cont_proper_vol_edges(self) -> Optional[np.ndarray]:
        return None if self.cont_proper_vol_slices is None else np.unique(self.cont_proper_vol_slices)

    @property
    def bin_edges(self) -> Optional[np.ndarray]:
        return None if self.bin_slices is None else np.unique(self.bin_slices)

    @property
    def is_contiguous(self) -> bool:
        return all(np.isclose(self.cont_slices[i, 1], self.cont_slices[i + 1, 0]) for i in range(self.n_slices - 1))


def _quadspace(start, end, num, first_step):
    delta = end - start
    a = (delta - first_step * (num - 1)) / ((num - 1) * (num - 2))
    b = (delta - a * (num - 1) ** 2) / (num - 1)

    def f(x): return a * x ** 2 + b * x + start

    # Note: We manually add start and end to avoid floating point errors.
    return np.hstack([start, np.apply_along_axis(f, 0, np.arange(1, num - 1)), end])


def _softidx(arr, key, op, value):
    try:
        key = list(key)
    except TypeError:
        key = [key]
    key += repeat(slice(None), arr.ndim - len(key))
    multipliers = []

    for dim in range(len(key)):
        if isinstance(key[dim], slice):
            start = key[dim].start
            stop = key[dim].stop
            if start is None:
                start = 0
            if stop is None:
                stop = arr.shape[dim]

            start_floor = np.floor(start)
            start_ceil = np.ceil(start)
            stop_floor = np.floor(stop)
            stop_ceil = np.ceil(stop)

            if start >= 0 and stop >= 0 and key[dim].step is None and \
                    (not isinstance(start, Integral) or not isinstance(stop, Integral)):
                key[dim] = slice(int(start_floor), int(stop_ceil))
                if start_floor == stop_floor:
                    multipliers.append((dim, 0, stop - start))
                elif start != start_floor:
                    multipliers.append((dim, 0, start_ceil - start))
                elif stop != stop_floor:
                    multipliers.append((dim, -1, stop - stop_floor))

    key = tuple(key)
    sub_arr = arr[key]
    value = np.array(np.broadcast_to(value, sub_arr.shape), dtype=float)  # Float is necessary because we multiply.
    for dim, idx, mul in multipliers:
        value[(*repeat(slice(None), dim), idx)] *= mul
    arr[key] = op(sub_arr, value)
