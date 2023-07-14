from __future__ import annotations

from typing import Union, Optional, Callable, NamedTuple, Sequence

import numpy as np

from timexplain._explanation import StatisticExplanation
from timexplain.om._base import Omitter


class Stats(NamedTuple):
    mean: float
    stddev: float


def stats_global(X_bg) -> Stats:
    assert X_bg is not None
    return Stats(np.mean(np.asarray(X_bg)),
                 np.mean(np.std(np.asarray(X_bg), axis=1)))


def stats_sample(X_bg) -> Stats:
    return Stats(np.mean(_sample(X_bg)),
                 np.std(_sample(X_bg)))


def _sample(X_bg):
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return X_bg[np.random.randint(X_bg.shape[0])]


class StatisticOmitter(Omitter):
    stats_repl: Optional[Stats]

    def __init__(self, stats_repl: Union[Stats, Callable[[np.ndarray], Stats]] = stats_sample, X_bg=None):
        self.constr_args = {"stats_repl": stats_repl, "X_bg": X_bg}
        self.deterministic = stats_repl != stats_sample
        self.requires_X_bg = callable(stats_repl)

        self.size_z = 2

        if callable(stats_repl):
            stats_repl = None if X_bg is None else stats_repl(np.asarray(X_bg))
        self.stats_repl = stats_repl

    def _omit(self, X, Z):
        if self.stats_repl is None:
            raise ValueError("This StatisticOmitter has no concrete stats_repl. "
                             "Have you maybe forgotten to pass in X_bg?")

        keep_means = Z[:, 0]
        keep_stddevs = Z[:, 1]

        sample_means = np.mean(X, axis=1)
        sample_stddevs = np.std(X, axis=1)

        X = X - sample_means[:, np.newaxis]
        X *= np.where(keep_stddevs, 1, self.stats_repl.stddev / sample_stddevs)[:, np.newaxis]
        X += np.where(keep_means, sample_means, self.stats_repl.mean)[:, np.newaxis]

        return X

    def create_explanation(self, x_specimen: Sequence[float], y_pred: Optional, impacts: np.ndarray) \
            -> StatisticExplanation:
        return StatisticExplanation(x_specimen, impacts, y_pred=y_pred)
