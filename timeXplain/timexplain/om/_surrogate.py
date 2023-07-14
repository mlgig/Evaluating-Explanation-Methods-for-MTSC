from typing import Sequence

import numpy as np


def x_zero(X_bg) -> Sequence[float]:
    assert X_bg is not None
    return np.zeros(np.shape(X_bg)[1])


def x_global_mean(X_bg) -> Sequence[float]:
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return np.repeat(_global_mean(X_bg), X_bg.shape[1])


def x_local_mean(X_bg) -> Sequence[float]:
    assert X_bg is not None
    return np.mean(np.asarray(X_bg), axis=0)


def x_zero_noise(X_bg) -> Sequence[float]:
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return np.random.normal(loc=0, scale=_global_stddev(X_bg), size=X_bg.shape[1])


def x_global_noise(X_bg) -> Sequence[float]:
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return np.random.normal(loc=_global_mean(X_bg), scale=_global_stddev(X_bg), size=X_bg.shape[1])


def x_local_noise(X_bg) -> Sequence[float]:
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return np.array([np.random.normal(loc=m, scale=s) for m, s in zip(np.mean(X_bg, axis=0), np.std(X_bg, axis=0))])


def x_sample(X_bg) -> Sequence[float]:
    assert X_bg is not None
    X_bg = np.asarray(X_bg)
    return X_bg[np.random.randint(X_bg.shape[0])]


def _global_mean(X_bg):
    return np.mean(np.asarray(X_bg))


def _global_stddev(X_bg):
    return np.mean(np.std(np.asarray(X_bg), axis=1))
