from functools import wraps
from inspect import isclass
from typing import TypeVar, Any, Union, Collection, Sequence, List, Mapping
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.fft import rfft
from scipy.signal import get_window

T = TypeVar("T")
SingleOrCollection = Union[T, Collection[T]]
SingleOrSeq = Union[T, Sequence[T]]
SingleOrList = Union[T, List[T]]
SingleOrMapping = Union[T, Mapping[Any, T]]
DenseOrSparse = Union[np.ndarray, sparse.csr_matrix]


def classname(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def without(seq, idx):
    if not isinstance(seq, List):
        seq = list(seq)
    return seq[:idx] + seq[idx + 1:]


# A custom implementation is required because functions like np.nanmean() raise warnings
# when encountering all-nan inputs, and we don't want those warnings.
def nan_op(ma_op, arr, *ma_op_args, **ma_op_kwargs):
    result = np.ma.filled(ma_op(np.ma.array(arr, mask=np.isnan(arr)), *ma_op_args, **ma_op_kwargs), np.nan)
    return result if result.ndim > 0 else result[()]


def take_if_deep(arr, idx, ndim, error):
    if not isinstance(arr, (np.ndarray, sparse.csr_matrix)):
        arr = np.array(arr)
    if arr.ndim < ndim:
        return arr
    else:
        if idx is None:
            if arr.shape[0] == 1:
                idx = 0
            else:
                raise ValueError(error)
        return arr[idx]


def binary_cube(size):
    assert 1 <= size <= 16
    return np.unpackbits(
        np.arange(2 ** size, dtype=np.uint16).byteswap()[:, np.newaxis].view(np.uint8),
        axis=1)[:, 16 - size:]


def rfft_magnitudes(sig, window: Union[None, str, float, tuple] = "hamming"):
    n = np.shape(sig)[-1]
    # 1. We apply a window and normalize the signal such that the zeroth frequency bin will be 0.
    if window is not None:
        win = get_window(window, n)
        sig = (sig - np.sum(sig * win) / np.sum(win)) * win
    # 2. rfft() computes the upper part (upper n//2 + 1 bins) of the symmetric and complex-valued spectrum.
    # 3. We discard the zeroth frequency bin, which is, by definition of DFT, just the sum of the signal.
    #    This leaves us with n//2 frequency bins.
    # 4. We take the absolute of the complex-valued frequency bins, resulting in the magnitude of each bin.
    # 5. Because we have discarded the lower half of the symmetric spectrum, we have to multiply the remaining
    #    magnitudes by two to account for the missing bins.
    # 6. Finally, we normalize the magnitudes by dividing them by n.
    return np.abs(rfft(sig)[..., 1:]) * 2 / n


def replace_zero_slices(slicing, repl_slices, repl_strat, X, Z):
    for row, z in enumerate(Z):
        for pos in np.where(z == 0)[0]:
            start, stop = slicing.bin_slices[pos]
            X[row, start:stop] = repl_strat(X[row, start:stop], repl_slices[pos])


def aggregate_predict_deaggregate(model_predict, jobs, dedup=False):
    agg_x = np.vstack([x for x, _ in jobs])

    if not dedup:
        agg_y = model_predict(agg_x)
    else:
        # To avoid letting the model predict the same samples multiple times, we remove duplicate samples.
        dedup_x, dup_idx = np.unique(agg_x, axis=0, return_inverse=True)
        dedup_y = model_predict(dedup_x)
        agg_y = dedup_y[dup_idx]

    sample_ctr = 0
    for x, y_target in jobs:
        job_samples = len(x)
        y_target(agg_y[sample_ctr:sample_ctr + job_samples])
        sample_ctr += job_samples


class UnpublishedWarning(UserWarning):
    pass


def unpublished(target):
    msg = f"{target.__name__} implements an unpublished research mechanism. " \
          "It may very well change in the future or disappear completely. Be cautious."

    if isclass(target):
        target_init = target.__init__

        def __init__(*args, **kwargs):
            warn(msg, UnpublishedWarning)
            target_init(*args, **kwargs)

        target.__init__ = __init__
        return target
    else:
        @wraps(target)
        def wrapper(*args, **kwargs):
            warn(msg, UnpublishedWarning)
            return target(*args, **kwargs)

        return wrapper


def optional_njit(target):
    try:
        from numba import njit
        return njit(target)
    except ImportError:
        @wraps(target)
        def wrapper(*args, **kwargs):
            warn(f"{target.__name__} is dog-slow right now and can be significantly sped up by installing numba.")
            return target(*args, **kwargs)

        return wrapper()


def optional_numba_list(content):
    try:
        from numba.typed import List
        return List(content)
    except ImportError:
        return list(content)


def optional_numba_dict(key_type, value_type):
    try:
        from numba import types
        from numba.typed import Dict
        return Dict.empty(getattr(types, key_type), getattr(types, value_type))
    except ImportError:
        return {}
