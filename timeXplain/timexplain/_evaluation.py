import heapq
from numbers import Integral
from operator import itemgetter, add
from typing import Union, Collection, List, Tuple

import numpy as np

from timexplain._explanation import TimeExplanation, FreqExplanation
from timexplain._similarity import correlation
from timexplain._utils import aggregate_predict_deaggregate, optional_njit
from timexplain.om import Omitter, TimeSliceOmitter, FreqDiceFilterOmitter, x_zero_noise


# n_intervals can be None to use all intervals.
# interval_sizes and warp_window_size can be absolute (>= 1) or relative (< 1).
def dtw_interval_fidelity(model_predict: callable, explanation: TimeExplanation, X_bg,
                          *, interval_sizes: Union[int, Collection[int], Collection[float]] = 10,
                          n_intervals_per_interval_size=50,
                          n_countersamples_per_interval=1,
                          warp_window_size: Union[int, float] = 0.1,
                          return_curves=False, return_points=False) \
        -> Union[np.ndarray,
                 Tuple[np.ndarray, np.ndarray],
                 Tuple[np.ndarray, List[np.ndarray]],
                 Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    x_specimen = explanation.x_specimen
    impacts = explanation.explode(divide_spread=True).impacts

    size_x = len(x_specimen)
    if isinstance(interval_sizes, Integral):
        interval_sizes = np.linspace(1, size_x, num=interval_sizes, dtype=int)
    elif any(s < 1 for s in interval_sizes):
        interval_sizes = [int(s * size_x) for s in interval_sizes]
    n_interval_sizes = len(interval_sizes)
    if warp_window_size < 1:
        warp_window_size = int(warp_window_size * size_x)

    Y_base = model_predict(np.array([x_specimen]))[0, :]

    pred_jobs = []
    points = [[] for _ in range(n_interval_sizes)]  # One sub-list for each interval size.

    for interval_size_idx, interval_size in enumerate(interval_sizes):
        n_intervals = n_intervals_per_interval_size
        max_intervals = size_x - interval_size + 1
        if n_intervals is None or n_intervals > max_intervals:
            n_intervals = max_intervals

        for interval_idx in range(n_intervals):
            interval_start = int((size_x - interval_size) * interval_idx / max(1, n_intervals - 1))  # inclusive
            interval_end = interval_start + interval_size  # exclusive

            interval_impact_sums = np.sum(impacts[:, interval_start:interval_end], axis=1)

            inversions = np.isin(np.arange(size_x), np.arange(interval_start, interval_end))
            warps = [(x_env, *_windowed_matching_dtw(x_specimen, x_env, inversions, warp_window_size))
                     for x_env in X_bg]
            countersamples = heapq.nsmallest(n_countersamples_per_interval, warps, key=itemgetter(1))
            X_countersamples = np.array([x_env[path] for x_env, _, path in countersamples])

            def callback(Y_countersamples, isi=interval_size_idx, iis=interval_impact_sums):
                points[isi].extend(np.transpose([iis, pred_deviation]) for pred_deviation in Y_base - Y_countersamples)

            pred_jobs.append((X_countersamples, callback))

    aggregate_predict_deaggregate(model_predict, pred_jobs)

    # Make model_output the first axis.
    points = [np.swapaxes(is_points, 0, 1) for is_points in points]
    # Compute correlation curves.
    curves = np.array([[correlation(is_points[model_output, :, 0], is_points[model_output, :, 1])
                        for is_points in points]
                       for model_output in range(len(Y_base))])

    if n_interval_sizes == 1:
        points = points[0]
        curves = curves[:, 0]
        auc = curves
    else:
        # Integrate curves.
        auc = np.trapz(curves, dx=1 / n_interval_sizes)

    if return_curves and return_points:
        return auc, curves, points
    elif return_curves:
        return auc, curves
    elif return_points:
        return auc, points
    else:
        return auc


@optional_njit
def _windowed_matching_dtw(target: np.ndarray, actual: np.ndarray, inversions: np.ndarray, window_size: int):
    length = target.shape[0]
    assert length == actual.shape[0]

    n_cols = window_size * 2 + 1

    forward = np.zeros((length, n_cols))
    backward = np.zeros_like(forward)

    # Note:
    # ti = target index
    # ai = actual index

    # Forward
    for ti in range(length):
        first_ai = max(0, ti - window_size)
        last_ai = min(length - 1, ti + window_size)
        for ai in range(first_ai, last_ai + 1):
            col = ai + window_size - ti

            curr_forward = np.inf

            # Above
            if ti == 0:
                curr_forward = 0
                curr_backward = 1
            elif col != n_cols - 1:
                curr_forward = forward[ti - 1, col + 1]
                curr_backward = 1

            # Diagonal
            if ti != 0 and ai != 0:
                diag = forward[ti - 1, col]
                if diag <= curr_forward:
                    curr_forward = diag
                    curr_backward = 2

            dist = np.abs(target[ti] - actual[ai])
            curr_forward += -np.sqrt(dist) if inversions[ti] else dist

            # Left
            if ai != first_ai:
                left = forward[ti, col - 1]
                if left < curr_forward:
                    curr_forward = left
                    curr_backward = 0

            forward[ti, col] = curr_forward
            backward[ti, col] = curr_backward

    # Backward
    path = []
    ti = length - 1
    ai = length - 1
    while ti != -1:
        col = ai + window_size - ti
        curr_backward = backward[ti, col]
        if curr_backward == 0:
            ai -= 1
        elif curr_backward == 1:
            path.append(ai)
            ti -= 1
        else:
            path.append(ai)
            ti -= 1
            ai -= 1

    return forward[length - 1, window_size], np.array(path[::-1])


def single_specimen_informativeness_eloss(
        model_predict: callable, explanation: Union[TimeExplanation, FreqExplanation], X_bg=None,
        *, n_perturbations=100, model_outputs=None, lowest_first=False, omitter: Omitter = None, return_curves=False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    x_specimen = explanation.x_specimen
    size_x = len(x_specimen)
    exploded_explanation = explanation.explode(divide_spread=False)
    abs_impacts = np.abs(exploded_explanation.impacts)

    if omitter is None:
        if isinstance(explanation, TimeExplanation):
            omitter = TimeSliceOmitter(size_x, exploded_explanation.time_slicing, x_zero_noise, add, X_bg=X_bg)
        elif isinstance(explanation, FreqExplanation):
            omitter = FreqDiceFilterOmitter(size_x, exploded_explanation.time_slicing,
                                            exploded_explanation.freq_slicing)

    if model_outputs is None:
        model_outputs = np.arange(abs_impacts.shape[0])

    curves = []
    for model_output in model_outputs:
        ordered_frags = np.argsort(abs_impacts[model_output])
        if not lowest_first:
            ordered_frags = ordered_frags[::-1]
        Z = np.ones((n_perturbations + 1, omitter.size_z))
        for i, k in enumerate(np.linspace(0, omitter.size_z, min(n_perturbations + 1, omitter.size_z + 1), dtype=int)):
            Z[i, ordered_frags[:k]] = 0
        curves.append(model_predict(omitter.omit(x_specimen, Z))[:, model_output])

    curves = np.array(curves)
    auc = np.trapz(curves, dx=1 / n_perturbations)

    if return_curves:
        return auc, curves
    else:
        return auc
