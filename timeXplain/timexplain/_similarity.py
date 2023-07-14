from typing import Sequence

import numpy as np
from scipy.spatial.distance import correlation as _correlation

from timexplain._utils import unpublished, optional_njit


def correlation(impacts_1: Sequence[float], impacts_2: Sequence[float]) -> float:
    if _is_constant(impacts_1) or _is_constant(impacts_2):
        return 0
    else:
        return 1 - _correlation(impacts_1, impacts_2)


@optional_njit
def _is_constant(arr):
    first = arr[0]
    for v in arr[1:]:
        if v != first:
            return False
    return True


@unpublished
def absolute(impacts_1: Sequence[float], impacts_2: Sequence[float], local_similarities: bool = False):
    assert len(impacts_1) == len(impacts_2)

    quotients = _impact_quotients(impacts_1, impacts_2)
    quotients = np.array([1 / q if abs(q) > 1 else q for q in quotients])

    if local_similarities:
        return np.mean(quotients), quotients
    else:
        return np.mean(quotients)


@unpublished
def structural(impacts_1: Sequence[float], impacts_2: Sequence[float], local_similarities: bool = False):
    assert len(impacts_1) == len(impacts_2)

    all_quotients = _impact_quotients(impacts_1, impacts_2)

    pos_quotient_indices = np.where(all_quotients > 0)[0]
    neg_quotient_indices = np.where(all_quotients < 0)[0]

    pos_local_sims = _do_structural_similarity(all_quotients[pos_quotient_indices])
    neg_local_sims = _do_structural_similarity(abs(all_quotients[neg_quotient_indices]))

    local_sims = []
    for q_idx in range(len(all_quotients)):
        if q_idx in pos_quotient_indices:
            local_sims.append(pos_local_sims.pop(0))
        elif q_idx in neg_quotient_indices:
            local_sims.append(-neg_local_sims.pop(0))
        else:
            local_sims.append(0)

    if local_similarities:
        return np.mean(local_sims), np.array(local_sims)
    else:
        return np.mean(local_sims)


def _impact_quotients(impacts_1, impacts_2):
    def _single_quotient(x):
        x1 = x[0]
        x2 = x[1]
        if x1 == 0 and x2 == 0:
            return 1.0
        elif x2 == 0:
            return 0.0
        else:
            return x1 / x2

    return np.apply_along_axis(_single_quotient, axis=0, arr=np.stack([impacts_1, impacts_2]))


def _do_structural_similarity(quotients):
    def _local_similarity(qsi_idx):
        local_similarity = 1

        lower_qsi_idx = qsi_idx
        upper_qsi_idx = qsi_idx

        while True:
            can_go_lower = lower_qsi_idx > 0
            can_go_higher = upper_qsi_idx < n_quotients - 1

            if can_go_lower and can_go_higher:
                qq_when_going_lower = _quotient_quotient(lower_qsi_idx - 1, upper_qsi_idx)
                qq_when_going_higher = _quotient_quotient(lower_qsi_idx, upper_qsi_idx + 1)
                if qq_when_going_lower > qq_when_going_higher:
                    lower_qsi_idx -= 1
                    local_similarity += qq_when_going_lower
                else:
                    upper_qsi_idx += 1
                    local_similarity += qq_when_going_higher
            else:
                if can_go_lower:
                    lower_qsi_idx -= 1
                elif can_go_higher:
                    upper_qsi_idx += 1
                else:
                    break
                local_similarity += _quotient_quotient(lower_qsi_idx, upper_qsi_idx)

        return local_similarity / n_quotients

    def _quotient_quotient(lower_qsi_idx, upper_qsi_idx):
        lower_quotient = quotients[quotients_sorting_indices[lower_qsi_idx]]
        upper_quotient = quotients[quotients_sorting_indices[upper_qsi_idx]]

        if upper_quotient == 0:  # implies that lower_quotient is also 0
            return 0.0
        else:
            return lower_quotient / upper_quotient

    n_quotients = len(quotients)
    quotients_sorting_indices = np.argsort(quotients)
    return [_local_similarity(np.where(quotients_sorting_indices == q_idx)[0][0]) for q_idx in range(n_quotients)]
