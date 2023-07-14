import math
from collections import OrderedDict
from dataclasses import dataclass
from itertools import count
from typing import Union, Tuple, Collection, Sequence, List, Mapping, OrderedDict as OrdDict

import numpy as np
from tqdm.auto import tqdm

from timexplain._utils import is_iterable, nan_op, unpublished
from timexplain.om._base import OmitterTemplate, Omitter
from timexplain.om._do import Task
from timexplain.om.crossinfl._toolkit import CrossInflToolkit, Mode, ImpossibleSubsetsError
from timexplain.om.link import Link, identity


@dataclass
class DependencyResult:
    index: OrdDict[OmitterTemplate, OrdDict[int, int]]
    reverse_index: List[Tuple[OmitterTemplate, int]]
    data: np.ndarray  # shape: n_frags x n_frags (x model_outputs)

    def get(self,
            influencer_ot: OmitterTemplate, influencer_frag: int,
            influenced_ot: OmitterTemplate, influenced_frag: int) -> Sequence[float]:
        return self.data[self.index[influencer_ot][influencer_frag],
                         self.index[influenced_ot][influenced_frag]]

    def fragment_at_idx(self, idx: int) -> Tuple[OmitterTemplate, int]:
        return self.reverse_index[idx]

    def format(self, model_outputs: Sequence[int] = None, precision=3):
        def label(om, frag):
            return f"{type(om).__name__[0]}~{frag:0{frag_len}}"

        frag_len = max(len(str(frag)) for _, frag in self.reverse_index)
        label_len = 1 + 1 + frag_len
        row_label_len = max(5, label_len)  # minimum of 5 to support "av b>"

        if model_outputs is None and self.data.ndim != 2:
            model_outputs = list(range(self.data.shape[2]))
        if model_outputs is not None and not is_iterable(model_outputs):
            model_outputs = [model_outputs]

        n_model_outputs = 1 if model_outputs is None else len(model_outputs)
        max_len_before_dp = max(str(float(val)).index(".")
                                for row in self.data for vals in row for val in vals[model_outputs]
                                if not np.isnan(val))
        value_len = max(max_len_before_dp + 1 + precision,
                        math.ceil((label_len - (n_model_outputs - 1)) / n_model_outputs))
        precision = value_len - max_len_before_dp - 1
        cell_len = (value_len + 1) * n_model_outputs - 1
        cell_spacer = " | "

        out = "r\u2193 d\u2192".rjust(label_len) + cell_spacer
        out += cell_spacer.join(
            label(om, frag).center(cell_len) for om, frag in self.reverse_index) + "\n"

        out += "\n".join(
            label(om, frag).rjust(row_label_len) + cell_spacer
            + cell_spacer.join(
                " ".join(
                    "--".center(value_len) if np.isnan(val) else f"{val:{value_len}.{precision}f}"
                    for val in (vals[np.newaxis] if model_outputs is None else vals[model_outputs]))
                for vals in row)
            for row, (om, frag) in zip(self.data, self.reverse_index))

        return out

    def __str__(self):
        return self.format()

    def __repr__(self):
        return type(self).__name__ + ":\n" + self.__str__()


@unpublished
class dependencies(Task[DependencyResult]):

    def __init__(self, fragments: Mapping[OmitterTemplate, Collection[int]],
                 *, max_n_samples=1000, max_n_baseline_zeros=5, min_n_full_rounds=1,
                 mode: Union[str, Mode] = Mode.B_influencer__A_influenced, link: Link = identity,
                 intra_omitter_deps=True):
        self.fragments = fragments
        self.max_n_samples = max_n_samples
        self.max_n_baseline_zeros = max_n_baseline_zeros
        self.min_n_full_rounds = min_n_full_rounds
        self.mode = mode
        self.link = link
        self.intra_omitter_deps = intra_omitter_deps

    def run(self, omitters: List[Omitter], model_predict: callable,
            x_specimen: Sequence[float], pbar: bool) -> DependencyResult:
        return _dependencies(omitters, self.fragments,
                             model_predict, x_specimen,
                             self.max_n_samples, self.max_n_baseline_zeros, self.min_n_full_rounds,
                             self.mode, self.link, self.intra_omitter_deps,
                             pbar)

    def mean(self, all_deps: List[DependencyResult]) -> DependencyResult:
        return DependencyResult(all_deps[0].index, all_deps[0].reverse_index,
                                nan_op(np.ma.mean, [dep.data for dep in all_deps], axis=0))


def _dependencies(omitters, fragments,
                  model_predict, x_specimen,
                  max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                  mode, link, intra_omitter_deps,
                  pbar) -> DependencyResult:
    # Remove omitters for which the user didn't supply any fragments.
    omitters = [om for om in omitters if len(fragments[om.template]) != 0]

    # Convert fragment dict to a list that is aligned with 'omitters'.
    fragments = [fragments[om.template] for om in omitters]

    # Convert string mode to Mode enum.
    mode = Mode(mode)

    # Split the samples evenly across all omitters.
    max_n_samples //= len(omitters)

    # Create indices for the DependencyResult.
    index = OrderedDict()
    offset = 0
    for om, om_frags in zip(omitters, fragments):
        index[om.template] = OrderedDict(zip(om_frags, count(offset)))
        offset += len(om_frags)
    reverse_index = [(om.template, frag) for om, om_frags in zip(omitters, fragments) for frag in om_frags]
    # This will collect the computed dependencies; will be initialized later.
    data = None

    itr = tqdm(list(zip(omitters, fragments)), leave=False, disable=not pbar)
    for A_om, A_frags in itr:
        if pbar:
            itr.set_postfix_str(f"A omitter: {A_om}")

        n_A_frags = len(A_frags)
        n_other_frags = sum(len(om_frags) for om, om_frags in zip(omitters, fragments) if om != A_om)

        def n_B_preds(subset_size):
            return n_other_frags + ((n_A_frags - subset_size) if intra_omitter_deps else 0)

        try:
            toolkit = CrossInflToolkit(model_predict, x_specimen,
                                       A_om, A_frags,
                                       max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                       n_B_preds)
        except ImpossibleSubsetsError:
            # In this case, there's no possible combination of some n_subsets and subset_size for A-om
            # and the toolkit constructor has already printed a warning.
            continue

        if intra_omitter_deps:
            # Calculate dependencies between fragments of the same omitter A-om.
            for B_frag in A_frags:
                toolkit.add_B_omission([A_om], [[B_frag]])

        # Calculate dependencies between all fragments from the omitters A-om and B-om.
        for B_om, B_frags in zip(omitters, fragments):
            if A_om != B_om:
                for B_frag in B_frags:
                    toolkit.add_B_omission([B_om], [[B_frag]])

        t_result = toolkit.result(mode, link)

        # We can create 'data' only now because we don't know 'n_model_outputs' beforehand!
        if data is None:
            n_frags = len(reverse_index)
            example_ci = next(iter(next(iter(t_result.cross_infls.values())).values()))
            data_shape = (n_frags, n_frags) if example_ci.ndim == 0 else (n_frags, n_frags, example_ci.shape[0])
            data = np.full(data_shape, np.nan)

        # Write the toolkit's results into our 'data' matrix.
        for A_frag, B_dict in t_result.cross_infls.items():
            for ([B_om], [[B_frag]]), cross_infl in B_dict.items():
                A_idx = index[A_om.template][A_frag]
                B_idx = index[B_om.template][B_frag]
                if mode == Mode.A_influencer__B_influenced:
                    data[A_idx, B_idx] = cross_infl
                elif mode == Mode.B_influencer__A_influenced:
                    data[B_idx, A_idx] = cross_infl

    return DependencyResult(index, reverse_index, data)
