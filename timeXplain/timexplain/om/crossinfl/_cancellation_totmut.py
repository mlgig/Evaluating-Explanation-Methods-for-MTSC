from collections import OrderedDict
from typing import Collection, Sequence, List, Tuple, Mapping, OrderedDict as OrdDict

import numpy as np
from tqdm.auto import tqdm

from timexplain._utils import SingleOrSeq, without, nan_op, unpublished
from timexplain.om._base import OmitterTemplate, Omitter
from timexplain.om._do import Task
from timexplain.om.crossinfl._toolkit import CrossInflToolkit, Mode, ImpossibleSubsetsError, Sample
from timexplain.om.link import Link, identity

TotMutCncls = OrdDict[OmitterTemplate, OrdDict[int, SingleOrSeq[float]]]


@unpublished
class total_mutual_cancellations(Task[TotMutCncls]):

    def __init__(self, fragments: Mapping[OmitterTemplate, Collection[int]],
                 *, max_n_samples=1000, max_n_baseline_zeros=5, min_n_full_rounds=1,
                 link: Link = identity):
        self.fragments = fragments
        self.max_n_samples = max_n_samples
        self.max_n_baseline_zeros = max_n_baseline_zeros
        self.min_n_full_rounds = min_n_full_rounds
        self.link = link

    def run(self, omitters: List[Omitter], model_predict: callable,
            x_specimen: Sequence[float], pbar: bool) -> TotMutCncls:
        cncls, _ = _total_mutual_cancellations(omitters, [self.fragments[om.template] for om in omitters],
                                               model_predict, x_specimen,
                                               self.max_n_samples, self.max_n_baseline_zeros, self.min_n_full_rounds,
                                               self.link,
                                               pbar, collect_samples=False)

        # Convert omitter keys to omitter template keys.
        return OrderedDict([(om.template, om_cncls) for om, om_cncls in cncls.items()])

    def mean(self, all_cncls: List[TotMutCncls]) -> TotMutCncls:
        return OrderedDict([
            (om, OrderedDict([
                (frag, nan_op(np.ma.mean, [cncls[om][frag] for cncls in all_cncls], axis=0))
                for frag in all_cncls[0][om]
            ]))
            for om in all_cncls[0]
        ])


@unpublished
def do_total_mutual_cancellations_collect_samples(omitters: Sequence[Omitter],
                                                  fragments: Sequence[Collection[int]],
                                                  model_predict: callable, x_specimen: Sequence[float] = None,
                                                  *, max_n_samples=1000, max_n_baseline_zeros=5, min_n_full_rounds=1,
                                                  link: Link = identity,
                                                  pbar=True) \
        -> Tuple[OrdDict[Omitter, OrdDict[int, Sequence[float]]],
                 OrdDict[Omitter, OrdDict[int, List[Sample]]]]:
    return _total_mutual_cancellations(omitters, fragments,
                                       model_predict, x_specimen,
                                       max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                       link,
                                       pbar, collect_samples=True)


def _total_mutual_cancellations(omitters, fragments,
                                model_predict, x_specimen,
                                max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                link,
                                pbar, collect_samples):
    # Remove omitters for which the user didn't supply any fragments.
    valid_om_indices = np.nonzero(list(map(len, fragments)))
    omitters = np.asarray(omitters)[valid_om_indices]
    fragments = np.asarray(fragments)[valid_om_indices]

    # Split the samples evenly across all A-omitters.
    max_n_samples //= len(omitters)

    cncls = OrderedDict()
    samples = OrderedDict() if collect_samples else None

    itr = tqdm(list(enumerate(zip(omitters, fragments))), leave=False, disable=not pbar)
    for A_idx, (A_om, A_frags) in itr:
        # Skip A-omitter if the user didn't supply any A-fragments.
        if len(A_frags) == 0:
            continue

        if pbar:
            itr.set_postfix_str(f"cancelled omitter: {A_om}")

        try:
            toolkit = CrossInflToolkit(model_predict, x_specimen,
                                       A_om, A_frags,
                                       max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                       n_B_preds=1)
        except ImpossibleSubsetsError:
            # In this case, there's no possible combination of some n_subsets and subset_size for A-om
            # and the toolkit constructor has already printed a warning.
            continue

        # This BOmission omits all fragments from all other omitters.
        toolkit.add_B_omission(B_oms=without(omitters, A_idx),
                               B_frags=without(fragments, A_idx))

        t_result = toolkit.result(Mode.B_influencer__A_influenced, link,
                                  abs_samples=True, collect_samples=collect_samples)

        cncls[A_om] = OrderedDict([
            # Since we only used one BOmission, we can collapse the BOmission dict.
            (A_frag, next(iter(A_frag_cncls.values())))
            for A_frag, A_frag_cncls in t_result.cross_infls.items()
        ])

        if collect_samples:
            samples[A_om] = OrderedDict([
                (A_frag, next(iter(samples.values())))
                for A_frag, A_frag_samples in t_result.samples.items()
            ])

    return cncls, samples
