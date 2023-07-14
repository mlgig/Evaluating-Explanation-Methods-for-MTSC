from collections import Mapping, OrderedDict
from operator import itemgetter
from typing import Collection, Sequence, List, Mapping as Map, OrderedDict as OrdDict

import numpy as np
from tqdm.auto import tqdm

from timexplain._utils import SingleOrSeq, SingleOrMapping, without, nan_op, unpublished
from timexplain.om._base import OmitterTemplate, Omitter
from timexplain.om._do import Task
from timexplain.om.crossinfl._toolkit import CrossInflToolkit, Mode, ImpossibleSubsetsError, BOmission
from timexplain.om.link import Link, identity

ImpactwiseCnclProg = OrdDict[OmitterTemplate, OrdDict[int, OrdDict[float, SingleOrSeq[float]]]]


@unpublished
class impactwise_cancellation_progression(Task[ImpactwiseCnclProg]):

    def __init__(self, impacts: Map[OmitterTemplate, SingleOrMapping[Sequence[float]]],
                 *, progfrags: Map[OmitterTemplate, Collection[int]] = None,
                 progfrag_impact_gte: float = 0.01, progfrag_impact_lte: float = None,
                 cnclfrag_impact_cumsum_gte: float = None, cnclfrag_impact_cumsum_lte: float = None,
                 max_n_samples=1000, max_n_baseline_zeros=5, min_n_full_rounds=1,
                 link: Link = identity):
        # Collapse multiple impacts (e.g., stemming from multiple model outputs) by maxing.
        # Also convert impact sequences to numpy arrays.
        impacts = {
            om: np.max(list(om_impacts.values()), axis=0)
            if isinstance(om_impacts, Mapping)
            else np.asarray(om_impacts)
            for om, om_impacts in impacts.items()
        }

        if progfrags is not None:
            if progfrag_impact_gte is not None or progfrag_impact_lte is not None:
                raise ValueError("When specifying progfrags directly, the progfrag_impact_gte and progfrag_impact_lte "
                                 "arguments are disregarded. Yet, they are not None. Please set them to None.")
        elif progfrag_impact_gte is not None or progfrag_impact_lte is not None:
            # Collect progfrags whose impacts lie between the specified impact bounds.
            progfrags = {
                om: np.where((om_impacts >= progfrag_impact_gte if progfrag_impact_gte is not None else True) &
                             (om_impacts <= progfrag_impact_lte if progfrag_impact_lte is not None else True))[0]
                for om, om_impacts in impacts.items()
            }
        else:
            raise ValueError("You must either specifiy progfrags directly, or provide bounds using "
                             "progfrag_impact_gte and/or progfrag_impact_lte. If you want to use all fragments "
                             "as progfrags, set progfrag_impact_gte to 0.")

        self.impacts = impacts
        self.progfrags = progfrags
        self.cnclfrag_impact_cumsum_gte = cnclfrag_impact_cumsum_gte
        self.cnclfrag_impact_cumsum_lte = cnclfrag_impact_cumsum_lte
        self.max_n_samples = max_n_samples
        self.max_n_baseline_zeros = max_n_baseline_zeros
        self.min_n_full_rounds = min_n_full_rounds
        self.link = link

    def run(self, omitters: List[Omitter], model_predict: callable,
            x_specimen: Sequence[float], pbar: bool) -> ImpactwiseCnclProg:
        return _impactwise_cncl_prog(omitters, self.impacts, self.progfrags,
                                     self.cnclfrag_impact_cumsum_gte, self.cnclfrag_impact_cumsum_lte,
                                     model_predict, x_specimen,
                                     self.max_n_samples, self.max_n_baseline_zeros, self.min_n_full_rounds,
                                     self.link,
                                     pbar)

    def mean(self, all_cncls: List[ImpactwiseCnclProg]) -> ImpactwiseCnclProg:
        return OrderedDict([
            (om, OrderedDict([
                (frag, OrderedDict([
                    (impact_sum, nan_op(np.ma.mean, [cncls[om][frag][impact_sum] for cncls in all_cncls], axis=0))
                    for impact_sum in all_cncls[0][om][frag]
                ]))
                for frag in all_cncls[0][om]
            ]))
            for om in all_cncls[0]
        ])


def _impactwise_cncl_prog(omitters, impacts, progfrags,
                          cnclfrag_impact_cumsum_gte, cnclfrag_impact_cumsum_lte,
                          model_predict, x_specimen,
                          max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                          link,
                          pbar):
    # Remove omitters for which the user didn't supply any progfrags.
    omitters = [om for om in omitters if len(progfrags[om.template]) != 0]

    # Split the samples evenly across all A-omitters.
    max_n_samples //= len(omitters)

    cnclp = OrderedDict()

    itr = tqdm(list(enumerate(omitters)), leave=False, disable=not pbar)
    for A_idx, A_om in itr:
        if pbar:
            itr.set_postfix_str(f"cancelled omitter: {A_om}")

        # Get all omitters that are not A-om. We call these B-oms.
        B_oms = without(omitters, A_idx)
        # Compile a list of all fragments of all B-oms.
        B_frags = ((om, frag, frag_impact)
                   for om in B_oms
                   for frag, frag_impact in enumerate(impacts[om.template]))
        # Sort this list by descending fragment impact.
        sorted_B_frags = sorted(B_frags, key=itemgetter(2), reverse=True)

        # Generate the B-omissions whose impact cumsum lies in the specified cnclfrag impact cumsum interval.
        # Also store the impact sum of each B-omission for later.
        B_omission_impact_sums = {}
        cnclfrags = {}  # running
        impact_cumsum = 0.0  # running
        for om, frag, frag_impact in sorted_B_frags:
            if frag_impact == 0:
                # All following frag combos would have the same impact_cumsum,
                # which is not allowed, so we ignore them.
                break

            if om not in cnclfrags:
                cnclfrags[om] = []
            cnclfrags[om].append(frag)

            impact_cumsum += frag_impact
            if cnclfrag_impact_cumsum_lte is not None and impact_cumsum > cnclfrag_impact_cumsum_lte:
                break
            elif cnclfrag_impact_cumsum_gte is None or impact_cumsum >= cnclfrag_impact_cumsum_gte:
                # We have found a B-omission that lies in the impact cumsum interval! Add it.
                B_omission_impact_sums[BOmission(cnclfrags.keys(), cnclfrags.values())] = impact_cumsum

        try:
            toolkit = CrossInflToolkit(model_predict, x_specimen,
                                       A_om, progfrags[A_om.template],
                                       max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                       n_B_preds=len(B_omission_impact_sums))
        except ImpossibleSubsetsError:
            # In this case, there's no possible combination of some n_subsets and subset_size for A-om
            # and the toolkit constructor has already printed a warning.
            continue

        # Apply all the found B-omissions.
        for B_omission in B_omission_impact_sums:
            toolkit.add_B_omission(*B_omission)

        t_result = toolkit.result(Mode.B_influencer__A_influenced, link,
                                  abs_samples=True)

        # Convert the omitter key to an omitter template key and convert BOmission keys to sorted impact sum keys.
        cnclp[A_om.template] = OrderedDict([
            (A_frag, OrderedDict([
                (impact_sum, A_frag_cncls[B_omission])
                for B_omission, impact_sum in sorted(((B_omission, B_omission_impact_sums[B_omission])
                                                      for B_omission in A_frag_cncls),
                                                     key=itemgetter(1))
            ]))
            for A_frag, A_frag_cncls in t_result.cross_infls.items()
        ])

    return cnclp
