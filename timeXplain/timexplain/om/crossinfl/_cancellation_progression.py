from collections import OrderedDict
from functools import reduce
from itertools import chain, combinations, zip_longest
from typing import Collection, Sequence, List, Tuple, NamedTuple, Mapping, Dict, OrderedDict as OrdDict

import numpy as np
from tqdm.auto import tqdm

from timexplain._utils import SingleOrList, nan_op, binary_cube, unpublished
from timexplain.om._base import OmitterTemplate, Omitter
from timexplain.om._do import Task
from timexplain.om.crossinfl._dependency import dependencies, DependencyResult
from timexplain.om.crossinfl._toolkit import CrossInflToolkit, Mode, ImpossibleSubsetsError, BOmission
from timexplain.om.link import Link, identity

CnclProg = OrdDict[OmitterTemplate, OrdDict[int, np.ndarray]]
BOmitters = Tuple[Omitter, ...]
BOmissionProg = List[Tuple[BOmission, float]]


class CnclProgResult(NamedTuple):
    deps: DependencyResult
    cnclp: CnclProg


@unpublished
class cancellation_progression(Task[CnclProgResult]):

    def __init__(self, fragments: Mapping[OmitterTemplate, Collection[int]],
                 *, max_n_samples=2000, max_n_baseline_zeros=5, min_n_full_rounds=1,
                 link: Link = identity,
                 intra_omitter_deps=False):
        self.fragments = fragments
        self.max_n_samples = max_n_samples // 2
        self.max_n_baseline_zeros = max_n_baseline_zeros
        self.min_n_full_rounds = min_n_full_rounds
        self.link = link

        self.dep_task = dependencies(fragments, max_n_samples=max_n_samples // 2,
                                     max_n_baseline_zeros=max_n_baseline_zeros, min_n_full_rounds=min_n_full_rounds,
                                     mode=Mode.B_influencer__A_influenced, link=link,
                                     intra_omitter_deps=intra_omitter_deps)

    def run(self, omitters: SingleOrList[Omitter], model_predict: callable,
            x_specimen: Sequence[float],
            pbar: bool) -> CnclProgResult:
        return _cancellation_progression(omitters, self.fragments,
                                         model_predict, x_specimen,
                                         self.dep_task,
                                         self.max_n_samples, self.max_n_baseline_zeros, self.min_n_full_rounds,
                                         self.link,
                                         pbar)

    def mean(self, results: List[CnclProgResult]) -> CnclProgResult:
        return CnclProgResult(
            self.dep_task.mean([result.deps for result in results]),
            OrderedDict([
                (om, OrderedDict([
                    (frag, nan_op(np.ma.mean, [result.cnclp[om][frag] for result in results], axis=0))
                    for frag in results[0].cnclp[om]
                ]))
                for om in results[0].cnclp
            ])
        )


def _cancellation_progression(omitters, fragments,
                              model_predict, x_specimen,
                              dep_task,
                              max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                              link,
                              pbar) -> CnclProgResult:
    # Remove omitters for which the user didn't supply any fragments.
    omitters = [om for om in omitters if len(fragments[om.template]) != 0]

    # Compute dependencies.
    deps = dep_task.run(omitters, model_predict, x_specimen, pbar)
    n_model_outputs = None if deps.data.ndim == 2 else deps.data.shape[2]

    # Use the dependencies to compute cancellation progressions and their B-omissions.
    B_omission_progs = _compute_B_omission_progs(omitters, fragments, deps, n_model_outputs)

    # Split the samples evenly across all omitters.
    max_n_samples //= len(omitters)

    cnclp = OrderedDict()

    itr = tqdm(list(enumerate(omitters)), leave=False, disable=not pbar)
    for A_idx, A_om in itr:
        if pbar:
            itr.set_postfix_str(f"cancelled omitter: {A_om}")

        # For each A-fragment, collect all B-omissions that appear somewhere in its B-omission progressions
        # and omit at least 2 fragments. We exclude the B-omissions that omit only 1 fragment because those
        # B-omissions have already been computed by the dependency task.
        B_omissions_per_A_frag = {}
        for A_frag, d1 in B_omission_progs[A_om].items():
            B_omissions_per_A_frag[A_frag] = set()
            for d2 in d1.values():
                for d3 in d2:
                    if n_model_outputs is None:
                        B_omission = d3[0]
                        if sum(map(len, B_omission.frags)) > 1:
                            B_omissions_per_A_frag[A_frag].add(B_omission)
                    else:
                        for B_omission, _ in d3:
                            if sum(map(len, B_omission.frags)) > 1:
                                B_omissions_per_A_frag[A_frag].add(B_omission)

        # Compute how many B_omissions there are in total, summing up over all A-fragments.
        n_total_B_omissions = len(reduce(set.union, B_omissions_per_A_frag.values()))
        # Compute how many B-omissions any A-fragment has at most.
        max_B_omissions_per_A_frag = max(map(len, B_omissions_per_A_frag.values()))

        def n_B_preds(subset_size):
            # This is an estimate that never underestimates.
            return min(subset_size * max_B_omissions_per_A_frag, n_total_B_omissions)

        try:
            toolkit = CrossInflToolkit(model_predict, x_specimen,
                                       A_om, fragments[A_om.template],
                                       max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                       n_B_preds)
        except ImpossibleSubsetsError:
            # In this case, there's no possible combination of some n_subsets and subset_size for A-om
            # and the toolkit constructor has already printed a warning.
            continue

        # For each cube, apply the collected B-omissions that are relevant for any A-fragment in the cube.
        for cube in toolkit.cubes:
            B_omissions = set(chain.from_iterable(B_omissions_per_A_frag[A_frag] for A_frag in cube.A_cube_frags))
            for B_omission in B_omissions:
                toolkit.add_B_omission(*B_omission, cube=cube)

        t_result = toolkit.result(Mode.B_influencer__A_influenced, link,
                                  check_wasted=False)

        cnclp[A_om.template] = OrderedDict([
            (A_frag, np.array(_for_all_model_outputs(_merge_cross_infls_to_cnclp, n_model_outputs,
                                                     B_omission_progs=B_omission_progs[A_om][A_frag],
                                                     cross_infls=B_dict)))
            for A_frag, B_dict in t_result.cross_infls.items()
        ])

    return CnclProgResult(deps, cnclp)


def _for_all_model_outputs(func, n_model_outputs, **kwargs):
    if n_model_outputs is None:
        return func(model_output=None, **kwargs)
    else:
        return [func(model_output=model_output, **kwargs)
                for model_output in range(n_model_outputs)]


def _compute_B_omission_progs(omitters, fragments, deps, n_model_outputs) \
        -> Dict[Omitter, Dict[int, Dict[BOmitters, SingleOrList[BOmissionProg]]]]:
    B_omission_progs = {}

    # Initialize the nested dict structure.
    for A_om in omitters:
        B_omission_progs[A_om] = {}
        for A_frag in fragments[A_om.template]:
            B_omission_progs[A_om][A_frag] = {}

    # Iterate over all subsets of 'omitters' where at least one omitter is missing.
    for B_oms in chain.from_iterable(combinations(omitters, n) for n in range(1, len(omitters))):
        # Create 'all_subset_masks'.
        n_B_frags = sum(len(fragments[B_om.template]) for B_om in B_oms)
        cube = binary_cube(n_B_frags)
        row_sums = np.sum(cube, axis=1)
        all_subset_masks = [cube[row_sums == subset_size] for subset_size in range(1, n_B_frags + 1)]

        # Iterate over all omitters that are not in the current B-om set.
        for A_om in set(omitters).difference(B_oms):
            for A_frag in fragments[A_om.template]:
                B_omission_progs[A_om][A_frag][B_oms] = _for_all_model_outputs(
                    _compute_B_omission_prog, n_model_outputs,
                    deps=deps, A_om=A_om, A_frag=A_frag, B_oms=B_oms,
                    all_subset_masks=all_subset_masks)

    return B_omission_progs


def _compute_B_omission_prog(deps, A_om, A_frag, B_oms, model_output, all_subset_masks) -> BOmissionProg:
    # Get the dependencies dependencies where (A_om, A_frag) is influenced
    # and all fragments of all B_oms are the influencers.
    indices_into_dep_data_for_B_frags = [frag for B_om in B_oms for frag in deps.index[B_om.template].values()]
    A_dep_slice = deps.data[indices_into_dep_data_for_B_frags, deps.index[A_om.template][A_frag]]
    if model_output is not None:
        A_dep_slice = A_dep_slice[..., model_output]

    B_omission_prog = []

    for subset_size, subset_masks in enumerate(all_subset_masks, 1):
        subsets = np.where(subset_masks, A_dep_slice, 0)
        est_subset_cncls = np.sum(subsets, axis=1)

        # Find the index of the subset whose estimated cancellation is closest to -1.
        best_subset_idx = np.argmin(np.abs(est_subset_cncls + 1))
        # Find that subset's estimated cancellation.
        best_est_subset_cncl = est_subset_cncls[best_subset_idx]

        B_omission = OrderedDict()
        for idx_into_subset_mask in np.where(subset_masks[best_subset_idx] == 1)[0]:
            # Convert the index into the subset mask to an index into 'deps.data'.
            idx_into_dep_data = indices_into_dep_data_for_B_frags[idx_into_subset_mask]
            # Get the omitter and fragment behind that index and put them into the B-omission.
            B_omitter, B_frag = deps.fragment_at_idx(idx_into_dep_data)
            B_om = next(B_om for B_om in B_oms if B_om.template == B_omitter)
            if B_om not in B_omission:
                B_omission[B_om] = []
            B_omission[B_om].append(B_frag)

        # Create a BOmission object and put it into the sequence.
        B_omission_prog.append((BOmission(B_omission.keys(), B_omission.values()), best_est_subset_cncl))

    return B_omission_prog


def _merge_cross_infls_to_cnclp(B_omission_progs: Dict[BOmitters, List[BOmissionProg]],
                                cross_infls: Dict[BOmission, np.ndarray], model_output):
    mo_cnclp = [[0, 0, len(B_omission_progs)]]  # first step: nothing cancelled so far

    for pairs in zip_longest(
            *(B_omission_progs[B_oms] if model_output is None else B_omission_progs[B_oms][model_output]
              for B_oms in B_omission_progs)):
        non_none_pairs = [pair for pair in pairs if pair is not None]
        best_error, best = None, np.nan
        for B_omission, est_cncl in non_none_pairs:
            # Recall that we don't compute B-omissions with only one frag a second time because they have
            # already been computed by the dependency task. Thus, their true cancellation is equal to
            # their estimated one.
            if B_omission in cross_infls:
                true_cncl = cross_infls[B_omission] if model_output is None else cross_infls[B_omission][model_output]
            else:
                true_cncl = est_cncl
            error = abs(est_cncl - true_cncl)
            if best_error is None or error < best_error:
                best_error = error
                best = [est_cncl, true_cncl, len(non_none_pairs)]
        mo_cnclp.append(best)

    return mo_cnclp
