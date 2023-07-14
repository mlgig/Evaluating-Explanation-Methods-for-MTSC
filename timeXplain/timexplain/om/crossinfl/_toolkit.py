from collections import OrderedDict
from dataclasses import dataclass, field, InitVar
from enum import Enum
from itertools import starmap, combinations
from operator import setitem
from typing import Union, Optional, Callable, Collection, Sequence, Tuple, List, Dict, OrderedDict as OrdDict
from warnings import warn

import numpy as np
from scipy.special import comb

from timexplain._utils import binary_cube, aggregate_predict_deaggregate
from timexplain.om._base import Omitter
from timexplain.om.link import Link, identity, convert_to_link

AFrag = int


@dataclass(frozen=True)
class BOmission:
    oms: Tuple[Omitter] = field(init=False)
    frags: Tuple[Tuple[int]] = field(init=False)

    init_oms: InitVar[Sequence[Omitter]]
    init_frags: InitVar[Sequence[Collection[int]]]

    def __post_init__(self, init_oms, init_frags):
        # This strange setattr is necessary because the class is frozen.
        object.__setattr__(self, "oms", tuple(init_oms))
        object.__setattr__(self, "frags", tuple(tuple(fs) for fs in init_frags))

    def __iter__(self):
        return iter((self.oms, self.frags))


@dataclass
class Cube:
    A_z_ambient: np.ndarray  # shape: size_z
    A_cube_frags: Sequence[AFrag]

    cube_one_masks: np.ndarray  # shape: A_cube_frags x (2^A_cube_frags)
    A_z_cube: np.ndarray  # shape: (2^A_cube_frags) x size_z
    A_x_cube: np.ndarray  # shape: (2^A_cube_frags) x size_x
    A_y_cube: Optional[np.ndarray] = field(default=None)  # shape: (2^A_cube_frags) (x model_outputs)

    B_x_cubes: Dict[BOmission, np.ndarray] = field(default_factory=dict)  # shape: like 'A_x_cube'
    B_y_cubes: Dict[BOmission, np.ndarray] = field(default_factory=dict)  # shape: like 'A_y_cube'
    copy_A_y_cube_to_B_y_cubes: List[BOmission] = field(default_factory=list)

    max_B_preds: int = field(default=-1)
    B_pred_ctr: int = field(default=0)

    def __str__(self):
        return str(self.A_cube_frags)

    def __repr__(self):
        return f"Cube{self.A_cube_frags}"


class Mode(Enum):
    A_influencer__B_influenced = "a"
    B_influencer__A_influenced = "b"


@dataclass(frozen=True)
class Sample:
    z_baseline: np.ndarray  # shape: size_z
    cross_infl: np.ma.masked_array  # shape: () OR model_outputs
    weight: float


@dataclass
class ToolkitResult:
    cross_infls: OrdDict[AFrag, Dict[BOmission, np.ndarray]]  # shape: () OR model_outputs
    samples: Optional[OrdDict[AFrag, Dict[BOmission, List[Sample]]]]


class ImpossibleSubsetsError(ValueError):
    pass


class CrossInflToolkit:
    model_predict: callable
    x_specimen: Sequence[float]
    A_om: Omitter
    cubes: List[Cube]

    def __init__(self,
                 model_predict: callable, x_specimen: Sequence[float],
                 A_om: Omitter, A_frags: Collection[AFrag],
                 max_n_samples: int, max_n_baseline_zeros: int, min_n_full_rounds: int,
                 n_B_preds: Union[int, Callable[[int], int]]):
        self.model_predict = model_predict
        self.x_specimen = x_specimen
        self.A_om = A_om

        n_A_frags = len(A_frags)
        A_ambient_frags = np.delete(np.arange(A_om.size_z), A_frags)
        n_B_preds = n_B_preds if callable(n_B_preds) else lambda _: n_B_preds

        assert n_A_frags != 0, "Empty A_frags neither allowed nor sensible."

        round_plan = _determine_round_plan(A_om, A_frags,
                                           max_n_samples, max_n_baseline_zeros, min_n_full_rounds,
                                           n_B_preds)

        self.cubes = []
        # Fill each round with randomized disjoint subsets of 'A_frags' and create a cube for each subset.
        # Later on, 'add_B_omission()' will initiate calculations for each subset.
        # The aggregator in 'result()' then averages the results in the end.
        for preferred_subset_size, (n_A_ambient_zeros, A_z_ambient) \
                in zip(round_plan, _make_z_ambients(A_om.size_z, A_ambient_frags)):
            A_frag_perm = np.random.permutation(A_frags)
            A_frag_subsets = np.split(A_frag_perm, np.arange(preferred_subset_size, n_A_frags, preferred_subset_size))

            for A_cube_frags in A_frag_subsets:
                subset_size = len(A_cube_frags)

                A_z_cube = np.broadcast_to(A_z_ambient, (2 ** subset_size, A_om.size_z)).copy()
                A_z_cube[:, A_cube_frags] = binary_cube(subset_size)

                cube_one_masks = (A_z_cube[:, A_cube_frags] == 1).T

                # Omit 'x_specimen' using all zs in the cube.
                # Later on, 'result()' will let the model predict ys for those xs.
                A_x_cube = A_om.omit(np.tile(x_specimen, (A_z_cube.shape[0], 1)), A_z_cube)

                self.cubes.append(Cube(A_z_ambient, A_cube_frags, cube_one_masks, A_z_cube, A_x_cube,
                                       max_B_preds=n_B_preds(subset_size)))

    def add_B_omission(self,
                       B_oms: Sequence[Omitter], B_frags: Sequence[Collection[int]],
                       cube: Cube = None):
        assert len(B_oms) == len(B_frags), "Number of B_oms must match number of B_frags."
        assert sum(map(len, B_frags)), "B_frags must contain at least one frag."

        # If no 'cube' is specified, do the B-omission for all cubes.
        if cube is None:
            for cube in self.cubes:
                self.add_B_omission(B_oms, B_frags, cube)
            return

        B_omission = BOmission(B_oms, B_frags)

        # Verify that:
        # - The user hasn't already used up all scheduled B-omissions for the specified cube.
        if cube.B_pred_ctr == cube.max_B_preds:
            raise ValueError(f"Already exhausted all {cube.max_B_preds} scheduled B-omission predictions "
                             f"for cube {cube}.")
        # - The user hasn't already applied the specified B-omission to the specified cube.
        if B_omission in cube.B_y_cubes:
            raise ValueError(f"Already applied B-omission {B_omission} to cube {cube}.")
        # - If the 'A_om' appears in 'B_oms', it does at the position zero.
        if any(B_om == self.A_om for B_om in B_oms[1:]):
            raise ValueError(f"When using the A-om also as a B-om, it must come first in the sequence of B-oms.")
        # - If the 'A_om' appears in 'B_oms', it may not omit fragments that are disabled in the ambient of the cube.
        if B_oms[0] == self.A_om and np.any(cube.A_z_ambient[(B_frags[0],)] == 0):
            raise ValueError(f"When using the A-om also as a B-om, you may not omit fragments that are disabled in "
                             f"the ambient of the cube. Use 'cube.A_z_ambient' to find out.")

        # Take the 'A_x_cube' (time series values) from the specified cube.
        B_x_cube = cube.A_x_cube

        # If the A-om is also a B-om...
        if B_oms[0] == self.A_om:
            B_non_cube_frags = list(set(B_frags[0]).difference(cube.A_cube_frags))
            # If B-frags contains frags that are not in the cube, disable those frags.
            # Do that by computing the first 'B_x_cube' directly from the 'x_specimen', without the
            # intermediary 'A_x_cube'.
            # This way, we avoid a double application of the same omitter.
            if B_non_cube_frags:
                B_z_cube = cube.A_z_cube.copy()
                B_z_cube[:, B_non_cube_frags] = 0
                B_x_cube = B_oms[0].omit(np.tile(self.x_specimen, (B_z_cube.shape[0], 1)), B_z_cube)
            # If, however, B-frag doesn't contain ambient frags, all the required omissions are already
            # present in the cube itself and thus, we don't need to compute anything for this B-om.
            else:
                # So, if no other B-oms are left, we are done here without having computed anything new.
                # The "copy reference" will be resolved later, after 'A_y_cube' has been computed.
                if len(B_oms) == 1:
                    cube.copy_A_y_cube_to_B_y_cubes.append(B_omission)
                    return

            B_oms = B_oms[1:]
            B_frags = B_frags[1:]

        # Omit all remaining specified B-fragments.
        for B_om, B_frags_single in zip(B_oms, B_frags):
            B_z = np.ones(B_om.size_z)
            B_z[(B_frags_single,)] = 0  # wrapper tuple is makes numpy behave correctly when B_frags_single is a tuple
            B_x_cube = B_om.omit(B_x_cube, np.tile(B_z, (B_x_cube.shape[0], 1)))

        cube.B_x_cubes[B_omission] = B_x_cube
        cube.B_pred_ctr += 1

    def result(self, mode: Union[str, Mode], link: Link = identity,
               *, abs_samples=False, collect_samples=False, check_wasted=True) -> ToolkitResult:
        if check_wasted:
            # Make sure that all scheduled B-omission predictions have been exhausted.
            for cube in self.cubes:
                if cube.B_pred_ctr != cube.max_B_preds:
                    raise ValueError(f"For cube {cube}, only {cube.B_pred_ctr} out of the {cube.max_B_preds} scheduled "
                                     f"B-omission predictions have been used. This wastes planned resources.")

        # Collect all x samples that need to be predicated, predict and write the results to the correct variables.
        pred_jobs = []
        for cube in self.cubes:
            if cube.A_y_cube is None:
                pred_jobs.append((cube.A_x_cube, _attrsetter(cube, "A_y_cube")))
            for B_omission, B_x_cube in cube.B_x_cubes.items():
                if B_omission not in cube.B_y_cubes:
                    pred_jobs.append((B_x_cube, _itemsetter(cube.B_y_cubes, B_omission)))
        aggregate_predict_deaggregate(self.model_predict, pred_jobs,
                                      # Due to a multitude of possible reasons, some samples might occur multiple times.
                                      dedup=True)

        # Now that we have computed 'A_y_cube', resolve all "copy A_y_cube references".
        for cube in self.cubes:
            for B_omission in cube.copy_A_y_cube_to_B_y_cubes:
                if B_omission not in cube.B_y_cubes:
                    cube.B_y_cubes[B_omission] = cube.A_y_cube

        # Convert string mode to Mode enum.
        mode = Mode(mode)
        # Convert string or 'shap.common.Link' to function.
        link = convert_to_link(link)

        agg = _ResultAggregator(abs_samples, collect_samples)

        for cube in self.cubes:
            A_y_cube = link(cube.A_y_cube)

            for B_omission, B_y_cube in cube.B_y_cubes.items():
                B_y_cube = link(B_y_cube)

                # If B-oms contains A-om and at least one B-frag is also an A-cube-frag, we need to split the cube
                # in four parts:
                # - A and B enabled
                # - A disabled and B enabled
                # - A enabled and B disabled
                # - A and B disabled
                # In this case, the B-omission information does come both from splitting the cube
                # and from 'B_y_cube'.
                B_cube_frags = set(B_omission.frags[0]).intersection(cube.A_cube_frags)
                if B_omission.oms[0] == self.A_om and B_cube_frags:
                    B_cube_masks = cube.cube_one_masks[np.isin(cube.A_cube_frags, list(B_cube_frags))]
                    B_one_mask = np.all(B_cube_masks, axis=0)
                    B_zero_mask = np.all(~B_cube_masks, axis=0)

                    for A_frag, A_one_mask in zip(cube.A_cube_frags, cube.cube_one_masks):
                        # We need a special case because the generic masking logic in the else branch can't handle it.
                        if A_frag in B_cube_frags:
                            if mode == Mode.A_influencer__B_influenced:
                                B_wo_A_cube_masks = cube.cube_one_masks[
                                    np.isin(cube.A_cube_frags, list(B_cube_frags.difference({A_frag})))]
                                B_one_A_zero_mask = ~A_one_mask & np.all(B_wo_A_cube_masks, axis=0)
                                numers = B_y_cube[B_zero_mask] - A_y_cube[B_one_A_zero_mask]
                                denoms = B_y_cube[B_zero_mask] - A_y_cube[B_one_mask]
                            elif mode == Mode.B_influencer__A_influenced:
                                # By definition, the cross-influence of this A-frag and B-omission combination
                                # must always be -1 anyways.
                                n_model_outputs = None if A_y_cube.ndim == 1 else A_y_cube.shape[1]
                                agg.add_const_sample(A_frag, B_omission,
                                                     size_z=self.A_om.size_z, n_model_outputs=n_model_outputs,
                                                     const=-1)
                                continue
                        # These are the generic cases now.
                        elif mode == Mode.A_influencer__B_influenced:
                            numers = B_y_cube[~A_one_mask & B_zero_mask] - A_y_cube[~A_one_mask & B_one_mask]
                            denoms = B_y_cube[A_one_mask & B_zero_mask] - A_y_cube[A_one_mask & B_one_mask]
                        elif mode == Mode.B_influencer__A_influenced:
                            numers = B_y_cube[~A_one_mask & B_zero_mask] - B_y_cube[A_one_mask & B_zero_mask]
                            denoms = A_y_cube[~A_one_mask & B_one_mask] - A_y_cube[A_one_mask & B_one_mask]

                        agg.add_samples(A_frag, B_omission,
                                        z_baselines=cube.A_z_cube[A_one_mask & B_one_mask],
                                        ci_numers=numers, ci_denoms=denoms)

                # Otherwise, we just need to split it in two parts:
                # - A enabled
                # - A disabled
                # In this case, the B-omission information does solely come from 'B_y_cube'.
                else:
                    for A_frag, A_one_mask in zip(cube.A_cube_frags, cube.cube_one_masks):
                        if mode == Mode.A_influencer__B_influenced:
                            numers = B_y_cube[~A_one_mask] - A_y_cube[~A_one_mask]
                            denoms = B_y_cube[A_one_mask] - A_y_cube[A_one_mask]
                        elif mode == Mode.B_influencer__A_influenced:
                            numers = B_y_cube[~A_one_mask] - B_y_cube[A_one_mask]
                            denoms = A_y_cube[~A_one_mask] - A_y_cube[A_one_mask]

                        agg.add_samples(A_frag, B_omission,
                                        z_baselines=cube.A_z_cube[A_one_mask],
                                        ci_numers=numers, ci_denoms=denoms)

        return agg.bake()


def _determine_round_plan(A_om, A_frags, max_n_samples, max_n_baseline_zeros, min_n_full_rounds, n_B_preds):
    # We calculate the maximum 'max_n_baseline_zeros' that is smaller than or equal to the specified one
    # such that it satisfies the following constraints.
    #
    # 1. The number of samples is bounded by 'max_n_samples'. The number of samples produced by some
    #    individual subset S is given by:
    #      2^|S| * [1 + n_B_preds(|S|)]
    #
    # 2. Subsets must have at least one element:
    #      subset_size >= 1
    #
    # 3. Subsets cannot be larger than the number of available fragments for the A omitter:
    #      subset_size <= n_A_fragments
    #
    # 4. Subsets cannot be larger than 16.
    #    This technical limitation could easily be overcome by changing 'np.uint16' in the code below
    #    to some bigger data type. However, since a subet with 16 fragments already yields a cube of
    #    size 2^16 = 65536, we probably never want to use such a large cube regarding computation time.
    #      subset_size <= 16
    #
    # 5. We must complete at least 'min_n_full_rounds' many full rounds.
    #
    # 6. We must cover each fragment of the A omitter in at least one subset,
    #    i.e., complete at least one full round:
    #      min_n_full_rounds >= 1

    if max_n_baseline_zeros < 0:
        warn(
            f"For A omitter {A_om}, you specified max_n_baseline_zeros={max_n_baseline_zeros}. This is "
            f"nonsensical and likely a bug. Will use max_n_baseline_zeros=0 instead.", stacklevel=3)
        max_n_baseline_zeros = 0

    if min_n_full_rounds < 1:
        warn(
            f"For A omitter {A_om}, you specified min_n_full_rounds={min_n_full_rounds}. This is "
            f"nonsensical and likely a bug. Will use min_n_full_rounds=1 instead.", stacklevel=3)
        min_n_full_rounds = 1

    # Reduce 'max_n_baseline_zeros' if it is so large that it would be capped when computing
    # preferred subset sizes later on.
    ambient_size = A_om.size_z - len(A_frags)
    max_n_baseline_zeros = min(max_n_baseline_zeros, min(16, len(A_frags)) + ambient_size - 1)

    for test_max_n_baseline_zeros in range(max_n_baseline_zeros, -1, -1):
        round_plan = _determine_single_round_plan(A_om, A_frags, max_n_samples, test_max_n_baseline_zeros, n_B_preds)
        if len(round_plan) >= min_n_full_rounds:
            # If 'min_n_full_rounds' is satisfied, we have found a solution.
            return round_plan
        elif test_max_n_baseline_zeros == 0 and len(round_plan) != 0:
            # In case no solution can be found that satisfies 'min_n_full_rounds', but there's a solution whose
            # n_full_ambients >= 1, i.e., which completes at least one full round, emit a warning and return
            # that solution.
            warn(
                f"For A omitter {A_om}, the specified min_n_full_rounds={min_n_full_rounds} is too high "
                f"resp. the specified max_n_samples={max_n_samples} is too low for any max_n_baseline_zeros. Thus, "
                f"for this omitter, min_n_full_rounds will be shrinked to {len(round_plan)}. To fix this "
                f"yourself, either decrease min_n_full_rounds manually or increase max_n_samples.", stacklevel=3)
            return round_plan

    warn(
        f"For A omitter {A_om}, the specified max_n_samples={max_n_samples} is too low for any possible "
        f"value of max_n_baseline_zeros, even while ignoring min_n_full_rounds. To fix this, please increase "
        f"max_n_samples.", stacklevel=3)
    raise ImpossibleSubsetsError


def _determine_single_round_plan(A_om, A_frags, max_n_samples, max_n_baseline_zeros, n_B_preds):
    n_A_frags = len(A_frags)
    ambient_size = A_om.size_z - n_A_frags

    round_plan = []
    remaining_n_samples = max_n_samples

    # Iterate over all possible rounds.
    # Each round features a different combination of ambient zeros.
    # We are only interested in the number of ambient zeros. We simulate the sequence like this:
    # - 1 round: 0 ambient zeros
    # - ('ambient_size' choose 1) rounds: 1 ambient zeros
    # - ('ambient_size' choose 2) rounds: 2 ambient zeros
    # ...
    rounds = (n for n in range(ambient_size + 1) for _ in range(comb(ambient_size, n, exact=True)))
    for n_ambient_zeros in rounds:
        # Compute the largest possible preferred subset size considering:
        # - the subset size limit of 16
        # - the total number of A fragments, which also bounds the subset size
        # - the given 'max_n_baseline_zeros' and the given 'n_ambient_zeros'
        max_pref_subset_size = max(1, min(16, n_A_frags, max_n_baseline_zeros - n_ambient_zeros + 1))

        # First, try the maximum possible preferred subset size. If it doesn't work, back off to smaller ones.
        for attempt, preferred_subset_size in enumerate(range(max_pref_subset_size, 0, -1)):
            # Simulate using as many subsets as possible with the preferred size and then using one smaller subset
            # to fill up the remaining A fragments.
            # Compute how many samples that would require.
            n_preferred_subsets, remainder_subset_size = divmod(n_A_frags, preferred_subset_size)
            required_n_samples = \
                (2 ** preferred_subset_size * (1 + n_B_preds(preferred_subset_size)) * n_preferred_subsets) + \
                (2 ** remainder_subset_size * (1 + n_B_preds(remainder_subset_size)) * 1
                 if remainder_subset_size else 0)

            if required_n_samples <= remaining_n_samples:
                # We have enough remaining samples for the current preferred subset size, so add it to the plan.
                round_plan.append(preferred_subset_size)

                if attempt == 0:
                    # If the remaining samples are enough for the _maximum possible_ preferred subset size,
                    # continue with the next round.
                    remaining_n_samples -= required_n_samples
                    break
                else:
                    # If the remaining samples are only enough for a smaller preferred subset size,
                    # there's little hope in continuing with more rounds, so we have reached the final round.
                    # Return the plan.
                    return round_plan
            elif preferred_subset_size == 1:
                # If the remaining samples aren't even enough for the smallest possible preferred subset size (1),
                # we can't possibly execute the current round. So return the plan up to the previous round.
                return round_plan

    # Note that if at this point 'remaining_n_samples' > 0, the inferred maximum possible max_n_baseline_zeros
    # is so low that only not all specified 'max_n_samples' can been used. We don't emit a warning because this
    # condition can happen quite regularly whenever 'A_frags' is small and the ambient is also small.

    return round_plan


def _make_z_ambients(size_z, ambient_frags):
    for n in range(len(ambient_frags) + 1):
        for zero_indices in combinations(ambient_frags, n):
            z = np.ones(size_z)
            z[(zero_indices,)] = 0
            yield n, z


def _attrsetter(obj, attr):
    return lambda value: setattr(obj, attr, value)


def _itemsetter(obj, index):
    return lambda value: setitem(obj, index, value)


class _ResultAggregator:
    abs_samples: bool
    # Note: CI is an abbr. for cross-influence!
    ci_sums: Dict[AFrag, Dict[BOmission, np.ndarray]]
    weight_sums: Dict[AFrag, Dict[BOmission, np.ndarray]]

    collect_samples: bool
    samples: Dict[AFrag, Dict[BOmission, List[Sample]]]

    def __init__(self, abs_samples, collect_samples):
        self.abs_samples = abs_samples
        self.ci_sums = {}
        self.weight_sums = {}

        self.collect_samples = collect_samples
        self.samples = {} if collect_samples else None

    def add_samples(self,
                    A_frag: AFrag, B_omission: BOmission,
                    z_baselines: np.ndarray,  # shape: samples x size_z
                    ci_numers: np.ndarray,  # shape: samples (x model_outputs)
                    ci_denoms: np.ndarray):  # shape: samples (x model_outputs)
        # First, compute the cross-influence values where the denominator is not too small.
        small_denom_mask = np.abs(ci_denoms) < 0.001  # shape: samples (x model_outputs)
        cis = ci_numers / np.ma.array(ci_denoms, mask=small_denom_mask)  # shape: samples (x model_outputs)

        if self.abs_samples:
            cis = np.ma.abs(cis)

        cis -= 1

        # Next, compute the weight for each sample.
        # Note: z_baselines are the zs with both the A-frag (and if one B-om is the A-om, those B-frags) still set to 1!
        size_z = z_baselines.shape[1]
        weights = np.exp(np.square(np.sum(z_baselines, axis=1) - size_z) * -16 / np.square(size_z))  # shape: samples

        self._add_sample_cis(A_frag, B_omission, z_baselines, cis, weights)

    def add_const_sample(self,
                         A_frag: AFrag, B_omission: BOmission,
                         size_z: int, n_model_outputs: Optional[int], const: float):
        cis_shape = (1,) if n_model_outputs is None else (1, n_model_outputs)
        self._add_sample_cis(A_frag, B_omission,
                             z_baselines=np.ones((1, size_z)),
                             cis=np.ma.array(np.full(cis_shape, const)),
                             weights=np.ones(1))

    def _add_sample_cis(self, A_frag, B_omission, z_baselines, cis, weights):
        # If there is more than one model output, broadcast the weights such that it matches the cis array,
        # i.e., duplicate the weights for each model output.
        # shape: samples (x model_outputs)
        if cis.ndim == 2:
            broadcast_weights = np.ma.array(np.broadcast_to(weights[:, np.newaxis], cis.shape), mask=cis.mask)
        else:
            broadcast_weights = weights
        # Mask the resulting array in the same way cis is masked.
        broadcast_weights = np.ma.array(broadcast_weights, mask=cis.mask)

        # Aggregate cross-influences and weights per model output.
        agg_cis = np.ma.sum(broadcast_weights * cis, axis=0)  # shape: () OR model_outputs
        agg_weights = np.ma.sum(broadcast_weights, axis=0)  # shape: () OR model_outputs

        if A_frag not in self.ci_sums:
            self.ci_sums[A_frag] = {}
            self.weight_sums[A_frag] = {}

        if B_omission not in self.ci_sums[A_frag]:
            self.ci_sums[A_frag][B_omission] = agg_cis
            self.weight_sums[A_frag][B_omission] = agg_weights
        else:
            # We have to stack() before summing because if either 'agg_cis' or 'self.ci_sums...' is a stand-alone
            # MaskedConstant, np.ma.sum() raises a false warning.
            self.ci_sums[A_frag][B_omission] = np.ma.sum(
                np.ma.stack([agg_cis, self.ci_sums[A_frag][B_omission]]), axis=0)
            self.weight_sums[A_frag][B_omission] = np.ma.sum(
                np.ma.stack([agg_weights, self.weight_sums[A_frag][B_omission]]), axis=0)

        # If the user requested it (e.g., for debugging), also store the individual sample zs, cis and weights.
        if self.collect_samples:
            if A_frag not in self.samples:
                self.samples[A_frag] = {}
            if B_omission not in self.samples[A_frag]:
                self.samples[A_frag][B_omission] = []
            self.samples[A_frag][B_omission].extend(starmap(Sample, zip(z_baselines, cis, weights)))

    def bake(self) -> ToolkitResult:
        cross_infls = OrderedDict([
            (A_frag, {
                B_omission: self._nan_filled(self.ci_sums[A_frag][B_omission] / self.weight_sums[A_frag][B_omission])
                for B_omission in self.ci_sums[A_frag]
            })
            for A_frag in sorted(self.ci_sums)
        ])

        if self.collect_samples:
            samples = OrderedDict([
                (A_frag, self.samples[A_frag])
                for A_frag in sorted(self.samples)
            ])
        else:
            samples = None

        return ToolkitResult(cross_infls, samples)

    @staticmethod
    def _nan_filled(arr):
        return arr.filled(np.nan) if isinstance(arr, np.ma.masked_array) else arr
