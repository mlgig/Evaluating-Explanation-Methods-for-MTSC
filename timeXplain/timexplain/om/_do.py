from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TypeVar, Generic, Union, Sequence, List, Tuple, OrderedDict as OrdDict

import numpy as np
from tqdm.auto import tqdm, trange

from timexplain._utils import SingleOrList, unpublished
from timexplain.om._base import Omitter

Y = TypeVar("Y")
R = TypeVar("R")


class Task(Generic[R], ABC):

    @abstractmethod
    def run(self, omitters: SingleOrList[Omitter], model_predict: callable,
            x_specimen: Sequence[float], pbar: bool) -> R:
        raise NotImplementedError

    @abstractmethod
    def mean(self, results: List[R]) -> R:
        raise NotImplementedError


@unpublished
def do(task: Task[R], omitter: Union[Omitter, Sequence[Omitter]],
       model_predict: callable,
       x_specimen: Sequence[float], X_bg=None,
       *, n_builds=10,
       pbar=True) -> R:
    x_specimen = np.asarray(x_specimen)
    X_bg = None if X_bg is None else np.asarray(X_bg)

    # If the omitter is deterministic, doing multiple builds serves no purpose.
    try:
        deterministic = all(om.deterministic for om in omitter)
    except TypeError:  # not iterable
        deterministic = omitter.deterministic
    if deterministic:
        n_builds = 1

    results = []
    for _ in trange(n_builds, leave=False, disable=not pbar or n_builds == 1):
        try:
            omitter = [om.refine(X_bg=X_bg) if om.requires_X_bg else om for om in omitter]
        except TypeError:  # not iterable
            omitter = omitter.refine(X_bg=X_bg) if omitter.requires_X_bg else omitter
        results.append(task.run(omitter, model_predict, x_specimen, pbar))

    return task.mean(results)


@unpublished
def do_clf(task: Task[R], omitter: Union[Omitter, Sequence[Omitter]],
           model_predict: callable,
           x_specimen: Sequence[float], X_bg, y_bg: Sequence[Y],
           *, max_n_bgclasses=10, n_builds_per_bgclass=10,
           pbar=True) -> Tuple[R, OrdDict[Y, R]]:
    # If no omitter cares about the X_bg anyways, we don't need to partition the env.
    try:
        requires_X_bg = any(ot.requires_X_bg for ot in omitter)
    except TypeError:  # not iterable
        requires_X_bg = omitter.requires_X_bg
    if not requires_X_bg:
        raise ValueError("None of the supplied omitters require X_bg. Thus, there is no reason to use "
                         "do_clf() over simple do().")

    if X_bg is None or y_bg is None:
        raise ValueError("do_clf() needs X_bg and y_bg. If you cannot supply of these, use vanilla do() instead.")

    x_specimen = np.asarray(x_specimen)
    X_bg = np.asarray(X_bg)
    y_bg = np.asarray(y_bg)

    bgclasses = np.unique(y_bg)
    if len(bgclasses) > max_n_bgclasses:
        bgclasses = np.random.choice(bgclasses, max_n_bgclasses, replace=False)

    results_per_bgclass = OrderedDict()
    for y in tqdm(bgclasses, leave=False, disable=not pbar or len(bgclasses) == 1):
        class_X_bg = X_bg[y_bg == y]
        results_per_bgclass[y] = do(task, omitter, model_predict,
                                    x_specimen, class_X_bg,
                                    n_builds=n_builds_per_bgclass, pbar=pbar)

    result_means = task.mean(list(results_per_bgclass.values()))

    return result_means, results_per_bgclass
