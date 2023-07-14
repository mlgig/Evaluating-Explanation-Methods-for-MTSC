from collections import OrderedDict
from typing import Any, Union, Sequence, List, Tuple, OrderedDict as OrdDict

import numpy as np
import shap
from tqdm.auto import tqdm, trange

from timexplain._explainer import Explainer
from timexplain._explanation import Explanation
from timexplain._utils import is_iterable
from timexplain.om._base import Omitter
from timexplain.om.link import Link, identity, convert_to_shap_link


class KernelShapExplainer(Explainer[Explanation]):
    omitter: Omitter
    model_predict: callable
    X_bg: np.ndarray
    y_bg: np.ndarray
    bgcs: bool
    max_n_bgclasses: int
    n_builds: int
    n_samples: int
    link: shap.common.Link
    l1_reg: str
    pbar: bool

    def __init__(self, omitter: Omitter, model_predict: callable, X_bg=None, y_bg: Sequence = None,
                 *, bgcs=True, max_n_bgclasses=10, n_builds=10, n_samples=1000,
                 link: Link = identity, l1_reg="aic", pbar=True):
        self.omitter = omitter
        self.model_predict = model_predict
        self.X_bg = None if X_bg is None else np.asarray(X_bg)
        self.y_bg = None if y_bg is None else np.asarray(y_bg)
        self.n_samples = n_samples
        self.n_builds = n_builds
        self.max_n_bgclasses = max_n_bgclasses
        self.bgcs = bgcs
        self.link = convert_to_shap_link(link)
        self.l1_reg = l1_reg
        self.pbar = pbar

    def _explain(self, X_specimens):
        X_specimens = tqdm(X_specimens, leave=False, disable=not self.pbar)

        if self.bgcs and self.omitter.requires_X_bg and self.X_bg is not None:
            if self.y_bg is None:
                raise ValueError("When using background set class splitting, you must supply y_bg. "
                                 "If you do not want to use background set class splitting, set bgcs=False.")
            return [self.omitter.create_explanation(x_specimen, None, self._impacts_per_bgclass(x_specimen)[0])
                    for x_specimen in X_specimens]
        else:
            return [self.omitter.create_explanation(x_specimen, None, self._impacts(x_specimen, self.X_bg))
                    for x_specimen in X_specimens]

    def explain_per_bgclass(self, X_specimens) \
            -> Union[Tuple[Explanation, OrdDict[Any, Explanation]],
                     List[Tuple[Explanation, OrdDict[Any, Explanation]]]]:
        if not self.bgcs:
            raise ValueError("Background set class splitting is not enabled.")
        if self.X_bg is None or self.y_bg is None:
            raise ValueError("When using background set class splitting, you must supply both X_bg and y_bg.")
        if not self.omitter.requires_X_bg:
            raise ValueError("Background set class splitting is not sensible because the omitter "
                             "does not require a background set.")

        return_single = not is_iterable(X_specimens[0])
        if return_single:
            X_specimens = X_specimens[np.newaxis]

        results = []
        for x_specimen in tqdm(X_specimens, leave=False, disable=not self.pbar):
            impacts, impacts_per_bgclass = self._impacts_per_bgclass(x_specimen)
            results.append((self.omitter.create_explanation(x_specimen, None, impacts),
                            OrderedDict([
                                (y, self.omitter.create_explanation(x_specimen, None, impacts))
                                for y, impcts in impacts_per_bgclass.items()
                            ])))

        return results[0] if return_single else results

    def _impacts(self, x_specimen, X_bg):
        x_specimen = np.asarray(x_specimen)

        # If the omitter is deterministic, doing multiple builds serves no purpose.
        n_builds = 1 if self.omitter.deterministic else self.n_builds

        impacts = []
        for _ in trange(n_builds, leave=False, disable=not self.pbar or n_builds == 1):
            omitter = self.omitter
            if omitter.requires_X_bg and X_bg is not None:
                omitter = self.omitter.refine(X_bg=X_bg)

            def Z2Y(Z):
                return self.model_predict(omitter.omit(x_specimen, Z))

            run_impacts = shap.KernelExplainer(Z2Y, omitter.z_empty(dims=2), self.link) \
                .shap_values(omitter.z_specimen(), nsamples=self.n_samples, l1_reg=self.l1_reg)
            impacts.append(run_impacts)

        impacts = np.mean(impacts, axis=0)
        return impacts

    def _impacts_per_bgclass(self, x_specimen):
        bgclasses = np.unique(self.y_bg)
        if len(bgclasses) > self.max_n_bgclasses:
            bgclasses = np.random.choice(bgclasses, self.max_n_bgclasses, replace=False)

        impacts_per_bgclass = OrderedDict()
        for y in tqdm(bgclasses, leave=False, disable=not self.pbar or len(bgclasses) == 1):
            class_X_bg = self.X_bg[self.y_bg == y]
            impacts_per_bgclass[y] = self._impacts(x_specimen, class_X_bg)

        impacts = np.mean(list(impacts_per_bgclass.values()), axis=0)
        return impacts, impacts_per_bgclass
