from typing import Union

import numpy as np
from shap.common import Link as _ShapLink, IdentityLink as _ShapIdentityLink, LogitLink as _ShapLogitLink

from timexplain._utils import unpublished

Link = Union[callable, str, _ShapLink]


def identity(x):
    return x


def logit(x):
    return np.log(x / (1 - x))


@unpublished
def stretched_logit(a=0.99999999):
    assert 0 < a < 1

    def f(x):
        x2 = a * (x - 0.5) + 0.5
        return np.log(x2 / (1 - x2))

    return f


@unpublished
def linext_logit(t=0.05):
    assert 0 < t < 0.5

    def f(x):
        y = np.empty_like(x)

        a = np.log(t / (1 - t))
        b = 1 / (t * (1 - t))

        mask = (t <= x) & (x <= 1 - t)
        y[mask] = np.log(x[mask] / (1 - x[mask]))

        y[x < t] = (x[x < t] - t) * b + a
        y[x > 1 - t] = (x[x > 1 - t] - (1 - t)) * b - a

        return y

    return f


def convert_to_link(link: Link) -> callable:
    if callable(link):
        return link
    elif link == "identity":
        return identity
    elif link == "logit":
        return logit
    elif isinstance(link, _ShapLink) and callable(getattr(link, "f", None)):
        return link.f
    else:
        raise ValueError(f"Link object must be callable, 'identity', 'logit', or a subclass of shap.common.Link. "
                         f"{link} is neither of those.")


def convert_to_shap_link(link: Link) -> _ShapLink:
    if isinstance(link, _ShapLink):
        return link
    elif link == "identity" or link == identity:
        return _ShapIdentityLink()
    elif link == "logit" or link == logit:
        return _ShapLogitLink()
    elif callable(link):
        return _ShapLinkWrapper(link)
    else:
        raise ValueError(f"Link object must be callable, 'identity', 'logit', or a subclass of shap.common.Link. "
                         f"{link} is neither of those.")


class _ShapLinkWrapper(_ShapLink):
    def __init__(self, link: callable):
        super().__init__()
        self.link = link

    def __str__(self):
        return self.link.__name__

    def f(self, x):
        return self.link(x)

    @staticmethod
    def finv(x):
        raise NotImplementedError
