from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from timexplain._utils import nan_op, unpublished
from timexplain.om.crossinfl import DependencyResult
from timexplain.plot._matrix import matrix as plot_matrix


@unpublished
def dependencies(deps: DependencyResult, model_output: int = None,
                 *, cmap="RdBu_r", fontcolor_thresh: float = None,
                 text_len: int = None, omit_leading_zero: bool = None, trailing_zeros: bool = None,
                 grid: bool = None, angle_left: bool = None, cbar: bool = None,
                 cbar_label: str = None,
                 ax: plt.Axes = None, figsize: Tuple[float, float] = None, cellsize: float = None, title: str = None):
    if model_output is None and deps.data.ndim != 2:
        n_model_outputs = deps.data.shape[2]
        if n_model_outputs != 1:
            raise ValueError("When plotting dependencies, the model_output argument can only be omitted when "
                             "there is only one model output in the DependencyResult. However, this Dependency "
                             f"stores {n_model_outputs} model outputs. So, please supply model_output.")
        else:
            model_output = 0

    plot_kwargs = {
        "cmap": cmap, "fontcolor_thresh": fontcolor_thresh,
        "text_len": text_len, "omit_leading_zero": omit_leading_zero, "trailing_zeros": trailing_zeros,
        "grid": grid, "angle_left": angle_left, "cbar": cbar, "cbar_label": cbar_label,
        "ax": ax, "figsize": figsize, "cellsize": cellsize, "title": title}
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

    if model_output is None:
        values = deps.data
    else:
        values = deps.data[:, :, model_output]

    labels = [f"{type(om).__name__[0]}~{frag}" for om, frag in deps.reverse_index]
    extremum = max(1, nan_op(np.ma.max, np.abs(values)))
    plot_matrix(values, row_labels=labels, col_labels=labels,
                norm=SymLogNorm(linthresh=0.1, linscale=0.2, vmin=-extremum, vmax=extremum),
                **plot_kwargs)
