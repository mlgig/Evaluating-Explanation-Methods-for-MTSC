from itertools import cycle
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb, rgb_to_hsv, hsv_to_rgb

from timexplain._utils import unpublished
from timexplain.om.crossinfl._cancellation_progression import CnclProg

_DEFAULT_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


@unpublished
def cancellation_progression(cnclp: CnclProg, model_output: int = None,
                             *, color_cycle: Sequence = None,
                             ax: plt.Axes = None, figsize=(8.0, 6.0), title: str = None,
                             xlabel="Estimated cancellation", ylabel="True cancellation", legend=True):
    if model_output is None:
        test_array = next(iter(next(iter(cnclp.values())).values()))
        if test_array.ndim != 2:
            n_model_outputs = test_array.shape[2]
            if n_model_outputs != 1:
                raise ValueError("When plotting cancellation progressions, the model_output argument can only be "
                                 "omitted when there is only one model output in the progression. However, this "
                                 f"progression stores {n_model_outputs} model outputs. So, please supply model_output.")
            else:
                model_output = 0

    # Create figure if necessary.
    if ax is None:
        ax = plt.figure(figsize=figsize).gca()

    # Set title, xlabel, and ylabel if applicable.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if color_cycle is None:
        color_cycle = _DEFAULT_COLOR_CYCLE

    for (om, om_cnclp), base_color in zip(cnclp.items(), cycle(color_cycle)):
        color_hue = rgb_to_hsv(to_rgb(base_color))[0]

        for (frag, frag_cnclp), color_value in zip(om_cnclp.items(), np.linspace(1, 0.7, len(om_cnclp))):
            if model_output is not None:
                frag_cnclp = frag_cnclp[model_output]

            if not np.all(np.isnan(frag_cnclp[:, 1])):
                third_col_max = int(frag_cnclp[0, 2])

                max_sat_color = hsv_to_rgb([color_hue, 1.0, color_value])
                min_sat_color = hsv_to_rgb([color_hue, 0.2, color_value])
                # Use linspace in the wrong order and then reverse the result, such that when
                # 'third_col_max' is 1, the cmap contains the max saturation and not the
                # min saturation color as its only color.
                cmap = ListedColormap(np.linspace(max_sat_color, min_sat_color, third_col_max)[::-1])

                segments = list(zip(frag_cnclp[:-1, :2], frag_cnclp[1:, :2]))
                lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(1, third_col_max),
                                    label=f"{type(om).__name__[0]}~{frag}", color=max_sat_color, zorder=1)
                lc.set_array(frag_cnclp[1:, 2])
                ax.add_collection(lc)

    # Necessary because 'ax.add_collection()' doesn't adjust xlim and ylim automatically.
    ax.autoscale_view()

    # Draw the diagonal dotted line.
    diag = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(diag, diag, ls="dotted", color="gray", zorder=0)

    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0, 1, 1))
