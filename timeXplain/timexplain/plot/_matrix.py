import re
from itertools import product
from math import ceil
from typing import Union, Tuple, Collection, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.ticker import FixedFormatter


def matrix(values: np.ndarray, row_labels: Sequence[str] = None, col_labels: Sequence[str] = None,
           row_seps: Union[int, Collection[int]] = None, col_seps: Union[int, Collection[int]] = None,
           cmap="RdBu", fontcolor_thresh=0.5, norm: plt.Normalize = None,
           text_len=4, omit_leading_zero=False, trailing_zeros=False,
           grid=True, angle_left=False, cbar=True, cbar_label: str = None,
           ax: plt.Axes = None, figsize: Tuple[int, int] = None, cellsize=0.65, title: str = None):
    cmap = get_cmap(cmap)

    # Create figure if necessary.
    if ax is None:
        if figsize is None:
            # Note the extra width factor for the colorbar.
            figsize = (cellsize * values.shape[1] * (1.2 if cbar else 1), cellsize * values.shape[0])
        ax = plt.figure(figsize=figsize).gca()

    # Set title if applicable.
    if title is not None:
        ax.set_title(title)

    if row_seps is not None:
        values = np.insert(values, row_seps, np.nan, axis=0)
        if row_labels is not None:
            row_labels = np.insert(row_labels, row_seps, "")
    if col_seps is not None:
        values = np.insert(values, col_seps, np.nan, axis=1)
        if col_labels is not None:
            col_labels = np.insert(col_labels, col_seps, "")

    # Plot the heatmap.
    im = ax.matshow(values, cmap=cmap, norm=norm)

    # Plot the text annotations showing each cell's value.
    norm_values = im.norm(values)
    for row, col in product(range(values.shape[0]), range(values.shape[1])):
        val = values[row, col]
        if not np.isnan(val):
            # Find text color.
            bg_color = cmap(norm_values[row, col])[:3]
            luma = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            color = "white" if luma < fontcolor_thresh else "black"

            # Plot cell text.
            annotation = _format_value(val, text_len, omit_leading_zero, trailing_zeros)
            ax.text(col, row, annotation, ha="center", va="center", color=color)

    # Add ticks and labels.
    if col_labels is None:
        ax.set_xticks([])
    else:
        col_labels = np.asarray(col_labels)
        labeled_cols = np.where(col_labels)[0]
        ax.set_xticks(labeled_cols)
        ax.set_xticklabels(col_labels[labeled_cols])
    if row_labels is None:
        ax.set_yticks([])
    else:
        row_labels = np.asarray(row_labels)
        labeled_rows = np.where(row_labels)[0]
        ax.set_yticks(labeled_rows)
        ax.set_yticklabels(row_labels[labeled_rows])

    ax.tick_params(which="major", bottom=False)

    plt.setp(ax.get_xticklabels(), rotation=40, ha="left", rotation_mode="anchor")

    # Turn off spines.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Rotate the left labels if applicable.
    if angle_left:
        plt.setp(ax.get_yticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    # Create the white grid if applicable.
    if grid:
        # Extra ticks required to avoid glitch.
        xticks = np.concatenate([[-0.56], np.arange(values.shape[1] + 1) - 0.5, [values.shape[1] - 0.44]])
        yticks = np.concatenate([[-0.56], np.arange(values.shape[0] + 1) - 0.5, [values.shape[0] - 0.44]])
        ax.set_xticks(xticks, minor=True)
        ax.set_yticks(yticks, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, top=False, left=False)

    # Create the colorbar if applicable.
    if cbar:
        bar = ax.figure.colorbar(im, ax=ax)
        bar.ax.set_ylabel(cbar_label)
        fmt = bar.ax.yaxis.get_major_formatter()
        if isinstance(fmt, FixedFormatter):
            fmt.seq = [_format_value(eval(re.sub(r"[a-z$\\{}]", "", label.replace("times", "*").replace("^", "**"))),
                                     text_len, omit_leading_zero, trailing_zeros)
                       if label else "" for label in fmt.seq]


def _format_value(val, text_len, omit_leading_zero, trailing_zeros):
    whole, fractional = str(float(val)).split(".")
    if fractional == "0":
        return whole
    else:
        if omit_leading_zero and whole in ("0", "-0"):
            whole = whole[:-1]
        fractional_len = text_len - len(whole) - 1
        if trailing_zeros:
            fractional = f"{val:.{fractional_len}f}".split(".")[1]
        else:
            fractional = str(round(val, fractional_len)).split(".")[1]
        # When val is something like 0.9999, rounding results in something like fractional="00".
        # In this case, we of course want to return 1, that is, ceil(val).
        if all(c == "0" for c in fractional):
            return str(ceil(val))
        else:
            return f"{whole}.{fractional}"
