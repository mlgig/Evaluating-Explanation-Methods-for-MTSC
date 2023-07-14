from typing import Union, Optional, Sequence, List, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from scipy import sparse

from timexplain._explanation import FreqExplanation, Slicing
from timexplain._utils import take_if_deep
from timexplain.om import FreqDiceOmitterBase
from timexplain.plot._slice import _plot_colorbar, _plot_legend

LegendManip = Callable[[List[plt.Artist]], List[plt.Artist]]


def saliency2d(expl_or_om: Union[FreqExplanation, FreqDiceOmitterBase] = None,
               impacts: Sequence = None,
               x_specimen: Sequence[float] = None,
               *, time_slicing: Slicing = None,
               freq_slicing: Slicing = None,
               model_output: int = None,
               dice_edge_style="tick",
               show_specimen=True, specimen_color="black", specimen_linewidth=1.5, specimen_legend="Specimen",
               impact_cmap="RdBu_r", impact_cmap_range=(0.0, 1.0), impact_bound: float = None, impact_legend="Impact",
               dice_edge_color="gray", dice_edge_linewidth=1.0, dice_edge_linestyle=":",
               legend_style="righthand", legend_manipulator: LegendManip = None, return_legend_handles=False,
               ax: plt.Axes = None, figsize=(10.0, 4.0), title: str = None, xlabel="auto", ylabel: str = "auto") \
        -> Optional[List[plt.Artist]]:
    if expl_or_om is not None:
        time_slicing = getattr(expl_or_om, "time_slicing", time_slicing)
        freq_slicing = getattr(expl_or_om, "freq_slicing", freq_slicing)
        x_specimen = getattr(expl_or_om, "x_specimen", x_specimen)
        impacts = getattr(expl_or_om, "impacts", impacts)

    if impacts is not None:
        if isinstance(impacts, sparse.spmatrix):
            impacts = impacts.toarray()
        impacts = take_if_deep(impacts, model_output, 2,
                               "Must also supply 'model_output' when impacts with multiple model outputs are supplied.")

    if impacts.ndim == 1:
        if x_specimen is None and time_slicing is None:
            raise ValueError("You supplied impacts that 1D, but didn't supply 'x_specimen' or 'time_slicing'. "
                             "Either reshape the impacts so they are 2D, or supply 'x_specimen' or 'time_slicing' "
                             "so that this function can reshape the impacts by itself.")
        impacts = impacts.reshape(len(x_specimen) if time_slicing is None else time_slicing.n_slices, -1)

    # Create figure if necessary.
    if ax is None:
        ax = plt.figure(figsize=figsize).gca()

    # Set title, xlabel, and ylabel if applicable.
    if title is not None:
        ax.set_title(title)
    if xlabel == "auto":
        if show_specimen and x_specimen is not None or hasattr(expl_or_om, "time_slicing"):
            ax.set_xlabel("Time")
    elif xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel == "auto":
        if hasattr(expl_or_om, "freq_slicing"):
            ax.set_ylabel("Frequency")
    elif ylabel is not None:
        ax.set_ylabel(ylabel)

    # We collect the legend handles separately so we can influence their order and apperance.
    legend_handles = []

    # Normalize impacts so that they range from -1 to 1.
    if impact_bound is None:
        impact_bound = np.max(np.abs(impacts))
    if impact_bound != 0:
        impacts = np.clip(impacts / impact_bound, -1, 1)

    # Retrieve impact cmap callable and adjust it to be limited to the user-defined range if necessary.
    impact_cmap = get_cmap(impact_cmap)
    if impact_cmap_range[0] != 0 or impact_cmap_range[1] != 1:
        impact_cmap = ListedColormap(impact_cmap(np.linspace(*impact_cmap_range, 256)))

    pcolormesh_args = [impacts.T]
    edgecolor = "none"
    if time_slicing is not None and freq_slicing is not None:
        if not time_slicing.is_contiguous:
            raise ValueError("Time slices must be contiguous.")
        if not freq_slicing.is_contiguous:
            raise ValueError("Frequency slices must be contiguous.")

        # Prepare to plot with dice edges.
        pcolormesh_args = [time_slicing.cont_edges, freq_slicing.cont_edges, impacts.T]

        # (Prepare to) plot the dice edges themselves.
        if dice_edge_style == "tick":
            ax.set_xticks(time_slicing.cont_edges, minor=True)
            ax.set_yticks(freq_slicing.cont_edges, minor=True)
        elif dice_edge_style == "line":
            edgecolor = dice_edge_color
        elif dice_edge_style is not None:
            raise ValueError(f"Unknown dice edge style '{dice_edge_style}'; only None, 'tick', or 'line' allowed.")

    # Plot the spectrogram-like saliency 2D map.
    ax.pcolormesh(*pcolormesh_args, shading="nearest", zorder=1,
                  cmap=impact_cmap, norm=plt.Normalize(-1, 1),
                  edgecolor=edgecolor, lw=dice_edge_linewidth, ls=dice_edge_linestyle)

    # Plot explanatory impact colorbar if applicable.
    if impact_legend is not None:
        _plot_colorbar(impact_cmap, impact_legend, legend_style, False, impact_bound, ax)

    # Plot specimen.
    if show_specimen and x_specimen is not None:
        ax2 = ax.twinx()
        ax2.get_yaxis().set_ticks([])
        line, = ax2.plot(x_specimen, zorder=2,
                         color=specimen_color, lw=specimen_linewidth, label=specimen_legend)
        legend_handles.insert(0, line)

    # Plot legend if applicable.
    if legend_manipulator is not None:
        legend_handles = legend_manipulator(legend_handles)
    if legend_style is not None:
        _plot_legend(ax, legend_style, legend_handles)

    # Return legend handles if applicable.
    if return_legend_handles:
        return legend_handles
