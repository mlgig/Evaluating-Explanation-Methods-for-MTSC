from typing import Union, Optional, Sequence, List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from timexplain._explanation import Explanation, TimeExplanation, FreqExplanation, Slicing
from timexplain._utils import take_if_deep
from timexplain.om import Omitter, TimeSliceOmitter, FreqDiceOmitterBase
from timexplain.plot._slice import slice as plot_slice, Layer, EdgeLayer, UniformSliceLayer, ValuedSliceLayer

LegendManip = Callable[[List[plt.Artist]], List[plt.Artist]]


# Parameter options:
# - domain: "auto", "t", "f"
# - impact_style: None, "trace", "bar"
# - slice_edge_style: None, "tick", "bar"
# - perturb_marker_style: None, "bar"
# - legend_style: None, "auto", "righthand", "above"
# - xlabel: None, "auto", custom
#
def saliency1d(expl_or_om: Union[TimeExplanation, FreqExplanation, TimeSliceOmitter, FreqDiceOmitterBase] = None,
               impacts: Sequence = None,
               x_specimen: Sequence[float] = None,
               *, domain="auto", slicing: Slicing = None, model_output: int = None,
               perturb_omitter: Union[TimeSliceOmitter, FreqDiceOmitterBase] = None, z_perturb: Sequence[int] = None,
               impact_style="trace", slice_edge_style="tick", perturb_marker_style="bar",
               specimen_color: str = None, specimen_linewidth: float = None, specimen_legend="auto",
               impact_cmap="RdBu_r", impact_cmap_range=(0.0, 1.0), impact_linewidth_factor=10.0,
               impact_bound: float = None, impact_legend="Impact",
               slice_edge_color="gray", slice_edge_linewidth=1.0, slice_edge_linestyle=":",
               perturb_marker_color="wheat", perturb_marker_linewidth=7.0,
               legend_style="righthand", legend_manipulator: LegendManip = None, return_legend_handles=False,
               extra_layers: Union[Layer, Sequence[Layer]] = None,
               ax: plt.Axes = None, figsize: Tuple[float, float] = None, title: str = None, xlabel="auto",
               ylabel: str = None) \
        -> Optional[List[plt.Artist]]:
    if x_specimen is None and not isinstance(expl_or_om, Explanation):
        raise ValueError("Either an explanation or 'x_specimen' has to be supplied.")

    if expl_or_om is not None:
        domain = "f" if hasattr(expl_or_om, "freq_slicing") else "t"
        if domain == "f" and hasattr(expl_or_om, "time_slicing") and getattr(expl_or_om, "time_slicing").n_slices != 1:
            raise ValueError("1D saliency plot works with frequency omitters/explanations only when there is "
                             "exactly one time slice.")
        slicing = getattr(expl_or_om, "freq_slicing", getattr(expl_or_om, "time_slicing", slicing))
        x_specimen = getattr(expl_or_om, "x_specimen", x_specimen)
        impacts = getattr(expl_or_om, "impacts", impacts)

    if impacts is not None:
        if isinstance(impacts, sparse.spmatrix):
            impacts = impacts.toarray()
        impacts = take_if_deep(impacts, model_output, 2,
                               "Must also supply 'model_output' when impacts with multiple model outputs are supplied.")

    if not domain.startswith(("t", "f")):
        raise ValueError("Either an explanation, an omitter, or 'domain' has to be supplied.")

    # Perturb the specimen as wished if applicable.
    if z_perturb is not None:
        if perturb_omitter is None:
            if isinstance(expl_or_om, Omitter):
                perturb_omitter = expl_or_om
            else:
                raise ValueError("In order to perturb the specimen, 'perturb_omitter' has to be supplied.")
        x_specimen = perturb_omitter.omit(x_specimen, z_perturb)

    layers = []

    # Plot perturbation markers if applicable.
    if perturb_marker_style is not None and z_perturb is not None:
        zero_slices = np.where(np.asarray(z_perturb) == 0)[0]
        layers.append(UniformSliceLayer(name="perturb markers",
                                        active_slices=zero_slices,
                                        style=perturb_marker_style,
                                        legend=("Disabled slices", "Disabled bands"),
                                        color=perturb_marker_color,
                                        linewidth=perturb_marker_linewidth))

    # Plot impacts if applicable.
    if impact_style is not None and impacts is not None:
        layers.append(ValuedSliceLayer(name="impacts",
                                       slice_values=impacts,
                                       style=impact_style,
                                       cmap=impact_cmap,
                                       cmap_range=impact_cmap_range,
                                       legend=impact_legend,
                                       linewidth_factor=impact_linewidth_factor,
                                       value_range=impact_bound))

    # Plot slice edges if applicable.
    if slice_edge_style is not None:
        layers.append(EdgeLayer(name="slice edges",
                                style=slice_edge_style,
                                legend=("Slice edges", "Band edges"),
                                color=slice_edge_color,
                                linewidth=slice_edge_linewidth,
                                linestyle=slice_edge_linestyle))

    if extra_layers is not None:
        try:
            layers += extra_layers
        except TypeError:
            layers.append(extra_layers)

    if specimen_legend == "auto":
        specimen_legend = "Specimen" if z_perturb is None else "Perturbed specimen"

    kwargs = {
        "domain": domain,
        "specimen_color": specimen_color,
        "specimen_linewidth": specimen_linewidth,
        "legend_manipulator": legend_manipulator,
        "ax": ax,
        "figsize": figsize,
        "title": title,
        "ylabel": ylabel
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return plot_slice(layers=layers, specimen_legend=specimen_legend,
                      x_specimen=x_specimen, slicing=slicing, legend_style=legend_style,
                      return_legend_handles=return_legend_handles, xlabel=xlabel, **kwargs)
