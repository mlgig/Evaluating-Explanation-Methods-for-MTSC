from dataclasses import dataclass, field
from typing import Union, Optional, Collection, Sequence, List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from timexplain._explanation import Slicing
from timexplain._utils import rfft_magnitudes

LegendManip = Callable[[List[plt.Artist]], List[plt.Artist]]


class Layer:
    pass


@dataclass
class EdgeLayer(Layer):
    name: str
    style: Optional[str]  # None, "tick", "bar"
    # The following options are only relevant when style = "bar":
    legend: Optional[Union[str, Tuple[str, str]]] = field(default=None)
    color: str = field(default="gray")
    linewidth: float = field(default=1)
    linestyle: str = field(default=":")


@dataclass
class UniformSliceLayer(Layer):
    name: str
    active_slices: Collection[int]
    style: Optional[str]  # None, "trace", "bar"
    legend: Optional[Union[str, Tuple[str, str]]] = field(default=None)
    color: str = field(default="wheat")
    linewidth: float = field(default=7)  # only relevant if style = "trace"


@dataclass
class ValuedSliceLayer(Layer):
    name: str
    slice_values: Sequence[float]
    style: Optional[str]  # None, "trace", "bar"
    cmap: Union[str, Colormap] = field(default="RdBu_r")
    cmap_range: Tuple[float, float] = field(default=(0, 1))
    legend: Optional[Union[str, Tuple[str, str]]] = field(default=None)
    linewidth_factor: float = field(default=10)  # only relevant if style = "trace"
    value_range: Optional[float] = field(default=None)


# Parameter options:
# - domain: "t", "f"
def slice(x_specimen: Sequence[float], domain: str,
          slicing: Slicing = None,
          layers: Sequence[Layer] = None,
          specimen_color="black", specimen_linewidth=1.5, specimen_legend="Specimen",
          legend_style="righthand", legend_manipulator: LegendManip = None, return_legend_handles=False,
          ax: plt.Axes = None, figsize=(10.0, 4.0), title: str = None, xlabel="auto", ylabel: str = None) \
        -> Optional[List[plt.Artist]]:
    freq_domain = domain.startswith("f")
    if freq_domain:
        # Transform specimen into the frequency domain and take the frequency bin magnitudes.
        x_specimen = rfft_magnitudes(x_specimen)

    x_specimen_xcoords = np.arange(len(x_specimen))
    if freq_domain:
        # In the time domain, a time series starts at zero, but in the frequency domain,
        # rfft_magnitudes() throws away the first bin, so the bins start at one.
        x_specimen_xcoords += 1

    # Create figure if necessary.
    if ax is None:
        ax = plt.figure(figsize=figsize).gca()

    # Set title, xlabel, and ylabel if applicable.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        if xlabel == "auto":
            xlabel = "Frequency" if freq_domain else "Time"
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # We collect the legend handles separately so we can influence their order and apperance.
    legend_handles = []
    # We also count up z order.
    zorder = 1

    # If the user did not supply an explicit slicing, implicitly put each x_specimen point in its own slice.
    if slicing is None:
        size_x = len(x_specimen)
        if freq_domain:
            slicing = Slicing(n_slices=size_x // 2 + 1, cont_interval=(0, size_x // 2))
        else:
            slicing = Slicing(n_slices=size_x, cont_interval=(0, size_x - 1))

    for layer in layers:
        if layer.style:
            if isinstance(layer, EdgeLayer):
                legend_handles += _plot_edge_layer(layer, freq_domain, slicing, ax, zorder)
            elif isinstance(layer, UniformSliceLayer):
                legend_handles += _plot_uniform_slice_layer(layer, freq_domain, x_specimen, x_specimen_xcoords,
                                                            slicing, ax, zorder)
            elif isinstance(layer, ValuedSliceLayer):
                _plot_valued_slice_layer(layer, freq_domain, x_specimen, x_specimen_xcoords,
                                         slicing, ax, zorder, legend_style)
            else:
                raise ValueError(f"Encountered layer of unknown type: {layer}")

            zorder += 1

    # Plot specimen.
    line, = ax.plot(x_specimen_xcoords, x_specimen, zorder=zorder,
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


def _plot_edge_layer(layer: EdgeLayer, freq_domain, slicing, ax, zorder):
    if layer.style == "tick":
        # Add one minor tick at each slice edge.
        ax.set_xticks(slicing.cont_edges, minor=True)

        # Legend
        return []
    elif layer.style == "bar":
        for edge in slicing.cont_edges:
            ax.axvline(edge, zorder=zorder,
                       color=layer.color, linewidth=layer.linewidth, linestyle=layer.linestyle)

        # Legend
        if layer.legend is not None:
            return [Line2D([0], [0],
                           color=layer.color, linewidth=layer.linewidth, linestyle=layer.linestyle,
                           label=_get_legend_label(layer.legend, freq_domain))]
        else:
            return []
    else:
        raise ValueError(f"In layer '{layer.name}': "
                         f"Unknown slice edge style '{layer.style}'; only None, 'tick', or 'bar' allowed.")


def _plot_uniform_slice_layer(layer: UniformSliceLayer, freq_domain, x_specimen, x_specimen_xcoords,
                              slicing, ax, zorder):
    if layer.style == "trace":
        segments = _prep_line_collection_segments(x_specimen, x_specimen_xcoords, slicing)
        segments = np.array(segments)[layer.active_slices]
        lc = LineCollection(segments, colors=(to_rgba(layer.color),), linewidths=(layer.linewidth,), zorder=zorder)
        ax.add_collection(lc)

        # Legend
        if layer.legend is not None:
            return [Line2D([0], [0],
                           color=layer.color, linewidth=layer.linewidth,
                           label=_get_legend_label(layer.legend, freq_domain))]
        else:
            return []
    elif layer.style == "bar":
        for i in layer.active_slices:
            ax.axvspan(*slicing.cont_slices[i], color=layer.color, zorder=zorder)

        # Legend
        if layer.legend is not None:
            return [Patch(facecolor=layer.color, label=_get_legend_label(layer.legend, freq_domain))]
        else:
            return []
    else:
        raise ValueError(f"In layer '{layer.name}': "
                         f"Unknown slice style '{layer.style}'; only None, 'trace', or 'bar' allowed.")


def _plot_valued_slice_layer(layer: ValuedSliceLayer, freq_domain, x_specimen, x_specimen_xcoords,
                             slicing, ax, zorder, legend_style):
    if len(layer.slice_values) != slicing.n_slices:
        raise ValueError(f"In layer '{layer.name}': "
                         f"Length of value array ({len(layer.slice_values)}) must match "
                         f"number of slices ({slicing.n_slices}).")

    # Normalize values so that they range from -1 to 1.
    value_range = layer.value_range or np.max(np.abs(layer.slice_values))
    if value_range != 0:
        normalized_values = np.clip(layer.slice_values / value_range, -1, 1)
    else:
        normalized_values = layer.slice_values

    # Retrieve cmap callable and adjust it to be limited to the user-defined range if necessary.
    cmap = get_cmap(layer.cmap)
    if layer.cmap_range[0] != 0 or layer.cmap_range[1] != 1:
        cmap = ListedColormap(cmap(np.linspace(*layer.cmap_range, 256)))

    if layer.style == "trace":
        segments = _prep_line_collection_segments(x_specimen, x_specimen_xcoords, slicing)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1),
                            linewidths=np.abs(normalized_values) * layer.linewidth_factor + 2, zorder=zorder)
        lc.set_array(normalized_values)
        ax.add_collection(lc)
    elif layer.style == "bar":
        for i, weight in enumerate(normalized_values):
            ax.axvspan(*slicing.cont_slices[i], color=cmap(weight / 2 + 0.5), zorder=zorder)
    else:
        raise ValueError(f"In layer '{layer.name}': "
                         f"Unknown slice style '{layer.style}'; only None, 'trace', or 'bar' allowed.")

    # Plot explanatory colorbar if applicable.
    if layer.legend is not None:
        _plot_colorbar(cmap, layer.legend, legend_style, freq_domain, value_range, ax)


def _plot_colorbar(cmap, legend, legend_style, freq_domain, value_range, ax):
    cbar_height = "50%" if legend_style == "righthand" else "100%"
    cbar_loc = "lower left" if legend_style == "righthand" else "center left"
    axins = inset_axes(ax, width=0.1, height=cbar_height, loc=cbar_loc,
                       bbox_to_anchor=(1, 0, 1, 1), bbox_transform=ax.transAxes)
    ColorbarBase(axins, cmap=cmap, norm=plt.Normalize(-value_range, value_range))
    axins.set_ylabel(_get_legend_label(legend, freq_domain))


def _get_legend_label(label, freq_domain):
    if not label or isinstance(label, str):
        return label
    else:
        return label[1] if freq_domain else label[0]


def _prep_line_collection_segments(x_specimen, x_specimen_xcoords, slicing):
    # format: [[x0,x1,x2,x3,x4], [x4,x5,x6,x7,...], ...]
    segment_xcoords = [[start, *np.arange(np.floor(start) + 1, np.ceil(stop)), stop]
                       for start, stop in slicing.cont_slices]
    # format: [segment1, segment2, ...] with each segment having the format: [[x0,y0],[x1,y1],...]
    return [np.transpose([xcoords, np.interp(xcoords, x_specimen_xcoords, x_specimen)])
            for xcoords in segment_xcoords]


def _plot_legend(ax, legend_style, legend_handles):
    if legend_style == "auto":
        ax.legend(handles=legend_handles)
    elif legend_style == "righthand":
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 0, 1, 1), borderaxespad=0)
    elif legend_style == "above":
        ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0, 1.05, 1, 0), borderaxespad=0,
                  ncol=len(legend_handles))
    else:
        raise ValueError(f"Unknown legend style '{legend_style}'; only None, 'auto', 'righthand', or 'above' allowed.")
