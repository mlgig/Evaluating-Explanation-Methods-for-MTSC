from functools import reduce, partial
from typing import Union, Collection, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from timexplain._utils import SingleOrSeq, is_iterable, unpublished
from timexplain.om import Omitter, TimeSliceOmitter, FreqDiceOmitterBase
from timexplain.om.link import Link, identity, convert_to_link
from timexplain.plot._saliency1d import saliency1d as plot_saliency1d
from timexplain.plot._slice import UniformSliceLayer


# Parameter options:
# - plot_frags: None, "influencer", "influenced", "both"
@unpublished
def crossinfl_demo(influencer_omitters: SingleOrSeq[Omitter],
                   influencer_frags: Union[int, Sequence[int], Sequence[Collection[int]]],
                   influenced_omitters: SingleOrSeq[Omitter],
                   influenced_frags: Union[int, Sequence[int], Sequence[Collection[int]]],
                   model_predict: callable = None, model_output: int = 0,
                   *, last="influencer", link: Link = identity,
                   plot_slices="both", plot_slices_style="bar", plot_slices_color="wheat", plot_slices_linewidth=7.0,
                   figsize=(12, 6), return_fig=False, **kwargs):
    influencer_omitters, influencer_frags, omit_influencer = _create_omit(influencer_omitters, influencer_frags)
    influenced_omitters, influenced_frags, omit_influenced = _create_omit(influenced_omitters, influenced_frags)

    # ul = upper left
    # ur = upper right
    # ll = lower left
    # lr = lower right
    x_specimen = influencer_omitters[0].x_specimen
    samples = {
        "lr": x_specimen,
        "ll": omit_influenced(x_specimen),
        "ur": omit_influencer(x_specimen)
    }
    if last == "influencer":
        samples["ul"] = omit_influencer(samples["ll"])
    elif last == "influenced":
        samples["ul"] = omit_influenced(samples["ur"])
    else:
        raise ValueError("Only 'influencer' or 'influenced' allowed as last omission.")

    if model_predict:
        link = convert_to_link(link)
        ys = {
            pos: link(model_predict([samples[pos]])[0][model_output])
            for pos in ["ul", "ur", "ll", "lr"]
        }

    print(f"[{last}({'influenced' if last == 'influencer' else 'influencer'}(x)) - influencer(x)] / "
          f"[influenced(x) - {'x':^8}] - 1")
    if model_predict:
        print(f"[{ys['ul']:^25.4f} - {ys['ur']:^13.4f}] / [{ys['ll']:^13.4f} - {ys['lr']:^8.4f}] - 1 = " +
              "{:.4f}".format((ys['ul'] - ys['ur']) / (ys['ll'] - ys['lr']) - 1))

    # For both influencer and influenced, find out whether they have a TimeSliceOmitter and FreqSliceOmitter
    # and store the results.
    influencer_omissions = _get_time_and_freq_omissions(influencer_omitters, influencer_frags)
    influenced_omissions = _get_time_and_freq_omissions(influenced_omitters, influenced_frags)

    # Determine which omission needs to be applied at which position, for both the time and frequency subplots.
    plot_omissions = {}
    if plot_slices == "influencer":
        plot_omissions["ur"] = influencer_omissions
        plot_omissions["ul"] = influencer_omissions
    elif plot_slices == "influenced":
        plot_omissions["ll"] = influenced_omissions
        plot_omissions["ul"] = influenced_omissions
    elif plot_slices == "both":
        if influencer_omissions["time"] and influenced_omissions["time"]:
            raise ValueError(f"When plotting both influencer and influenced omitter slices, "
                             f"only one of the two may contain a {TimeSliceOmitter.__name__}.")
        if influencer_omissions["freq"] and influenced_omissions["freq"]:
            raise ValueError(f"When plotting both influencer and influenced omitter slices, "
                             f"only one of the two may contain a {FreqDiceOmitterBase.__name__}.")
        plot_omissions["ll"] = influenced_omissions
        plot_omissions["ur"] = influencer_omissions
        if influencer_omissions["time"]:
            plot_omissions["ul"] = {"time": influencer_omissions["time"], "freq": influenced_omissions["freq"]}
        else:
            plot_omissions["ul"] = {"time": influenced_omissions["time"], "freq": influencer_omissions["freq"]}
    else:
        raise ValueError(f"Unknown value for plot_slices '{plot_slices}'; "
                         f"only None, 'influencer', 'influenced', or 'both' allowed.")

    # For each position and for both time and frequency, generate an extra UniformSliceLayer
    # that marks the omitted fragments.
    extra_layers = {
        pos: {
            "time": _create_slice_layer(plot_omissions, pos, "time",
                                        plot_slices_style, plot_slices_color, plot_slices_linewidth),
            "freq": _create_slice_layer(plot_omissions, pos, "freq",
                                        plot_slices_style, plot_slices_color, plot_slices_linewidth)
        }
        for pos in ["ul", "ur", "ll"]
    }

    gs = plt.GridSpec(nrows=5, ncols=3, width_ratios=[1, 0.15, 1], height_ratios=[1, 1, 0.25, 1, 1])
    fig = plt.figure(figsize=figsize)

    # Plot a time and frequency plot at every position.
    first_time_ax = fig.add_subplot(gs[0, 0])
    first_freq_ax = fig.add_subplot(gs[1, 0])
    for pos, row, col in (("ul", 0, 0), ("ur", 0, 2), ("ll", 3, 0), ("lr", 3, 2)):
        if pos == "ul":
            time_ax, freq_ax = first_time_ax, first_freq_ax
        else:
            time_ax = fig.add_subplot(gs[row, col], sharey=first_time_ax)
            freq_ax = fig.add_subplot(gs[row + 1, col], sharey=first_freq_ax)

        if col != 0:
            plt.setp(time_ax.get_yticklabels(), visible=False)
            plt.setp(freq_ax.get_yticklabels(), visible=False)

        time_om = _get_or_none(plot_omissions, [pos, "time", "om"])
        freq_om = _get_or_none(plot_omissions, [pos, "freq", "om"])

        plot_saliency1d(domain="time", x_specimen=samples[pos],
                        slicing=time_om.time_slicing if time_om else None,
                        legend_style=("above" if pos == "ul" else None),
                        extra_layers=_get_or_none(extra_layers, [pos, "time"]), ax=time_ax,
                        xlabel=None, ylabel=("Time" if col == 0 else None), **kwargs)
        plot_saliency1d(domain="freq", x_specimen=samples[pos],
                        slicing=freq_om.freq_slicing if freq_om else None,
                        legend_style=None,
                        extra_layers=_get_or_none(extra_layers, [pos, "freq"]), ax=freq_ax,
                        xlabel=None, ylabel=("Frequency" if col == 0 else None), **kwargs)

    # Draw the division sign.
    div_ax = fig.add_subplot(gs[2, :])
    div_ax.add_patch(Rectangle((0, 0), 1, 1, color="black"))
    div_ax.set_ylim(0, 4)
    div_ax.axis("off")

    # Draw the minus signs.
    for row in (0, 3):
        minus_ax = fig.add_subplot(gs[row:row + 2, 1])
        minus_ax.add_patch(Rectangle((-1, -1), 2, 2, color="black"))
        minus_ax.set_xlim(-2, 2)
        minus_ax.set_ylim(-18, 18)
        minus_ax.axis("off")

    fig.tight_layout()

    if return_fig:
        return fig


def _create_omit(omitters, frags):
    def frags_to_z(size_z, frags_):
        z = np.ones(size_z)
        z[frags_] = 0
        return z

    # Convert omitters and frags to the correct form.
    if is_iterable(omitters):
        if not is_iterable(frags) or not is_iterable(next(iter(frags))):
            raise ValueError("When providing multiple omitters, you must supply a sequence of collections of"
                             "fragments. Just a single fragment or a collection of fragments is not allowed"
                             "in this case.")
    else:
        omitters = [omitters]
        if not is_iterable(frags):
            frags = [[frags]]
        elif not is_iterable(next(iter(frags))):
            frags = [frags]

    zs = [frags_to_z(om.size_z, frags) for om, frags in zip(omitters, frags)]
    return omitters, frags, partial(reduce, (lambda x, mz: mz[0].omit(x, mz[1])), list(zip(omitters, zs)))


def _get_time_and_freq_omissions(omitters, frags):
    return {
        "time": next(({"om": om, "frags": frags} for om, frags in zip(omitters, frags)
                      if isinstance(om, TimeSliceOmitter)), None),
        "freq": next(({"om": om, "frags": frags} for om, frags in zip(omitters, frags)
                      if isinstance(om, FreqDiceOmitterBase)), None)
    }


def _create_slice_layer(plot_omissions, pos, domain, style, color, linewidth):
    if pos in plot_omissions and plot_omissions[pos][domain]:
        return UniformSliceLayer(name="frags", active_slices=plot_omissions[pos][domain]["frags"],
                                 style=style, legend="Disabled slices or bands", color=color, linewidth=linewidth)
    else:
        return None


def _get_or_none(obj, path):
    for key in path:
        if key in obj and obj[key] is not None:
            obj = obj[key]
        else:
            return None
    return obj
