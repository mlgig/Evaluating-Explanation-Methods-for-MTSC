from __future__ import annotations

import collections
from functools import reduce
from typing import TYPE_CHECKING, Tuple, Sequence, Mapping, Dict, FrozenSet

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from timexplain.plot._matrix import matrix as plot_matrix

if TYPE_CHECKING:
    from pandas import DataFrame as TC_DataFrame

Simdict = Dict[FrozenSet[str], Sequence[float]]


def prepare_simdict_from_df(similarity_df: TC_DataFrame,
                            archives: Sequence[str] = None, datasets: Sequence[str] = None,
                            models: Sequence[str] = None, explainers: Sequence[str] = None,
                            archive_col="auto", dataset_col="auto", model_1_col="auto", model_2_col="auto",
                            explainer_col="auto", similarity_col="auto") -> Simdict:
    def auto_col(options):
        return (set(similarity_df.columns) & options).pop()

    # Automatically determine column names if applicable
    if archive_col == "auto" and archives is not None:
        archive_col = auto_col({"archive"})
    if dataset_col == "auto" and datasets is not None:
        dataset_col = auto_col({"dataset"})
    if model_1_col == "auto":
        model_1_col = auto_col({"model_1", "model1", "estimator_1", "estimator1", "est_1", "est1",
                                "classifier_1", "classifier1", "clf_1", "clf1", "regressor_1", "regressor1"})
    if model_2_col == "auto":
        model_2_col = auto_col({"model_2", "model2", "estimator_2", "estimator2", "est_2", "est2",
                                "classifier_2", "classifier2", "clf_2", "clf2", "regressor_2", "regressor2"})
    if explainer_col == "auto" and explainers is not None:
        explainer_col = auto_col({"explainer"})
    if similarity_col == "auto":
        similarity_col = auto_col({"similarity", "correlation_similarity",
                                   "absolute_similarity", "structural_similarity"})

    # Filter data if applicable
    conds = []
    if archives is not None:
        conds.append(similarity_df[archive_col].isin(archives))
    if datasets is not None:
        conds.append(similarity_df[dataset_col].isin(datasets))
    if models is not None:
        conds.append(similarity_df[model_1_col].isin(models) &
                     similarity_df[model_2_col].isin(models))
    if explainers is not None:
        conds.append(similarity_df[explainer_col].isin(explainers))

    if conds:
        similarity_df = similarity_df.loc[reduce(lambda a, b: a & b, conds)]

    sim_dict = {}
    grouped = similarity_df.groupby([model_1_col, model_2_col])[similarity_col].apply(list)
    for model_tuple, sims in grouped.iteritems():
        key = frozenset(model_tuple)
        if key not in sim_dict:
            sim_dict[key] = []
        sim_dict[key] += sims
    return sim_dict


def similarity_median_matrix(simdict: Simdict, model_labels: Mapping[str, str] = None,
                             cmap="viridis", fontcolor_thresh=0.5, linthresh=0.1, vmin: int = None, vmax: int = None,
                             text_len=3, omit_leading_zero=True, trailing_zeros=True,
                             grid: bool = None, angle_left: bool = None, cbar: bool = None,
                             cbar_label="Median similarity",
                             ax: plt.Axes = None, figsize: Tuple[float, float] = None, cellsize: float = None,
                             title: str = None):
    plot_kwargs = {
        "cmap": cmap, "fontcolor_thresh": fontcolor_thresh,
        "text_len": text_len, "omit_leading_zero": omit_leading_zero, "trailing_zeros": trailing_zeros,
        "grid": grid, "angle_left": angle_left, "cbar": cbar, "cbar_label": cbar_label,
        "ax": ax, "figsize": figsize, "cellsize": cellsize, "title": title}
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

    model_names, model_labels = _preprocess_model_names_and_labels(simdict, model_labels)

    values = np.array(
        [[np.nan if model_1 == model_2 else np.median(simdict[frozenset({model_1, model_2})])
          for model_1 in model_names]
         for model_2 in model_names])

    # Plot the heatmap
    norm = SymLogNorm(linthresh=linthresh, linscale=0.1, vmin=vmin, vmax=vmax)
    plot_matrix(values, row_labels=model_labels, col_labels=model_labels, norm=norm, **plot_kwargs)


def similarity_violin_matrix(simdict: Simdict, model_labels: Mapping[str, str] = None,
                             figsize=(20, 8), return_fig=False, title: str = None):
    model_names, model_labels = _preprocess_model_names_and_labels(simdict, model_labels)

    # Create figure
    fig, axs = plt.subplots(1, len(model_names) - 1, figsize=figsize, sharex=True, sharey=True)

    # Set title if applicable.
    if title is not None:
        fig.suptitle(title)

    for row, model_1 in enumerate(model_names[:-1]):
        # Plot visual helper lines.
        axs[row].axvline(-1, color="gray", linewidth=1, linestyle=(0, [1, 3]))
        axs[row].axvline(1, color="gray", linewidth=1, linestyle=(0, [1, 3]))
        axs[row].axvline(0, color="gray", linewidth=1, linestyle=(0, [1, 10]))

        # Plot the violins.
        x = [simdict[frozenset({model_1, model_2})]
             for j, model_2 in enumerate(model_names[1:]) if row <= j]
        axs[row].violinplot(x, showmedians=True, vert=False, positions=np.arange(row, len(model_names) - 1))

        axs[row].set_title(model_labels[row])
        axs[row].spines["top"].set_visible(False)
        axs[row].spines["left"].set_visible(False)
        axs[row].spines["right"].set_visible(False)
        axs[row].tick_params(left=False)

    # Add y labels.
    plt.setp(axs[0], yticks=np.arange(len(model_names) - 1), yticklabels=model_labels[1:])

    fig.subplots_adjust(wspace=1)
    fig.tight_layout()

    if return_fig:
        return fig


def _preprocess_model_names_and_labels(simdict, user_model_labels):
    existing_models = sorted(set().union(*simdict.keys()))

    if isinstance(user_model_labels, collections.Mapping):
        model_names = np.array(list(user_model_labels.keys()))
        model_labels = np.array(list(user_model_labels.values()))
    else:
        if user_model_labels is None:
            model_names = np.array(existing_models)
        else:
            model_names = np.array(list(user_model_labels))
        model_labels = model_names

    mask = np.isin(model_names, existing_models)
    model_names = model_names[mask]
    model_labels = model_labels[mask]

    return model_names, model_labels
