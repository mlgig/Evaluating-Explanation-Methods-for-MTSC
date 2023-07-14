from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from scipy import sparse

from timexplain._explainer import Explainer
from timexplain._explanation import TimeExplanation, FreqExplanation, Slicing
from timexplain._utils import optional_njit, optional_numba_list, optional_numba_dict

if TYPE_CHECKING:
    from pyts.classification import SAXVSM as TC_SAXVSM
    from pyts.transformation import WEASEL as TC_WEASEL


class SaxVsmWordSuperposExplainer(Explainer[TimeExplanation]):
    model: TC_SAXVSM
    divide_spread: bool

    def __init__(self, model: TC_SAXVSM, *, divide_spread=False):
        self.model = model
        self.divide_spread = divide_spread

    def _explain(self, X_specimens):
        from pyts.bag_of_words import BagOfWords

        X_specimens = np.asarray(X_specimens)

        size_x = X_specimens.shape[1]
        n_model_outputs = self.model.tfidf_.shape[0]
        vocab = self.model._tfidf.vocabulary_
        normed_tfidf = self.model.tfidf_ / np.linalg.norm(self.model.tfidf_, axis=1, keepdims=True)

        # Matrix of the form:
        #    1   -1/3 -1/3 -1/3
        #   -1/3  1   -1/3 -1/3
        #   -1/3 -1/3  1   -1/3
        #   -1/3 -1/3 -1/3  1
        mat = np.full((n_model_outputs, n_model_outputs), -1 / (n_model_outputs - 1))
        np.fill_diagonal(mat, 1)

        word_impacts = mat @ normed_tfidf

        # No numerosity reduction here!
        bow = BagOfWords(self.model.window_size, self.model.window_step, numerosity_reduction=False)
        sentences = bow.transform(self.model._sax.transform(X_specimens))

        explanations = []

        for x_specimen, sentence in zip(X_specimens, sentences):
            sentence = sentence.split(" ")
            word_contribs = {word: word_impacts[:, vocab[word]] / (cnt if self.divide_spread else 1)
                             for word, cnt in zip(*np.unique(sentence, return_counts=True))
                             if word in vocab}

            impacts = np.zeros((n_model_outputs, size_x))
            starts_at = 0
            for word in sentence:
                if word in word_contribs:
                    impacts[:, starts_at:starts_at + self.model.window_size] += word_contribs[word][:, np.newaxis]
                starts_at += self.model.window_step
            explanations.append(TimeExplanation(x_specimen, impacts))

        return explanations


class WeaselExplainer(Explainer[Union[TimeExplanation, FreqExplanation]]):
    model: TC_WEASEL
    domain: str

    def __init__(self, model: TC_WEASEL, *, domain="time"):
        if not domain.startswith(("t", "f")):
            raise ValueError("Domain must start with either 't' (time) or 'f' (frequency).")

        self.model = model
        self.domain = domain

    def _explain(self, X_specimens):
        from pyts.utils import windowed_view

        X_specimens = np.asarray(X_specimens)
        n_specimens, size_x = X_specimens.shape
        n_freq_bins = max(self.model._window_sizes) // 2
        time_domain = self.domain.startswith("t")

        overall_y_preds = []
        overall_impacts = []

        for (window_size, window_step, sfa, vectorizer, relevant_features) \
                in zip(self.model._window_sizes, self.model._window_steps, self.model._sfa_list,
                       self.model._vectorizer_list, self.model._relevant_features_list):
            n_windows = (size_x - window_size + window_step) // window_step
            X_windowed = windowed_view(X_specimens, window_size=window_size, window_step=window_step)
            X_windowed = X_windowed.reshape(n_specimens * n_windows, window_size)
            X_sfa = sfa.transform(X_windowed)

            X_word = np.array(["".join(X_sfa[i]) for i in range(n_specimens * n_windows)])
            X_word = X_word.reshape(n_specimens, n_windows)

            # Predictions
            X_bow = np.asarray([" ".join(X_word[i]) for i in range(n_specimens)])
            overall_y_preds.append(vectorizer.transform(X_bow)[:, relevant_features])

            # Impacts

            # 1. Create an array of pairs:
            #    (ngram length, numba dict from ngrams of that length to actual model outputs)
            ngram_range = range(vectorizer.ngram_range[0], vectorizer.ngram_range[1] + 1)
            ngramlen_to_ngram_to_modelout = {ngram_len: optional_numba_dict("unicode_type", "int64")
                                             for ngram_len in ngram_range}
            for ngram, ngram_idx in vectorizer.vocabulary_.items():
                find = np.where(relevant_features == ngram_idx)[0]
                if find.size != 0:
                    ngramlen_to_ngram_to_modelout[ngram.count(" ") + 1][ngram] = find[0]
            ngramlen_to_ngram_to_modelout = optional_numba_list(ngramlen_to_ngram_to_modelout.items())

            if time_domain:
                # Dummy data to make numba not complain.
                global_freq_bins = np.zeros((1, 2))
            else:
                # 2. If drop_sum is False, retroactively drop the sum from the support indices.
                win_freq_bins = np.copy(sfa.support_)
                if not self.model.drop_sum:
                    win_freq_bins = win_freq_bins[win_freq_bins != 0]
                    win_freq_bins -= 1
                #    Also convert the support indices (two consecutive indices represent the real and imag parts
                #    of one bin's output) to bin indices by dividing by 2 and rounding down.
                win_freq_bins //= 2
                #    Convert the support bin indices for this window to global support bin indices along with a weight
                #    for each index which is smaller than 1 when the local bin doesn't fully cover the respective
                #    global bin.
                n_win_freq_bins = window_size // 2
                bin_split = _soft_range_split(n_freq_bins, n_win_freq_bins)
                global_freq_bins = np.vstack([bin_split[freq_bin] for freq_bin in win_freq_bins])

            # 3. Compute a sparse representation of the impacts.
            rowptr, cols, data = \
                _weasel_impacts_csr(time_domain,
                                    n_specimens, size_x, n_freq_bins,
                                    window_size, window_step, n_windows,
                                    ngramlen_to_ngram_to_modelout, global_freq_bins,
                                    X_word,
                                    np.array(" "))

            # 4. Construct the sparse matrix object.
            impacts_shape = (n_specimens, len(relevant_features) * size_x * (1 if time_domain else n_freq_bins))
            overall_impacts.append(sparse.csr_matrix((data, cols, rowptr), impacts_shape))

        overall_y_preds = sparse.hstack(overall_y_preds, format="csr")
        overall_impacts = sparse.hstack(overall_impacts, format="csr")

        if not getattr(self.model, "sparse", True):
            overall_y_preds = overall_y_preds.toarray()

        n_model_outputs = overall_y_preds.shape[1]
        if time_domain:
            constr = TimeExplanation
            kwargs = {}
        else:
            constr = FreqExplanation
            kwargs = {"freq_slicing": Slicing(bin_rate=size_x, n_slices=n_freq_bins, cont_interval=(0, 0.5))}
        return [constr(x_specimen, impact_row.reshape((n_model_outputs, -1)).tocsr(), y_pred=y_pred, **kwargs)
                for x_specimen, y_pred, impact_row
                in zip(X_specimens, overall_y_preds, overall_impacts)]


@optional_njit
def _soft_range_split(range_len, n_segments):
    edges = np.linspace(0, range_len, n_segments + 1)
    split = []
    for segment_idx in range(n_segments):
        segment = []
        lower_edge = edges[segment_idx]
        lower_edge_floor = np.floor(lower_edge)
        lower_edge_ceil = np.ceil(lower_edge)
        upper_edge = edges[segment_idx + 1]
        upper_edge_floor = np.floor(upper_edge)
        if lower_edge < lower_edge_ceil:
            segment.append([lower_edge_floor, lower_edge_ceil - lower_edge])
        for inner_idx in np.arange(lower_edge_ceil, upper_edge_floor):
            segment.append([inner_idx, 1])
        if upper_edge > upper_edge_floor:
            segment.append([upper_edge_floor, upper_edge - upper_edge_floor])
        split.append(np.array(segment))
    return split


@optional_njit
def _weasel_impacts_csr(time_domain,
                        n_specimens, size_x, total_freq_bins,
                        window_size, window_step, n_windows, ngramlen_to_ngram_to_modelout, global_freq_bins,
                        X_word,
                        join_char):  # Pass in np.array(" ") here!
    # Currently, numba cannot combine numpy unichr strings and numba unicode strings.
    # There also seems no way to create a unichr string inside a njit function.
    # For this reason, we pass in the join character " " from the outside, which gives us a unichr.
    join_char = join_char[()]

    rowptr = [0]
    cols = []
    data = []

    for specimen_idx in range(n_specimens):
        for ngram_len, ngram_to_modelout in ngramlen_to_ngram_to_modelout:
            # Note: A frame is a sequence of ngram_len windows.
            frame_size = window_step * (ngram_len - 1) + window_size
            for frame_idx in range(n_windows - ngram_len + 1):
                frame_starts_at = window_step * frame_idx
                frame_ngram = join_char.join(X_word[specimen_idx, frame_idx:frame_idx + ngram_len])
                if frame_ngram in ngram_to_modelout:
                    modelout = ngram_to_modelout[frame_ngram]
                    # In the following, we might write data to cells where data has actually been written before.
                    # When the matrix will be coalesced later, these duplicate entries will be summed up,
                    # which is exactly the behavior we want.
                    if time_domain:
                        from_idx = modelout * size_x + frame_starts_at
                        cols.extend(np.arange(from_idx, from_idx + frame_size))
                        data.extend(np.repeat(1, frame_size))
                    else:
                        for freq_bin, bin_weight in global_freq_bins:
                            from_idx = (modelout * size_x + frame_starts_at) * total_freq_bins + freq_bin
                            cols.extend(np.arange(from_idx, from_idx + total_freq_bins * frame_size, total_freq_bins))
                            data.extend(np.repeat(bin_weight, frame_size))
        rowptr.append(len(cols))

    return np.array(rowptr), np.array(cols), np.array(data)
