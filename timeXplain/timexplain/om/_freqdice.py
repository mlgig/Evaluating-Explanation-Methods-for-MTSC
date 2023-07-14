from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Union, Sequence, List, Tuple, Callable
from typing import Optional

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import iirdesign, firls, filtfilt, sosfiltfilt

from timexplain._explanation import FreqExplanation, Slicing
from timexplain._utils import replace_zero_slices
from timexplain.om._base import Omitter
from timexplain.om._surrogate import x_sample


class FreqDiceOmitterBase(Omitter, ABC):
    size_x: int
    time_slicing: Slicing
    freq_slicing: Slicing

    @property
    def size_z(self):
        return self.time_slicing.n_slices * self.freq_slicing.n_slices

    # Creates the so-called "fade matrix". It has shape n_time_slices x size_x.
    # Each row starts with 0 up to half the "fade length" before the start of the corresponding time slice.
    # There, the values linearly increase to 1 until half the fade length after the time slice start.
    # The time slice end fades out in an equivalent fashion.
    # Later on, this fade matrix will be multiplied element-wise with a matrix of time series that should be
    # faded together such that each time series is "active" in the time slice corresponding to its row.
    # The resulting matrix will then be summed over all rows to create the final faded time series.
    def _init_fade_matrix(self):
        n_time_slices = self.time_slicing.n_slices
        fade_matrix = np.zeros((n_time_slices, self.size_x))
        half_fade_len = np.clip(self.size_x // (n_time_slices * 8), 1, 500)  # TODO: Make this a parameter?
        fade_in = np.linspace(0, 1, half_fade_len * 2 + 2)[1:-1]
        for t_slice, (t_start, t_end) in enumerate(self.time_slicing.bin_slices):
            # Center
            fade_matrix[t_slice, t_start + half_fade_len:t_end - half_fade_len] = 1
            # Rising edge (left)
            if t_slice == 0:
                fade_matrix[t_slice, :t_start + half_fade_len] = 1
            else:
                fade_matrix[t_slice, t_start - half_fade_len:t_start + half_fade_len] = fade_in
            # Falling edge (right)
            if t_slice == n_time_slices - 1:
                fade_matrix[t_slice, t_end - half_fade_len:] = 1
            else:
                fade_matrix[t_slice, t_end - half_fade_len:t_end + half_fade_len] = 1 - fade_in
        self._fade_matrix = fade_matrix

    def _omit(self, X, Z):
        if X.shape[1] != self.size_x:
            raise ValueError(f"Length of each sample in X must match size_x of {type(self).__name__}.")

        # We will later write the omitted X into this array.
        X_out = np.zeros_like(X)

        # The Z matrix is inputted as n_specimens x size_z. Reshape it for convenience.
        n_time_slices = self.time_slicing.n_slices
        Z = Z.reshape((len(Z), n_time_slices, self.freq_slicing.n_slices))

        fade_matrix = self._fade_matrix

        # For each unique x...
        X_uniq, uniq_map_X = np.unique(X, axis=0, return_inverse=True)
        for uniq_idx_x, x in enumerate(X_uniq):
            # Find the indices in X where this unique x occurs.
            global_indices_x = np.where(uniq_map_X == uniq_idx_x)[0]
            # Find the z (shape: n_time_slices x n_freq_slices) of each occurrence of x and and stack them together.
            # The resulting array has shape n_occurrences*n_time_slices + n_freq_slices.
            # So each row is a "omitband profile" (OMB) that instructs us which frequency bands to omit.
            obps = np.vstack(Z[global_indices_x])
            # Determine the unique omitband profiles.
            obps_uniq, uniq_map_obps = np.unique(obps, axis=0, return_inverse=True)
            # Apply each of these unique omitband profiles to the whole x,
            # yielding one processed x for each unique omitband profile.
            X_omitted = self._omit_band_profiles(x, obps_uniq)
            # For each occurrence of x...
            for stacked_idx_x, global_idx_x in enumerate(global_indices_x):
                # For each time slice, this occurrence of x has a certain omitband profile.
                # We now stack the processed versions of x for these omitband profiles.
                start_idx = stacked_idx_x * n_time_slices
                X_time_slices = X_omitted[uniq_map_obps[start_idx:start_idx + n_time_slices]]
                # Utilize the fade matrix (explained above) to fade together these processed versions of x,
                # yielding the output x for the current occurrence of x.
                X_out[global_idx_x] = np.sum(X_time_slices * fade_matrix, axis=0)

        return X_out

    @abstractmethod
    def _omit_band_profiles(self, x, band_profiles):
        raise NotImplementedError

    def create_explanation(self, x_specimen: Sequence[float], y_pred: Optional, impacts: np.ndarray) -> FreqExplanation:
        return FreqExplanation(x_specimen, impacts,
                               y_pred=y_pred, time_slicing=self.time_slicing, freq_slicing=self.freq_slicing)

    @staticmethod
    @abstractmethod
    def max_n_freq_slices(size_x: int) -> int:
        raise NotImplementedError


# === Filter omitter ==========================================

FilterDesigner = Callable[[int, Sequence[Tuple[float, float]]], Sequence[Tuple[str, Any]]]

# Stems from empirical experiments; with 1 as min bandwidth, the ripples become too extreme.
_MIN_BANDWIDTH = 2.0
# Attenuation in the stopbands in dB.
_ATTENUATION = 30.0
# Widths of the transition bands for both filter types. These values stem from empirical experiments.
_ELLIP_TRANSITION_BANDWIDTH = 0.1
_FIRLS_TRANSITION_BANDWIDTH = 0.5


# IIR filters are sometimes unstable, but they feature very sharp cutoffs.
# We choose an elliptic filter because of its particularly sharp cutoff.
# We only advise using this filter if you know it will perform well with your type of signals.
def ellip_filter(size_x: int, stopbands: Sequence[Tuple[float, float]]) -> Sequence[Tuple[str, Any]]:
    filters = []

    for f_lower, f_upper in stopbands:
        lowcut = f_lower - _ELLIP_TRANSITION_BANDWIDTH <= 0
        highcut = f_upper + _ELLIP_TRANSITION_BANDWIDTH >= size_x // 2

        if lowcut and highcut:
            # This sos filter cuts everything.
            return [("sos", [[0, 0, 0, 1, 0, 0]])]

        wp = [f_lower - _ELLIP_TRANSITION_BANDWIDTH, f_upper + _ELLIP_TRANSITION_BANDWIDTH]
        ws = [f_lower, f_upper]

        if lowcut:
            wp = wp[1]
            ws = ws[1]
        if highcut:
            wp = wp[0]
            ws = ws[0]

        filters.append(("sos", iirdesign(wp=wp, ws=ws, gpass=0.1, gstop=_ATTENUATION,
                                         ftype="ellip", output="sos", fs=size_x)))

    return filters


# FIR filters are pretty much always reliable. However, they often have softer cutoffs.
# We choose FIRLS over other methods because...
#  - Remez minimizes the maximum error (in contrast to least squares), potentially producing large errors everywhere.
#  - Window methods are fast and simple, but don't produce optimal results.
def firls_filter(size_x: int, stopbands: Sequence[Tuple[float, float]]) -> Sequence[Tuple[str, Any]]:
    if len(stopbands) == 0:
        return []

    bands = []
    desired = []

    for f_lower, f_upper in stopbands:
        bands += [f_lower - _FIRLS_TRANSITION_BANDWIDTH, f_lower, f_upper, f_upper + _FIRLS_TRANSITION_BANDWIDTH]
        desired += [1, 0, 0, 1]

    if bands[0] <= 0:  # lowcut
        del bands[0]
        del desired[0]
        bands[0] = 0
    else:
        bands.insert(0, 0)
        desired.insert(0, 1)

    if bands[-1] >= size_x // 2:  # highcut
        del bands[-1]
        del desired[-1]
        bands[-1] = size_x // 2
    else:
        bands.append(size_x // 2)
        desired.append(1)

    # "fred harris rule of thumb"
    numtaps = int(np.ceil(size_x * _ATTENUATION / (_FIRLS_TRANSITION_BANDWIDTH * 22)))
    # Avoid filter orders that are too large for the default padding of filtfilt().
    numtaps = min(size_x // 3 - 1, numtaps)
    # Necessary because FIRLS only supports odd filter orders.
    if numtaps % 2 == 0:
        numtaps -= 1

    b = firls(numtaps=numtaps, bands=bands, desired=desired, fs=size_x)
    return [("tf", (b, 1))]


class FreqDiceFilterOmitter(FreqDiceOmitterBase):
    filter_designer: FilterDesigner

    def __init__(self, size_x: int, time_slicing: Union[int, Slicing], freq_slicing: Union[int, Slicing],
                 filter_designer: FilterDesigner = firls_filter,
                 sample_rate=1.0):
        self.constr_args = {"size_x": size_x, "time_slicing": time_slicing, "freq_slicing": freq_slicing,
                            "filter_designer": filter_designer, "sample_rate": sample_rate}
        self.deterministic = True
        self.requires_X_bg = False

        self.size_x = size_x
        self.filter_designer = filter_designer

        if isinstance(time_slicing, Integral):
            time_slicing = Slicing(n_slices=time_slicing)
        time_slicing = time_slicing.refine(bin_rate=sample_rate, bin_interval=(0, size_x))
        if time_slicing.bin_slices is None:
            raise ValueError("FreqDiceFilterOmitter requires a time slicing with bin slices.")
        if np.any(time_slicing.bin_edges < 0) or np.any(time_slicing.bin_edges > size_x):
            raise ValueError(f"Time slice bin edges must not exceed [0, size_x]: {time_slicing.bin_slices}")
        self.time_slicing = time_slicing

        if isinstance(freq_slicing, Integral):
            freq_slicing = Slicing(n_slices=freq_slicing, spacing="quadratic")
        # Note: We make sure that the lowest frequency band always contains _MIN_BANDWIDTH Hz.
        # This is to ensure that no funny things happen due to too narrow stop bands.
        first_freq_slice_vol = None if freq_slicing.n_slices is None else \
            max(_MIN_BANDWIDTH, size_x / (2 * 3 * freq_slicing.n_slices)) * sample_rate / size_x
        freq_slicing = freq_slicing.refine(bin_rate=size_x / sample_rate, cont_interval=(0, sample_rate / 2),
                                           first_slice_vol=first_freq_slice_vol)
        min_bandwidth = _MIN_BANDWIDTH * sample_rate / size_x
        if np.any(np.diff(freq_slicing.cont_slices) < min_bandwidth - 1e-5):
            raise ValueError("FreqDiceFilterOmitter requires all continuous frequency slices to have "
                             f"volume >= {min_bandwidth}: {freq_slicing.cont_slices}")
        self.freq_slicing = freq_slicing

        super()._init_fade_matrix()

    def _omit_band_profiles(self, x, band_profiles):
        return np.array([self._filter_stopbands(x, self._compute_stopbands(bp)) for bp in band_profiles])

    def _compute_stopbands(self, band_profile):
        stopbands = np.array([self.freq_slicing.cont_slices[pos]
                              for pos, enbld in enumerate(band_profile) if enbld == 0])

        if len(stopbands) != 0:
            # Merge neighboring stopbands.
            r = np.reshape(stopbands, -1)
            mask = r[1:] != r[:-1]
            stopbands = r[np.insert(mask, 0, True) & np.append(mask, True)].reshape(-1, 2)

        return stopbands

    def _filter_stopbands(self, x, stopbands):
        filters = self.filter_designer(self.size_x, stopbands * self.freq_slicing.bin_rate)

        for filter_type, filter_coeffs in filters:
            if filter_type == "tf":
                x = filtfilt(*filter_coeffs, x)
            elif filter_type == "sos":
                x = sosfiltfilt(filter_coeffs, x)
            else:
                raise ValueError(f"Unknown filter type: '{filter_type}'. Only 'tf' or 'sos' allowed.")

        return x

    @staticmethod
    def max_n_freq_slices(size_x: int) -> int:
        return int(np.ceil(size_x / (2 * _MIN_BANDWIDTH)) - 1)


# === Patch omitter ===========================================

XPatch = Sequence[float]
PatchSpectrum = Sequence[float]
PatchStrat = Callable[[np.ndarray, np.ndarray], np.ndarray]


class FreqDicePatchOmitter(FreqDiceOmitterBase):
    patch_spectrum: Optional[Sequence[float]]
    patch_slices: List[Sequence[float]]
    patch_strat: PatchStrat

    def __init__(self, size_x: int, time_slicing: Slicing, freq_slicing: Slicing,
                 x_patch: Union[None, XPatch, Callable[[np.ndarray], XPatch]] = x_sample,
                 patch_spectrum: Union[None, PatchSpectrum, Callable[[np.ndarray], PatchSpectrum]] = None,
                 patch_strat: PatchStrat = lambda a, b: b,
                 sample_rate=1.0, X_bg=None):
        self.constr_args = {"size_x": size_x, "time_slicing": time_slicing, "freq_slicing": freq_slicing,
                            "x_patch": x_patch, "patch_spectrum": patch_spectrum, "patch_strat": patch_strat,
                            "sample_rate": sample_rate, "X_bg": X_bg}
        self.deterministic = x_patch != x_sample
        self.requires_X_bg = (patch_spectrum is None and callable(x_patch)) or callable(patch_spectrum)

        self.size_x = size_x
        self.patch_strat = patch_strat

        if isinstance(time_slicing, Integral):
            time_slicing = Slicing(n_slices=time_slicing)
        time_slicing = time_slicing.refine(bin_rate=sample_rate, bin_interval=(0, size_x))
        if time_slicing.bin_slices is None:
            raise ValueError("FreqDicePatchOmitter requires a time slicing with bin slices.")
        if np.any(time_slicing.bin_edges < 0) or np.any(time_slicing.bin_edges > size_x):
            raise ValueError("Time slice bin edges must not exceed [0, size_x].")
        self.time_slicing = time_slicing

        if isinstance(freq_slicing, Integral):
            freq_slicing = Slicing(n_slices=freq_slicing, spacing="quadratic")
        # Note: rfft() returns: [y(0), y(1), ..., y(size_x//2)]
        # We ignore the first bin since it is just the sum of the signal.
        first_freq_slice_vol = None if freq_slicing.n_slices is None else \
            max(1.0, size_x / (2 * 3 * freq_slicing.n_slices))
        freq_slicing = freq_slicing.refine(bin_rate=size_x / sample_rate, bin_interval=(1, size_x // 2 + 1),
                                           first_slice_vol=first_freq_slice_vol)
        if freq_slicing.bin_slices is None:
            raise ValueError("FreqDicePatchOmitter requires a freq slicing with bin slices.")
        if np.any(freq_slicing.bin_edges < 1) or np.any(freq_slicing.bin_edges > size_x // 2 + 1):
            raise ValueError("Frequency slice bin edges must not exceed [1, size_x // 2 + 1].")
        self.freq_slicing = freq_slicing

        super()._init_fade_matrix()

        self.patch_spectrum = None
        if patch_spectrum is not None and not callable(patch_spectrum):
            self.patch_spectrum = patch_spectrum
        elif x_patch is not None and not callable(x_patch):
            self.patch_spectrum = rfft(x_patch)
        elif X_bg is not None:
            if callable(patch_spectrum):
                self.patch_spectrum = patch_spectrum(np.asarray(X_bg))
            elif callable(x_patch):
                self.patch_spectrum = rfft(x_patch(np.asarray(X_bg)))

        if self.patch_spectrum is not None:
            if len(self.patch_spectrum) != size_x // 2 + 1:
                raise ValueError(f"Length of patch_spectrum {len(self.patch_spectrum)} must match "
                                 f"size_x//2+1 of FreqDicePatchOmitter ({size_x // 2 + 1}).")
            self.patch_slices = [self.patch_spectrum[start:stop] for start, stop in freq_slicing.bin_slices]

    def _omit_band_profiles(self, x, band_profiles):
        if self.patch_spectrum is None:
            raise ValueError("This FreqDicePatchOmitter has no concrete patch_spectrum. "
                             "Have you maybe forgotten to pass in X_bg?")

        spectra = np.tile(rfft(x), (len(band_profiles), 1))
        replace_zero_slices(self.freq_slicing, self.patch_slices, self.patch_strat, spectra, band_profiles)
        return irfft(spectra)

    @staticmethod
    def max_n_freq_slices(size_x: int) -> int:
        return size_x // 2
