#!/usr/bin/env python

"""
This Python module provides frequency-domain analysis tools to identify and model
oscillatory behavior in chromatographic signals from mass spectrometry data.

@contents  :  FFT-based frequency detection, local frequency estimation, and phase modeling.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  frequency_analyzer.py
@version   :  0.0.1, 18 July 2025
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    https://www.python.org/dev/peps/pep-0020/
    https://www.python.org/dev/peps/pep-0008/
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@dependencies :
    - numpy
    - scipy.fftpack
    - scipy.integrate
    - sicritfix.utils.intensity_analyzer

@functions :
    - calculate_freq
    - local_frequencies_with_fft
    - apply_polynomial_regression
    - obtain_freq_from_signal

@notes :
    Phase and frequency estimation is central to the SICRITfix correction algorithm,
    enabling accurate signal modeling for subtraction-based corrections.

@copyright :
    Copyright 2025 GNU AFFER
"""


import numpy as np
from scipy.fftpack import fft
from scipy.integrate import cumulative_trapezoid
from sicritfix.utils.intensity_analyzer import build_xic


def calculate_freq(xic, sampling_interval=1.0):
    """
    Estimates the dominant frequency of a signal using the Fast Fourier Transform (FFT).

    This function analyzes the frequency content of an extracted ion chromatogram (XIC)
    by computing its FFT. It returns the positive frequencies, their magnitudes, and the
    dominant frequency (i.e., the frequency with the highest spectral magnitude).

    Parameters
    ----------
    xic : np.ndarray
        Array of signal intensities (e.g., extracted ion chromatogram over time).

    sampling_interval : float, optional (default=1.0)
        Time between samples in the signal, in the same units as the desired frequency output.
        For example, if the signal is sampled once per second, use 1.0.

    Returns
    -------
    fft_freqs : np.ndarray
        Array of positive frequency components (in cycles per unit time).

    fft_magnitude : np.ndarray
        Corresponding magnitude of each frequency component in the spectrum.

    main_freq : float
        The dominant frequency component (i.e., frequency with the highest magnitude).
    """
    centered_signal = xic - np.mean(xic)
    fft_result = fft(centered_signal)
    freqs = np.fft.fftfreq(len(centered_signal), d=sampling_interval)
    
    #Positive freqs
    pos_mask = freqs > 0
    fft_freqs = freqs[pos_mask]
    fft_magnitude = np.abs(fft_result[pos_mask])
    main_freq = fft_freqs[np.argmax(fft_magnitude)]
        

    return fft_freqs, fft_magnitude, main_freq

def local_frequencies_with_fft(xic, rts, window_scan_size, sampling_interval):
    """
    Estimates local dominant frequencies in a signal using a sliding window FFT approach.

    This function divides the extracted ion chromatogram (XIC) into overlapping 
    segments and calculates the dominant frequency in each window using FFT. 
    It returns the local frequencies along with their corresponding central retention times.

    Parameters
    ----------
    xic : np.ndarray
        The intensity signal (e.g., extracted ion chromatogram over time).

    rts : np.ndarray
        Array of retention times corresponding to each point in the XIC.

    window_scan_size : int
        Number of points in each sliding window used to estimate local frequency.

    sampling_interval : float
        Time between consecutive samples in the XIC, in the same units as `rts`.

    Returns
    -------
    times : np.ndarray
        Array of mean retention times for each analyzed window.

    freqs : np.ndarray
        Array of dominant frequencies (in Hz) estimated for each window.
    """
    
    freqs = []
    times = []
    step = window_scan_size // 2

    for i in range(0, len(xic) - window_scan_size, step):
        segment = xic[i:i+window_scan_size]
        rt_segment = rts[i:i+window_scan_size]
        
        _, _, dom_freq = calculate_freq(segment, sampling_interval)
        
        freqs.append(dom_freq)
        times.append(np.mean(rt_segment))

    return np.array(times), np.array(freqs)


def find_stable_signal_start(xic, intensity_threshold=0.0, min_consecutive_scans=1):
    """
    Find the first scan where the reference signal has reached a stable level.

    This is used to ignore the reference-mass insertion ramp before estimating
    phase/frequency. If the threshold is disabled or is never reached, the full
    signal is used from scan 0.
    """
    xic = np.asarray(xic, dtype=float)
    if xic.size == 0 or intensity_threshold is None or intensity_threshold <= 0:
        return 0

    min_consecutive_scans = int(max(1, min_consecutive_scans))
    above_threshold = xic >= float(intensity_threshold)
    if not np.any(above_threshold):
        return 0

    run_start = None
    run_length = 0
    for idx, is_above in enumerate(above_threshold):
        if is_above:
            if run_start is None:
                run_start = idx
            run_length += 1
            if run_length >= min_consecutive_scans:
                return int(run_start)
        else:
            run_start = None
            run_length = 0

    return int(np.argmax(above_threshold))


def find_stable_signal_range(xic, intensity_threshold=0.0, min_consecutive_scans=1):
    """
    Find the scan range where the reference signal is stable enough to correct.

    The returned end index is exclusive. When the threshold is disabled or the
    signal never reaches it, the full signal is returned to preserve legacy
    behavior.
    """
    xic = np.asarray(xic, dtype=float)
    if xic.size == 0:
        return 0, 0
    if intensity_threshold is None or intensity_threshold <= 0:
        return 0, int(xic.size)

    min_consecutive_scans = int(max(1, min_consecutive_scans))
    above_threshold = xic >= float(intensity_threshold)
    if not np.any(above_threshold):
        return 0, int(xic.size)

    stable_start_idx = find_stable_signal_start(
        xic,
        intensity_threshold=intensity_threshold,
        min_consecutive_scans=min_consecutive_scans,
    )

    run_end = None
    run_length = 0
    for idx in range(xic.size - 1, stable_start_idx - 1, -1):
        if above_threshold[idx]:
            if run_end is None:
                run_end = idx
            run_length += 1
            if run_length >= min_consecutive_scans:
                return int(stable_start_idx), int(run_end + 1)
        else:
            run_end = None
            run_length = 0

    return int(stable_start_idx), int(xic.size)


def _rolling_quantile(signal, window_points, quantile):
    signal = np.asarray(signal, dtype=float)
    window_points = int(max(1, min(window_points, signal.size)))
    if window_points <= 1:
        return signal.copy()
    if window_points % 2 == 0:
        window_points = max(1, window_points - 1)
    half = window_points // 2
    padded = np.pad(signal, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_points)
    return np.quantile(windows, quantile, axis=-1)


def _detrend_reference_signal(xic, sampling_interval, baseline_seconds=300.0):
    if len(xic) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    baseline_points = max(3, int(round(float(baseline_seconds) / sampling_interval)))
    baseline = _rolling_quantile(xic, baseline_points, 0.5)
    return xic - baseline, baseline


def _weighted_corr(x, y, weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    if np.sum(valid) < 3:
        return 0.0
    x = x[valid]
    y = y[valid]
    weights = weights[valid]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0
    x_mean = float(np.sum(weights * x) / weight_sum)
    y_mean = float(np.sum(weights * y) / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.sqrt(np.sum(weights * x_centered**2) * np.sum(weights * y_centered**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * x_centered * y_centered) / denom)


def _fit_phase_offset(oscillation, phase, weights):
    oscillation = np.asarray(oscillation, dtype=float)
    phase = np.asarray(phase, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(oscillation) & np.isfinite(phase) & np.isfinite(weights)
    if np.sum(valid) < 3:
        return 0.0
    design = np.column_stack([np.sin(phase[valid]), np.cos(phase[valid])])
    sqrt_weights = np.sqrt(weights[valid])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            oscillation[valid] * sqrt_weights,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.arctan2(coeffs[1], coeffs[0]))


def _phase_from_constant_frequency(rt_array, frequency, start_idx, end_idx):
    phase = np.zeros_like(np.asarray(rt_array, dtype=float))
    if frequency <= 0 or start_idx >= end_idx:
        return phase
    rt_array = np.asarray(rt_array, dtype=float)
    t = rt_array[start_idx:end_idx] - rt_array[start_idx]
    phase[start_idx:end_idx] = 2.0 * np.pi * frequency * t
    return phase


def _estimate_median_frequency(local_freqs, fallback=0.0):
    local_freqs = np.asarray(local_freqs, dtype=float)
    valid = local_freqs[np.isfinite(local_freqs) & (local_freqs > 0)]
    if valid.size == 0:
        return float(fallback)
    return float(np.median(valid))


def _window_correlations(rt_array, oscillation, phase, weights, start_idx, end_idx, window_seconds, step_seconds):
    windows = []
    if start_idx >= end_idx:
        return windows

    rt_array = np.asarray(rt_array, dtype=float)
    model = np.sin(phase)
    current_start = float(rt_array[start_idx])
    final_rt = float(rt_array[end_idx - 1])
    while current_start <= final_rt:
        current_end = current_start + window_seconds
        idx = np.where(
            (rt_array >= current_start)
            & (rt_array < current_end)
            & (np.arange(rt_array.size) >= start_idx)
            & (np.arange(rt_array.size) < end_idx)
        )[0]
        if idx.size >= 5:
            windows.append(
                {
                    "start_idx": int(idx[0]),
                    "end_idx": int(idx[-1] + 1),
                    "start_rt": float(rt_array[idx[0]]),
                    "end_rt": float(rt_array[idx[-1]]),
                    "corr": _weighted_corr(oscillation[idx], model[idx], weights[idx]),
                }
            )
        current_start += step_seconds
    return windows


def _merge_bad_windows(windows, threshold):
    intervals = []
    for window in windows:
        if window["corr"] >= threshold:
            continue
        if not intervals or window["start_idx"] > intervals[-1]["end_idx"]:
            intervals.append(
                {
                    "start_idx": window["start_idx"],
                    "end_idx": window["end_idx"],
                    "min_corr": window["corr"],
                }
            )
        else:
            intervals[-1]["end_idx"] = max(intervals[-1]["end_idx"], window["end_idx"])
            intervals[-1]["min_corr"] = min(intervals[-1]["min_corr"], window["corr"])
    return intervals


def _median_frequency_in_interval(rt_freqs, local_freqs, start_rt, end_rt, fallback):
    rt_freqs = np.asarray(rt_freqs, dtype=float)
    local_freqs = np.asarray(local_freqs, dtype=float)
    mask = (rt_freqs >= start_rt) & (rt_freqs <= end_rt) & np.isfinite(local_freqs) & (local_freqs > 0)
    if np.any(mask):
        return float(np.median(local_freqs[mask]))
    return float(fallback)


def _integrate_frequency_profile(rt_array, freq_profile, start_idx, end_idx):
    phase = np.zeros_like(np.asarray(rt_array, dtype=float))
    if start_idx >= end_idx:
        return phase
    rt_segment = np.asarray(rt_array, dtype=float)[start_idx:end_idx]
    freq_segment = np.asarray(freq_profile, dtype=float)[start_idx:end_idx]
    phase[start_idx:end_idx] = 2.0 * np.pi * cumulative_trapezoid(
        freq_segment,
        rt_segment - rt_segment[0],
        initial=0,
    )
    return phase


def apply_polynomial_regression(rts, rt_freqs, local_freqs, freq_deg=2):
    
    """
    Smooths local frequency estimates using polynomial regression and computes the accumulated phase.

    This function interpolates the local frequency estimates to match the full retention time array,
    fits a polynomial of specified degree to the interpolated data, and uses it to calculate a smoothed
    frequency profile. It then integrates this frequency profile over time to obtain the accumulated phase.

    Parameters
    ----------
    rts : array-like
        Full array of retention times (in seconds).

    rt_freqs : array-like
        Retention times corresponding to the original local frequency estimates.

    local_freqs : array-like
        Estimated local dominant frequencies (in Hz) at `rt_freqs`.

    freq_deg : int, optional (default=2)
        Degree of the polynomial used to smooth the frequency data.

    Returns
    -------
    phase : np.ndarray
        Accumulated phase (in radians) computed by integrating the smoothed frequency
        over time.
    """
    rts = np.array(rts, dtype=float)
    rt_freqs = np.array(rt_freqs, dtype=float)
    local_freqs = np.array(local_freqs, dtype=float)

    if rts.size == 0:
        return np.array([], dtype=float)

    # No local frequency estimates -> no oscillatory phase progression.
    if rt_freqs.size == 0 or local_freqs.size == 0:
        return np.zeros_like(rts, dtype=float)

    t = (rts - rts[0])
    
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(t, freq_interp, freq_deg)
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)


    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    return phase 


def build_dual_reference_phase(
    rt_array,
    main_xic,
    secondary_xic,
    window_scan_size,
    sampling_interval,
    stable_start_idx,
    stable_end_idx,
    quality_window_minutes=4.0,
    quality_step_minutes=2.0,
    quality_corr_threshold=0.5,
    baseline_seconds=300.0,
):
    rt_array = np.asarray(rt_array, dtype=float)
    main_xic = np.asarray(main_xic, dtype=float)
    secondary_xic = np.asarray(secondary_xic, dtype=float)

    main_rt_freqs, main_local_freqs = local_frequencies_with_fft(
        main_xic[stable_start_idx:stable_end_idx],
        rt_array[stable_start_idx:stable_end_idx],
        window_scan_size=window_scan_size,
        sampling_interval=sampling_interval,
    )
    secondary_rt_freqs, secondary_local_freqs = local_frequencies_with_fft(
        secondary_xic[stable_start_idx:stable_end_idx],
        rt_array[stable_start_idx:stable_end_idx],
        window_scan_size=window_scan_size,
        sampling_interval=sampling_interval,
    )

    main_frequency = _estimate_median_frequency(main_local_freqs)
    secondary_frequency = _estimate_median_frequency(secondary_local_freqs, fallback=main_frequency)
    stable_rt_array = rt_array[stable_start_idx:stable_end_idx]

    main_quality_phase = _phase_from_constant_frequency(
        rt_array,
        main_frequency,
        stable_start_idx,
        stable_end_idx,
    )
    secondary_quality_phase = _phase_from_constant_frequency(
        rt_array,
        secondary_frequency,
        stable_start_idx,
        stable_end_idx,
    )

    main_oscillation, _ = _detrend_reference_signal(main_xic, sampling_interval, baseline_seconds=baseline_seconds)
    secondary_oscillation, _ = _detrend_reference_signal(secondary_xic, sampling_interval, baseline_seconds=baseline_seconds)

    main_weights = np.abs(main_oscillation)
    secondary_weights = np.abs(secondary_oscillation)
    valid_slice = slice(stable_start_idx, stable_end_idx)
    main_high_weight = float(np.percentile(main_weights[valid_slice], 95)) if stable_end_idx > stable_start_idx else 0.0
    secondary_high_weight = float(np.percentile(secondary_weights[valid_slice], 95)) if stable_end_idx > stable_start_idx else 0.0
    main_weights = np.clip(main_weights / main_high_weight, 0.05, 1.0) if main_high_weight > 0 else np.ones_like(main_weights)
    secondary_weights = np.clip(secondary_weights / secondary_high_weight, 0.05, 1.0) if secondary_high_weight > 0 else np.ones_like(secondary_weights)

    main_offset = _fit_phase_offset(
        main_oscillation[valid_slice],
        main_quality_phase[valid_slice],
        main_weights[valid_slice],
    )
    secondary_offset = _fit_phase_offset(
        secondary_oscillation[valid_slice],
        secondary_quality_phase[valid_slice],
        secondary_weights[valid_slice],
    )

    quality_windows = _window_correlations(
        rt_array,
        main_oscillation,
        main_quality_phase + main_offset,
        main_weights,
        stable_start_idx,
        stable_end_idx,
        window_seconds=quality_window_minutes * 60.0,
        step_seconds=quality_step_minutes * 60.0,
    )
    bad_intervals = _merge_bad_windows(quality_windows, quality_corr_threshold)

    corrected_local_freqs = np.asarray(main_local_freqs, dtype=float).copy()
    bad_local_mask = np.zeros_like(corrected_local_freqs, dtype=bool)
    for interval in bad_intervals:
        start_rt = float(rt_array[interval["start_idx"]])
        end_rt = float(rt_array[interval["end_idx"] - 1])
        bad_local_mask |= (main_rt_freqs >= start_rt) & (main_rt_freqs <= end_rt)

    valid_main_local = (
        np.isfinite(main_rt_freqs)
        & np.isfinite(main_local_freqs)
        & (main_local_freqs > 0)
        & ~bad_local_mask
    )
    if np.sum(valid_main_local) >= 2:
        interpolated_main_freqs = np.interp(
            main_rt_freqs,
            main_rt_freqs[valid_main_local],
            main_local_freqs[valid_main_local],
        )
    else:
        interpolated_main_freqs = np.full_like(corrected_local_freqs, main_frequency, dtype=float)

    source_intervals = []
    for interval in bad_intervals:
        start_idx = interval["start_idx"]
        end_idx = interval["end_idx"]
        start_rt = float(rt_array[start_idx])
        end_rt = float(rt_array[end_idx - 1])
        local_interval_mask = (main_rt_freqs >= start_rt) & (main_rt_freqs <= end_rt)
        secondary_corr = _weighted_corr(
            secondary_oscillation[start_idx:end_idx],
            np.sin(secondary_quality_phase[start_idx:end_idx] + secondary_offset),
            secondary_weights[start_idx:end_idx],
        )

        if secondary_corr >= quality_corr_threshold:
            valid_secondary_local = np.isfinite(secondary_local_freqs) & (secondary_local_freqs > 0)
            if np.any(local_interval_mask) and np.sum(valid_secondary_local) >= 2:
                corrected_local_freqs[local_interval_mask] = np.interp(
                    main_rt_freqs[local_interval_mask],
                    secondary_rt_freqs[valid_secondary_local],
                    secondary_local_freqs[valid_secondary_local],
                )
            interval_values = corrected_local_freqs[local_interval_mask]
            interval_frequency = _estimate_median_frequency(interval_values, fallback=secondary_frequency)
            source = "secondary_reference"
        else:
            if np.any(local_interval_mask):
                corrected_local_freqs[local_interval_mask] = interpolated_main_freqs[local_interval_mask]
            interval_values = corrected_local_freqs[local_interval_mask]
            interval_frequency = _estimate_median_frequency(interval_values, fallback=main_frequency)
            source = "interpolated_frequency"

        source_intervals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_rt": start_rt,
                "end_rt": end_rt,
                "start_min": start_rt / 60.0,
                "end_min": end_rt / 60.0,
                "main_min_corr": float(interval["min_corr"]),
                "secondary_corr": float(secondary_corr),
                "frequency_hz": float(interval_frequency),
                "source": source,
            }
        )

    phase = np.zeros_like(rt_array, dtype=float)
    if stable_rt_array.size:
        phase[stable_start_idx:stable_end_idx] = apply_polynomial_regression(
            stable_rt_array,
            main_rt_freqs,
            corrected_local_freqs,
        )
    metadata = {
        "main_frequency_hz": float(main_frequency),
        "secondary_frequency_hz": float(secondary_frequency),
        "quality_window_minutes": float(quality_window_minutes),
        "quality_step_minutes": float(quality_step_minutes),
        "quality_corr_threshold": float(quality_corr_threshold),
        "quality_windows": quality_windows,
        "source_intervals": source_intervals,
    }
    return main_rt_freqs, corrected_local_freqs, phase, metadata


def obtain_freq_from_signal(
    rt_array,
    mz_array,
    intensity_array,
    rt_window,
    window_scan_size=70,
    mz_ref=922.098,
    secondary_mz_ref=None,
    mz_window=0.1,
    ignore_until_intensity=0.0,
    ignore_consecutive_scans=1,
    quality_window_minutes=4.0,
    quality_step_minutes=2.0,
    quality_corr_threshold=0.5,
    return_metadata=False,
):
    """
    Estimates the local frequency and phase of oscillations from a given reference m/z signal.

    Extracts the XIC at a reference m/z, computes local frequency estimates using a windowed FFT-based approach, 
    and fits a polynomial regression to obtain a smooth phase signal. The output is used to correct oscillatory 
    behavior in related signals.

    Parameters
    ----------
    rt_array : np.ndarray
        Retention time values for each scan.

    mz_array : np.ndarray
        Array of m/z values for each scan.

    intensity_array : np.ndarray
        Intensity values corresponding to each m/z and retention time.

    rt_window : float
        Retention time window (in seconds) used to extract the XIC around the reference m/z value.

    window_scan_size : int, optional (default=70)
        Size of the sliding window (in scans) used for local frequency estimation.

    mz_ref : float, optional (default=922.098)
        Reference m/z value used to extract the XIC for frequency analysis.

    secondary_mz_ref : float, optional (default=None)
        Secondary reference m/z used to provide frequency estimates in local
        regions where the main reference has poor sine correlation.
    
    mz_window : float, optional (default=0.1)
        m/z tolerance window (in Da) around the reference m/z for extracting the XIC. Peaks within [mz_ref - mz_window, mz_ref + mz_window] will be included in the analysis.

    ignore_until_intensity : float, optional (default=0.0)
        If greater than zero, scans before the reference XIC reaches this
        intensity for `ignore_consecutive_scans` consecutive scans are ignored
        for frequency/phase estimation.

    ignore_consecutive_scans : int, optional (default=1)
        Number of consecutive scans above `ignore_until_intensity` required to
        mark the start of the stable reference-mass region.

    quality_window_minutes : float, optional (default=4.0)
        Window length used to score local main-reference phase quality.

    quality_step_minutes : float, optional (default=2.0)
        Step between quality windows.

    quality_corr_threshold : float, optional (default=0.5)
        Windows below this correlation are treated as low quality.

    Returns
    -------
    local_freqs_ref : np.ndarray
        Estimated local frequencies (in Hz) along the retention time.

    phase_ref : np.ndarray
        Smoothed phase (in radians) derived from polynomial regression on frequency data.
    """
    rt_array = np.asarray(rt_array, dtype=float)
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz=mz_ref, rt_window=rt_window, mz_window=mz_window)

    
    
    sampling_interval = np.mean(np.diff(rt_array))

    stable_start_idx, stable_end_idx = find_stable_signal_range(
        xic,
        intensity_threshold=ignore_until_intensity,
        min_consecutive_scans=ignore_consecutive_scans,
    )
    stable_rt_array = rt_array[stable_start_idx:stable_end_idx]
    stable_xic = xic[stable_start_idx:stable_end_idx]

    dual_reference_metadata = {}
    if secondary_mz_ref is not None:
        secondary_xic = build_xic(
            mz_array,
            intensity_array,
            rt_array,
            target_mz=secondary_mz_ref,
            rt_window=rt_window,
            mz_window=mz_window,
        )
        rt_freqs, local_freqs_ref, phase_ref, dual_reference_metadata = build_dual_reference_phase(
            rt_array,
            xic,
            secondary_xic,
            window_scan_size=window_scan_size,
            sampling_interval=sampling_interval,
            stable_start_idx=stable_start_idx,
            stable_end_idx=stable_end_idx,
            quality_window_minutes=quality_window_minutes,
            quality_step_minutes=quality_step_minutes,
            quality_corr_threshold=quality_corr_threshold,
        )
    else:
        rt_freqs, local_freqs_ref = local_frequencies_with_fft(
            stable_xic,
            stable_rt_array,
            window_scan_size=window_scan_size,
            sampling_interval=sampling_interval,
        )
        phase_ref = np.zeros_like(rt_array, dtype=float)
        if stable_rt_array.size:
            phase_ref[stable_start_idx:stable_end_idx] = apply_polynomial_regression(
                stable_rt_array,
                rt_freqs,
                local_freqs_ref,
            )
    
    if return_metadata:
        metadata = {
            "stable_start_idx": int(stable_start_idx),
            "stable_end_idx": int(stable_end_idx),
            "stable_start_rt": float(rt_array[stable_start_idx]) if rt_array.size else 0.0,
            "stable_end_rt": float(rt_array[stable_end_idx - 1]) if stable_end_idx > stable_start_idx else 0.0,
            "main_reference_mz": float(mz_ref),
            "secondary_reference_mz": float(secondary_mz_ref) if secondary_mz_ref is not None else None,
            "ignore_until_intensity": float(ignore_until_intensity or 0.0),
            "ignore_consecutive_scans": int(max(1, ignore_consecutive_scans)),
            "dual_reference": dual_reference_metadata,
        }
        return local_freqs_ref, phase_ref, metadata

    return local_freqs_ref, phase_ref
