#processing/corrector.py

#!/usr/bin/env python

"""
This Python module provides functionality to correct oscillations in MS signals
by modeling sinusoidal artifacts and subtracting them from the original signal.

@contents  :  Oscillation correction based on sinusoidal signal modeling.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  corrector.py
@version   :  0.0.1, 18 July 2025
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    https://www.python.org/dev/peps/pep-0020/
    https://www.python.org/dev/peps/pep-0008/
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@dependencies :
    - numpy
    - sicritfix.utils.intensity_analyzer

@functions :
    - generate_modulated_signal
    - correct_oscillations

@notes :
    The core logic assumes the oscillatory component is a single-frequency sinusoid
    estimated from local frequencies and reference phase information.

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""
__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"



import numpy as np
from sicritfix.utils.intensity_analyzer import build_xic, get_amplitude


def generate_modulated_signal(amplitude, phase, offset=0.0):
    """
    Generates a modulated sinusoidal signal for oscillation correction.
    Creates a sine wave based on the provided amplitude and phase.
    It is used to subtract from an original signal to correct for oscillatory artifacts at each m/z value.

    Parameters
    ----------
    amplitude : np.ndarray or float
        The amplitude (s) of the sinusoidal oscillation.

    phase : np.ndarray or float
        The phase(s) (in radians) of the sinusoidal oscillation.

    offset : np.ndarray or float, optional
        Baseline offset added to the sinusoid for visualization/model centering.

    Returns
    -------
    modulated_signal : np.ndarray or float
        The resulting modulated sinusoidal signal.
    """
    
    modulated_signal = offset + amplitude * np.sin(phase)
    
    return modulated_signal


def _estimate_period_points(local_freqs, sampling_interval, default_points=70):
    local_freqs = np.asarray(local_freqs, dtype=float)
    valid_freqs = local_freqs[np.isfinite(local_freqs) & (local_freqs > 0)]
    if valid_freqs.size == 0 or sampling_interval <= 0:
        return default_points
    median_freq = float(np.median(valid_freqs))
    return max(3, int(round(1.0 / (median_freq * sampling_interval))))


def _moving_average(signal, window_points):
    signal = np.asarray(signal, dtype=float)
    if window_points <= 1:
        return signal.copy()
    window_points = min(window_points, signal.size)
    if window_points <= 1:
        return signal.copy()
    if window_points % 2 == 0:
        window_points = max(1, window_points - 1)
    kernel = np.ones(window_points, dtype=float) / window_points
    return np.convolve(signal, kernel, mode="same")


def fit_phase_offset(xic, phase_ref, local_freqs_ref, sampling_interval):
    """
    Estimate the best per-m/z phase offset against the shared reference phase.

    The target XIC is detrended with a slow moving average so the fit focuses on
    the oscillatory component instead of the chromatographic baseline.
    """
    xic = np.asarray(xic, dtype=float)
    phase_ref = np.asarray(phase_ref, dtype=float)
    if xic.size == 0 or phase_ref.size != xic.size:
        return 0.0

    period_points = _estimate_period_points(local_freqs_ref, sampling_interval)
    trend_window = max(25, period_points * 5)
    if trend_window % 2 == 0:
        trend_window += 1

    detrended = xic - _moving_average(xic, trend_window)
    sin_ref = np.sin(phase_ref)
    cos_ref = np.cos(phase_ref)
    design = np.column_stack([sin_ref, cos_ref])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(design, detrended, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    sin_coeff, cos_coeff = coeffs
    if np.isclose(sin_coeff, 0.0) and np.isclose(cos_coeff, 0.0):
        return 0.0

    return float(np.arctan2(cos_coeff, sin_coeff))
    
def correct_oscillations(
    rt_array,
    mz_array,
    intensity_array,
    phase_ref,
    local_freqs_ref,
    target_mz,
    rt_window=5,
    mz_tol=0.01,
    debug_signals=False,
    amplitude_method="local_robust_detrended",
    amplitude_percentile=75,
    amplitude_multiplier=1.0,
):
    """
    Corrects oscillations in an extracted ion chromatogram (XIC) by subtracting a
    modulated sinusoidal signal based on local frequency and amplitude estimates.
    
    For a given target m/z, this function extracts the corresponding XIC, estimates the signal's amplitude using local 
    frequency data, generates a sinusoidal model of the oscillation using a reference phase, and subtracts it from the 
    original signal to produce a residual signal with reduced oscillatory artifacts.
    
       Parameters
       ----------
       rt_array : np.ndarray
           Retention time values corresponding to each scan.
    
       mz_array : np.ndarray
           Array of m/z values for all scans.
    
       intensity_array : np.ndarray
           Array of intensity values corresponding to each m/z and retention time.
    
       phase_ref : np.ndarray
           Reference phase array (in radians) for the sinusoidal oscillation.
    
       local_freqs_ref : np.ndarray
           Local frequency estimates (in Hz) corresponding to the XIC.
    
       target_mz : float
           The m/z value for which the oscillation correction is applied.
    
        rt_window : float, optional (default=1)
            Retention time window in seconds for smoothing the XIC. If 0.01, no smoothing is applied.

        mz_tol : float, optional (default=0.01)
            Tolerance window around the target m/z in Da. Peaks within 
            [target_mz - mz_tol, target_mz + mz_tol] will be included.
        
       Returns
       -------
       xic : np.ndarray
           The original extracted ion chromatogram at the target m/z.
    
       modulated_signal : np.ndarray
           The generated sinusoidal signal modeled from the phase and amplitude.
    
       residual_signal : np.ndarray
           The corrected signal obtained by subtracting the modulated signal 
           from the original XIC.
   """
    #1. Extract XIC from original signal (intensities for each RT at target_mz)
    xic = build_xic(
        mz_array,
        intensity_array,
        rt_array,
        target_mz,
        rt_window,
        mz_window=mz_tol,
    )
    

    #2. Frequency with polynomial regression
    sampling_interval = np.mean(np.diff(rt_array))
    
    
    # 3. Amplitude at each m/z
    amplitude = get_amplitude(
        xic,
        local_freqs_ref,
        sampling_interval,
        method=amplitude_method,
        summary_percentile=amplitude_percentile,
    )
    amplitude *= amplitude_multiplier
    
    
    # 4. Fit a per-m/z phase offset while keeping the chosen amplitude strategy.
    phase_offset = fit_phase_offset(xic, phase_ref, local_freqs_ref, sampling_interval)
    modulated_signal = generate_modulated_signal(amplitude, phase_ref + phase_offset)
    
    # 5. Computation of the residual/final signal
    residual_signal = xic - modulated_signal
    
    
    return xic, modulated_signal, residual_signal
    
    
