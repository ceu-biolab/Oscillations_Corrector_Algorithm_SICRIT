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


def generate_modulated_signal(amplitude, phase):
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

    Returns
    -------
    modulated_signal : np.ndarray or float
        The resulting modulated sinusoidal signal.
    """
    
    modulated_signal = amplitude * np.sin(phase) 
    
    return modulated_signal
    
def correct_oscillations(rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz, rt_window, window_size=70):
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
    
       window_size : int, optional (default=70)
           The size of the window (in scans) used for extracting the XIC 
           around the target m/z.
    
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
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz, rt_window)
    

    #2. Frequency with polynomial regression
    sampling_interval = np.mean(np.diff(rt_array))
    
    
    # 3. Amplitude at each m/z
    amplitude=get_amplitude(target_mz, xic, rt_array, local_freqs_ref, sampling_interval)
    
    
    # 4. Creation of the modulated signal
    modulated_signal = generate_modulated_signal(amplitude, phase_ref)
    
    # 5. Computation of the residual/final signal
    residual_signal = xic - modulated_signal
    
    
    return xic, modulated_signal, residual_signal
    
    