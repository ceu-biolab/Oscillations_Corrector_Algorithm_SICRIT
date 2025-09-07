#utils/intensity_analyzer.py

#!/usr/bin/env python

"""
This Python module provides intensity-based utilities for processing MS data,
specifically for building XICs and estimating local oscillation amplitudes.

@contents  :  Signal intensity tools: XIC construction and amplitude estimation.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  intensity_analyzer.py
@version   :  0.0.1, 18 July 2025
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    https://www.python.org/dev/peps/pep-0020/
    https://www.python.org/dev/peps/pep-0008/
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@dependencies :
    - numpy

@functions :
    - build_xic
    - get_amplitude

@notes :
    These functions are used to support frequency analysis and correction modeling
    in the SICRITfix pipeline by quantifying oscillation intensity behavior.

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""
__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"



import numpy as np

def build_xic(mz_array, intensity_array, rt_array, target_mz, rt_window=None, mz_tol=0.1):
    """
    Builds an Extracted Ion Chromatogram (XIC) for a target m/z value.

    This function extracts the signal intensity corresponding to a specified 
    m/z value across all retention times, within a given mass tolerance. 
    It sums the intensities of ions within the tolerance window for each scan 
    and returns the resulting intensity profile as the XIC.

    Parameters
    ----------
    mz_array : list of np.ndarray
        List containing arrays of m/z values for each scan (typically from MS1).

    intensity_array : list of np.ndarray
        List containing arrays of intensity values corresponding to each m/z 
        array in `mz_array`.

    rt_array : np.ndarray
        Array of retention times corresponding to each scan.

    target_mz : float
        The m/z value of interest to extract the chromatogram for.

    mz_tol : float, optional (default=0.1)
        Tolerance window around the target m/z. Peaks within 
        [target_mz - mz_tol, target_mz + mz_tol] will be included.

    Returns
    -------
    xic : np.ndarray
        1D array of summed intensities at each retention time for the 
        specified m/z window.
    """
    xic = []
    for mzs, intensities in zip(mz_array, intensity_array):
        is_in_tol = np.abs(mzs - target_mz) < mz_tol
        if np.any(is_in_tol):
            xic.append(np.sum(intensities[is_in_tol]))
        else:
            xic.append(0.0)
            
    xic=np.array(xic)
    
    # If rt_window size is specified: apply binning at RT
    if rt_window is not None:
        rt_min, rt_max = np.min(rt_array), np.max(rt_array)
        bins = np.arange(rt_min, rt_max + rt_window, rt_window)

        # Grouping of intensities in bins
        digitized = np.digitize(rt_array, bins)
        binned_xic = np.array([xic[digitized == i].sum() for i in range(1, len(bins))])
        binned_rt = bins[:-1] + rt_window/2

        return binned_rt, binned_xic

    # If rt_widow not specified → return raw XIC
    return rt_array, xic

def get_amplitude(target_mz, xic, rt_array, local_freqs, sampling_interval):
    
    """
    Estimates the amplitude of a signal in an extracted ion chromatogram (XIC)
    using local frequency information and percentile-based statistics.The final amplitude is taken 
    as the 75th percentile of the computed local amplitudes.

    Parameters
    ----------
    target_mz : float
        Target mass-to-charge ratio (m/z) of the ion of interest.
    
    xic : np.ndarray
        Extracted ion chromatogram.
    
    rt_array : np.ndarray
        Retention time array corresponding to the XIC.
    
    local_freqs : np.ndarray
        Array of local frequency estimates (in Hz) across the signal.
    
    sampling_interval : float
        Time interval between samples in the XIC (in seconds).

    Returns
    -------
    amplitude : float
        Estimated amplitude of the signal, based on the 75th percentile
        of the local amplitude estimates across all valid local frequencies.
    """
    
    local_amplitudes=[]
    
    
    for i, freq in enumerate (local_freqs):
        
        if freq<=0:
            continue
        
        period=int(1/(freq*sampling_interval))
        
        center=i*int(len(xic) / len(local_freqs))
        start=int(max(0, center-period/2))
        end=int(min(len(xic), center+period/2))
        window=xic[start:end]
        
        q25, q75 = np.percentile(window, [25, 75])
        local_amplitude = (q75 - q25) / 2
        local_amplitudes.append(local_amplitude)
        
        
    amplitude = np.percentile(local_amplitudes, 75)
    
    return amplitude
