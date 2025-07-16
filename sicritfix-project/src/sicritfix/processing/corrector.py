#processing/corrector.py

import numpy as np


def build_xic(mz_array, intensity_array, rt_array, target_mz, mz_tol=0.1):
    """
    Builds XIC for a given m/z value
    
    Returns: Array of intensities along RT for that given m/z
    """
    xic = []
    for mzs, intensities in zip(mz_array, intensity_array):
        is_in_tol = np.abs(mzs - target_mz) < mz_tol
        if np.any(is_in_tol):
            xic.append(np.sum(intensities[is_in_tol]))
        else:
            xic.append(0.0)
            
    return np.array(xic)

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
    
def correct_oscillations(rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz, window_size=70):
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
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz)
    

    #2. Frequency with polynomial regression
    sampling_interval = np.mean(np.diff(rt_array))
    
    
    # 3. Amplitude at each m/z
    amplitude=get_amplitude(target_mz, xic, rt_array, local_freqs_ref, sampling_interval)
    #print(f"Amplitude for m/z: {target_mz} is: {amplitude}")
    
    # 4. Creation of the modulated signal
    modulated_signal = generate_modulated_signal(amplitude, phase_ref)
    
    # 5. Computation of the residual/final signal
    residual_signal = xic - modulated_signal
    
    
    return xic, modulated_signal, residual_signal
    
    