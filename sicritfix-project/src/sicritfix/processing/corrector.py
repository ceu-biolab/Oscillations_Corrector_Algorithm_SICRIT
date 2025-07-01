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
    
    
    local_amplitudes=[]
    
    
    for i, freq in enumerate (local_freqs):
        
        if freq<=0:
            continue
        
        period=int(1/(freq*sampling_interval))
        
        #if period<3: #para que sea al menos 3 si no es muy pequeña
            #continue
        
        center=i*int(len(xic) / len(local_freqs))
        start=int(max(0, center-period/2))
        end=int(min(len(xic), center+period/2))
        
        
        window=xic[start:end]
        #local_amplitude=np.max(window)-np.min(window)
        #local_amplitudes.append(local_amplitude)
        
        q25, q75 = np.percentile(window, [25, 75])
        local_amplitude = (q75 - q25) / 2
        local_amplitudes.append(local_amplitude)
        
        
    #amplitude=np.median(local_amplitudes)/4
    amplitude = np.percentile(local_amplitudes, 75)
    
    return amplitude
    
def generate_modulated_signal(amplitude, phase):
    """
    Generates the signal that is going to substract the original one at each m/z so the oscillations can be corrected

    Returns:
        - mmodulated_signal: 
    -------
    
    """
    
    modulated_signal = amplitude * np.sin(phase) 
    
    return modulated_signal
    
def correct_oscillations(rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz, window_size=70):
    """
    For each target mz, generates a modulated signal that is created with the frequency
    and amplitude of the signal. 
    
    Parameters:
        - 

    Returns: A residual signal for each mz that is the result of the original xic - modulated signal
    -------

    """
    #1. Extract XIC from original signal (intensities for each RT at target_mz)
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz)
    

    #2. Frequency with polynomial regression
    
    #TODO-> SHOULD BE CONSTANT FOR ALL MZ (EXTRACTED FROM MZ 922.098) BUT ADAPT WITH A 
    #POLYNOMIAL REGRESSION TO THE SIGNAL
    sampling_interval = np.mean(np.diff(rt_array))
    #rt_freqs, local_freqs = local_frequencies_with_fft(xic, rt_array, window_size, sampling_interval)
    #phase=apply_polynomial_regression(rt_array, rt_freqs, local_freqs)
    
    
    # 3. Amplitude at each m/z
    amplitude=get_amplitude(target_mz, xic, rt_array, local_freqs_ref, sampling_interval)
    #print(f"Amplitude for m/z: {target_mz} is: {amplitude}")
    
    # 4. Creation of the modulated signal
    modulated_signal = generate_modulated_signal(amplitude, phase_ref)
    
    # 5. Computation of the residual/final signal
    
    #baseline = np.median(xic)
    #modulated_signal = baseline + amplitude * np.sin(phase) señal centrada en la baseline de la xic
    #residual_signal = xic - (modulated_signal - baseline)#le resto solo la amplitud con respecto a esa baseline
    
    #que viene a ser lo mismo que hacer esto: 
    residual_signal = xic - modulated_signal
    #residual_signal_adjusted=np.clip(residual_signal, 0, None)
    
    
    return xic, modulated_signal, residual_signal
    
    