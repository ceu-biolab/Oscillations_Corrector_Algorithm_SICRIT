import numpy as np
from scipy.fftpack import fft
from scipy.integrate import cumulative_trapezoid

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

def local_frequencies_with_fft(xic, rts, window_size, sampling_interval):
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

    window_size : int
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
    step = window_size // 2

    for i in range(0, len(xic) - window_size, step):
        segment = xic[i:i+window_size]
        rt_segment = rts[i:i+window_size]
        
        _, _, dom_freq = calculate_freq(segment, sampling_interval)
        
        freqs.append(dom_freq)
        times.append(np.mean(rt_segment))

    return np.array(times), np.array(freqs)

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
    rts = np.array(rts)
    t = (rts - rts[0])
    
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(rts, freq_interp, freq_deg)
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)


    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    return phase 