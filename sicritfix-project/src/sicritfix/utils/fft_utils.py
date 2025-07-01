import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.integrate import cumulative_trapezoid

def calculate_freq(xic, sampling_interval=1.0, plot_spectrum=False):
    """
    Estima la frecuenciaa dominante usando fft
    
    Parameters
    ----------
    intensities : signal intensities
        DESCRIPTION.
    sampling_interval : TYPE, optional
        DESCRIPTION. The default is 1.0.
    plot_spectrum : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fft_freqs : TYPE
        DESCRIPTION.
    fft_magnitude : TYPE
        DESCRIPTION.
    main_freq : TYPE
        DESCRIPTION.

    """
    centered_signal = xic - np.mean(xic)
    fft_result = fft(centered_signal)
    freqs = np.fft.fftfreq(len(centered_signal), d=sampling_interval)
    
    #Solo frecuencias positivas
    pos_mask = freqs > 0
    fft_freqs = freqs[pos_mask]
    fft_magnitude = np.abs(fft_result[pos_mask])
    
    main_freq = fft_freqs[np.argmax(fft_magnitude)]

    if plot_spectrum:
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freqs, fft_magnitude, color='darkgreen')
        plt.title("Espectro de Frecuencia (FFT)")
        plt.xlabel("Frecuencia (ciclos/minuto)")
        plt.ylabel("Magnitud")
        plt.tight_layout()
        plt.show()
        

    return fft_freqs, fft_magnitude, main_freq

def local_frequencies_with_fft(xic, rts, window_size, sampling_interval):
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
    rts = np.array(rts)
    t = (rts - rts[0])
    
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(rts, freq_interp, freq_deg)#ajusta el polinomio a los datos
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)#frecuencia suavizada en cada punto t

    # 3. Calcular fase acumulada φ(t) con integración
    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    return phase 