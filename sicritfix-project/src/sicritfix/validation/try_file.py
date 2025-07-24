# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fftpack import fft
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks


#Calculating signal

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

def obtain_amplitudes(mzs, intensities, bin_size):
    min_mz = np.min(mzs)
    max_mz = np.max(mzs)
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)
    amplitudes, _ = np.histogram(mzs, bins, weights=intensities)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return amplitudes, bin_centers

def compute_local_amplitudes(signal):
    """
    Calcula la amplitud local como mediana de picos - mediana de valles.
    """

    # Encuentra máximos y mínimos
    peaks, _ = find_peaks(signal)
    valleys, _ = find_peaks(-signal)
    
    if len(peaks) > 0 and len(valleys)>0:
        peak_vals = signal[peaks] 
        valley_vals = signal[valleys]
        amplitude = np.median(peak_vals) - np.median(valley_vals)
    else:
        peak_vals=[0]
    
    return amplitude

def build_xic(mz_array, intensity_array, rt_array, target_mz, mz_tol=0.01):
    """
    Construye el XIC para un m/z de referencia.
    """
    xic = []
    for mzs, intensities in zip(mz_array, intensity_array):
        mask = np.abs(mzs - target_mz) < mz_tol
        if np.any(mask):
            xic.append(np.sum(intensities[mask]))
        else:
            xic.append(0.0)
    return np.array(xic)

# 4. Plotting
def plot_signal_with_features(rt_array, xic_signal, amplitudes, freq):
    amplitude = np.median(amplitudes)
    plt.figure(figsize=(12, 5))
    plt.plot(rt_array, xic_signal, label="XIC Signal", color='darkblue')
    plt.axhline(np.median(xic_signal), color='gray', linestyle='--', label='Median baseline')
    plt.axhline(np.median(xic_signal) + amplitude / 2, color='green', linestyle='--', label='Estimated Peak')
    plt.axhline(np.median(xic_signal) - amplitude / 2, color='red', linestyle='--', label='Estimated Trough')
    plt.title(f"XIC with Frequency = {freq:.4f} Hz | Amplitude = {amplitude:.2e}")
    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def apply_savgol_filter(intensities, window_length, filter_order):
    """
    Aplica un filtro Savitzky-Golay para suavizar la señal.

    Args:
        intensities (array-like): Intensidades de la señal.
        window_length (int): Tamaño de ventana (debe ser impar).
        filter_order (int): Orden del polinomio del filtro.

    Returns:
        np.array: Intensidades suavizadas.
    """
    if len(intensities) > window_length:
        smoothed = savgol_filter(intensities, window_length, filter_order)
    else:
        smoothed = np.array(intensities)  # Sin suavizado si es muy corta

    return smoothed

def normalize_to_range(array, lower, upper):
    """
    Normaliza un array a un rango específico [lower, upper].

    Args:
        array (np.array): Array original (valores numéricos).
        lower (float): Límite inferior del nuevo rango.
        upper (float): Límite superior del nuevo rango.

    Returns:
        np.array: Array escalado a [lower, upper].
    """
    min_val = np.min(array)
    max_val = np.max(array)

    if max_val == min_val:
        return np.full_like(array, (lower + upper) / 2)  # evita división por cero

    normalized = (array - min_val) / (max_val - min_val)
    return normalized * (upper - lower) + lower

def local_frequencies_with_fft(xic, rt, window_size, sampling_interval):
    freqs = []
    times = []
    step = window_size // 2

    for i in range(0, len(xic) - window_size, step):
        segment = xic[i:i+window_size]
        rt_segment = rt[i:i+window_size]
        
        _, _, dom_freq = calculate_freq(segment, sampling_interval)
        
        freqs.append(dom_freq)
        times.append(np.mean(rt_segment))

    return np.array(times), np.array(freqs)

def extract_amplitudes_at_mz(input_map, target_mz=922.098, bin_size=0.01, window_length=15, filter_order=3, plot=False):
    """
    Extrae amplitudes en un m/z específico a lo largo del RT de un archivo mzML cargado con pyOpenMS.
    
    Args:
        input_map: objeto MSExperiment cargado con pyOpenMS.
        target_mz (float): m/z objetivo.
        bin_size (float): tamaño del bin para histogramas.
        window_length (int): ventana del filtro Savitzky-Golay.
        filter_order (int): orden del polinomio para el filtro.
        plot (bool): si True, genera gráfico.
        
    Returns:
        rt_array (np.array): tiempos de retención.
        amplitude_array (np.array): amplitudes extraídas en target_mz.
    """
    amplitude_list = []
    rt_list = []
    mz_tol=0.09

    for spectrum in input_map:
        mzs, intensities = spectrum.get_peaks()
        rt = spectrum.getRT()

        # Suaviza intensidades
        smoothed_intensities = apply_savgol_filter(intensities, window_length, filter_order)
        
        
        # Calcula histogramas (bins)
        amplitudes, bin_centers = obtain_amplitudes(mzs, smoothed_intensities, bin_size)


        # Busca el índice del mz más cercano
        idx_closest = np.argmin(np.abs(mzs - target_mz))
        mz_closest = mzs[idx_closest]


        # Si está dentro de la tolerancia, guarda su intensidad
        if np.abs(mz_closest - target_mz) < mz_tol:
            amplitude = smoothed_intensities[idx_closest]
        else:
            amplitude = 0.0

        amplitude_list.append(amplitude)
        rt_list.append(rt)

    rt_array = np.array(rt_list)
    amplitude_array = np.array(amplitude_list)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(rt_array, amplitude_array, color='darkred')
        plt.xlabel("Retention Time (RT)")
        plt.ylabel(f"Amplitude at m/z {target_mz}")
        plt.title(f"Amplitudes over RT at m/z ≈ {target_mz}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rt_array, amplitude_array

#con el polinomio y amplitudes
def build_variable_frequency_sine_and_plot(input_map, mz_array, rts, tic_original, freq_deg=2, window_size=70):
    """
    POLINOMIO Y AMPLITUDES
    """
    
    rts = np.array(rts)
    t = (rts - rts[0])
    
    #Smooth original tic signal
    
    smoothed_tic = apply_savgol_filter(tic_original, window_length=15, filter_order=3)
   
    # Paso 1: calcular frecuencias locales con tu FFT y amplitudes para ese mz
    _, amplitudes=extract_amplitudes_at_mz(input_map)
    
    sampling_interval = np.mean(np.diff(rts))
    rt_freqs, local_freqs = local_frequencies_with_fft(amplitudes, rts, window_size, sampling_interval)

    # Paso 2: ajustar polinomio f(t)
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(rts, freq_interp, freq_deg)#ajusta el polinomio a los datos
    freq_poly = np.poly1d(fit)
    print(f"Freq poly: {freq_poly}")
    f_t = freq_poly(t)#frecuencia suavizada en cada punto t

    # 3. Calcular fase acumulada φ(t) con integración
    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    
    # 4. Normalizar amplitudes a un rango razonable del TIC
    target_range = 3.3e7 - 2.7e7    # = 6e6
    #target_center = 2.7e7 + target_range / 2   # = 3e7
    #scale_factor=0.07
    #max_amp=np.max(amplitudes)
    if np.max(np.abs(amplitudes)) != 0:
        #tic_range = np.max(smoothed_tic) - np.min(smoothed_tic)
        amp_norm = amplitudes / np.max(np.abs(amplitudes))
        amp_scaled = amp_norm * (target_range/2)
        #amp_scaled = amplitudes / max_amp * np.max(smoothed_tic)*scale_factor
    else:
        amplitudes

    # 5. Construcción de señal modulada
    offset = np.median(smoothed_tic)
    modulated_signal = amp_scaled * np.sin(phase)+ offset
    
    #PARA COMPROBAR QUE EL POLINOMIO ESTÁ BIEN
    plt.figure(figsize=(8, 4))
    plt.plot(rt_freqs, local_freqs, 'o', label='Frecuencia local', alpha=0.5)
    plt.plot(rts, f_t, '-', label=f'Ajuste polinomial (grado {freq_deg})', color='orange')
    plt.xlabel("Retention Time")
    plt.ylabel("Frecuencia estimada (Hz)")
    plt.title("Ajuste de frecuencia en función de RT")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Paso 5: plot
    plt.figure(figsize=(14, 5))
    plt.plot(rts, smoothed_tic, label="TIC suavizado", color="gray")
    plt.plot(rts, modulated_signal, label="Senoide modulada", color="green", linestyle="--")
    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Intensidad")
    plt.title("TIC vs Señal Senoidal con Frecuencia Variable")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return smoothed_tic, modulated_signal

#para calcular la media de la señal
def substract_modulated_baseline(rts, smoothed_tic, modulated_signal, use_mean=True, clip_negative=True, plot=True):
    """
    Resta el baseline de la señal modulada a la señal original.
    
    Args:
        rts (array): tiempos de retención.
        smoothed_tic (array): señal TIC suavizada original.
        modulated_signal (array): señal senoide modulada.
        use_mean (bool): si True, usa solo la media de la modulated_signal; si False, resta punto a punto.
        clip_negative (bool): si True, pone a cero los valores negativos.
        plot (bool): si True, grafica los resultados.
    
    Returns:
        residual_signal (array): señal residual tras la resta.
    """
    if use_mean:
        baseline_value = np.mean(modulated_signal)
        print(f"Usando media del baseline: {baseline_value:.2f}")
        #residual_signal = smoothed_tic - baseline_value
        residual_signal=smoothed_tic-modulated_signal
    else:
        residual_signal = smoothed_tic - modulated_signal

    if clip_negative:
        residual_signal = np.clip(residual_signal, 0, None)

    if plot:
        plt.figure(figsize=(14, 5))
        plt.plot(rts, smoothed_tic, label="Original TIC (gris)", color="gray")
        plt.plot(rts, modulated_signal, label="Modulated Sine (verde)", color="green", linestyle="--")
        plt.plot(rts, residual_signal, label="Residual", color="red")
        plt.xlabel("Retention Time (RT)")
        plt.ylabel("Intensity")
        plt.title("Residual Signal after Subtracting Modulated Baseline")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return residual_signal

def build_and_plot_senoidal_signal(mz_array, intensity_array, rts, tic_original):
    """
    Amplitudes como XIC, SOLO UNA FRECUENCIA CTE

    Parameters
    ----------
    mz_array : TYPE
        DESCRIPTION.
    intensity_array : TYPE
        DESCRIPTION.
    rts : TYPE
        DESCRIPTION.
    tic_original : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    target_mz=922.098
    mz_tol=0.01
    
    smoothed_tic=apply_savgol_filter(tic_original, window_length=15, filter_order=3)
    
    # Convert retention times to numpy array
    rts = np.array(rts)
    sampling_interval = np.mean(np.diff(rts))

    # 1. Construir la señal XIC
    xic_signal = build_xic(mz_array, intensity_array, rts, target_mz, mz_tol)
    #xic son las intensidades para ese mz a lo largo del rt

    # 2. Suavizar señal y calcular frecuencia
    #smoothed_xic = savgol_filter(xic_signal, 15, 3)
    _, _, main_freq = calculate_freq(xic_signal, sampling_interval)
    #print(f"Frecuencia dominante: {main_freq:.4f} Hz")

    # 3. Calcular amplitud local
    #_, amplitudes = extract_intensities_at_mz(mz_array, intensity_array, rts, target_mz)
    amplitudes=xic_signal
    
    #4. Crear el senoide con polinomio
    offset = np.median(smoothed_tic)
    t=rts-rts[0]
    modulated_signal = amplitudes * np.sin(2 * np.pi * main_freq * t) + offset
    #esto es lo mismo que esto
    #for i, rt in enumerate(rts):
        #modulated_signal= amplitudes[i] * np.sin(2*np.pi * main_freq * t[i]) + offset
    
    # Paso 5: plot
    
    plt.figure(figsize=(14, 5))
    plt.plot(rts, smoothed_tic, label="TIC suavizado", color="gray")
    plt.plot(rts, modulated_signal, label="Senoide modulada", color="green", linestyle="--")
    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Intensidad")
    plt.title("TIC vs Señal Senoidal con Frecuencia Variable")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plot_signal_with_features(rts, xic_signal, amplitudes, main_freq)

# Función principal para las amplitudes
def validate_freq_amp_with_tic2(mz_array, intensity_array, tic_array, rt_array, target_mz=922.0098):
    rts = np.array(rt_array)
    tic_array = np.array(tic_array)
    smoothed_tic = apply_savgol_filter(tic_array, window_length=15, filter_order=3)
    sampling_interval = np.mean(np.diff(rts))
    
    # 1. Extraer amplitudes con histogramas
    #rt_array, amplitude_array = extract_intensities_at_mz(mz_array, intensity_array, rt_array, target_mz, bin_size=0.01)
    
    #print("Sample amplitudes:", amplitude_array[:10])
    #print("Max amplitude:", np.max(amplitude_array))
    #print("Min amplitude:", np.min(amplitude_array))
    # 2. Calcular frecuencia
    mz_tol=0.5
    xic = build_xic(mz_array, intensity_array, rts, target_mz, mz_tol)
    _ ,_ ,freq = calculate_freq(xic, sampling_interval)
    print(f"Frecuencia dominante estimada: {freq:.4f} Hz")

    # 3. Construir senoide modulada
    amplitude_array=xic
    # Escala al nuevo rango
    scaled_amplitudes = normalize_to_range(amplitude_array, lower=-1, upper=1)
    
    t = rts - rts[0]
    offset = np.median(smoothed_tic)
    max_amp = np.max(amplitude_array)
    scale_factor=0.3
    
    if max_amp == 0:
        scaled_amplitudes = np.zeros_like(amplitude_array)
    else:
        scaled_amplitudes = amplitude_array.copy()  # no escalar
        #scaled_amplitudes = amplitude_array / max_amp * np.max(smoothed_tic)*scale_factor  # escalar a rango TIC

    #reconstructed_signal = scaled_amplitudes * np.sin(2 * np.pi * freq * t) + offset
    reconstructed_signal = (scaled_amplitudes / 2) * np.sin(2 * np.pi * freq * t) + offset
    # 4. Graficar
    plt.figure(figsize=(12, 5))
    plt.plot(rts, smoothed_tic, label="TIC suavizado", color='gray')
    plt.plot(rt_array, reconstructed_signal, label=f"Senoide modulada (f={freq:.4f} Hz)", color='green', linestyle='--')
    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Intensidad")
    plt.title("TIC vs Señal Senoidal Modulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
