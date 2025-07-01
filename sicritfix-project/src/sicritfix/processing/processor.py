# processing/processor.py
import pyopenms as oms
import os
import time
import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
from sicritfix.processing.corrector import correct_oscillations, build_xic
from sicritfix.io.io_utils import convert_mzxml_2_mzml
from sicritfix.utils.fft_utils import local_frequencies_with_fft, apply_polynomial_regression
from sicritfix.validation.validator import plot_all, export_xic_signals_2_csv, plot_original_and_corrected


        
def correct_spectra(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1, threshold=0.0012):
    """
    MZ WITH MODULATED SIGNAL 

    Parameters
    ----------
    input_map : TYPE
        DESCRIPTION.
    target_mz_list : TYPE
        DESCRIPTION.
    rts : TYPE
        DESCRIPTION.
    residual_signals : TYPE
        DESCRIPTION.
    mz_tol : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    corrected_map : TYPE
        DESCRIPTION.

    """
    
    # para mantener el input_map intacto
    corrected_map = oms.MSExperiment()
    
    
    n_corrected = 0
    
    for k in residual_signals.keys():
        mz_keys = np.array(sorted([float(k)]))
    
    # Calcula el umbral relativo
    for v in residual_signals.values():
        all_vals = np.concatenate([np.array(v)])
    max_val = np.max(all_vals)
    intensity_threshold = threshold * max_val
    #print(f"[DEBUG] Threshold = {intensity_threshold:.2e} ({threshold*100:.1f}% of max {max_val:.2e})")

    
    
    # Obtengo los mzs e intensidades del file original
    for i, spectrum in enumerate(input_map):
        mzs, original_intensities = spectrum.get_peaks()
        mzs = np.array(mzs)
        corrected_intensities = np.zeros_like(mzs)
        #le paso el input_map para poder crear el corrected
        #print(f"[DEBUG] Spectrum {i} has {len(mzs)} peaks.")
        #print(f"[DEBUG] First 5 mzs: {mzs[:5]}")
        #print(f"[DEBUG] First 5 original intensities: {original_intensities[:5]}")
        
        for j, mz in enumerate(mzs):
            idx = np.where(np.abs(mz_keys - mz) <= mz_tol)[0]
            if idx.size > 0:
                closest_mz = mz_keys[idx[np.argmin(np.abs(mz_keys[idx] - mz))]]
                intensity = residual_signals[closest_mz][i]

                if intensity >= intensity_threshold:
                    corrected_intensities[j] = intensity
                    n_corrected += 1
                else:
                    corrected_intensities[j] = 0  # Elimina la oscilación base
            else:
                corrected_intensities[j] = original_intensities[j]  # Deja la señal original si no hay corrección posible
        
        # Reconstruye espectro corregido
        new_spectrum = oms.MSSpectrum()
        new_spectrum.set_peaks((mzs, corrected_intensities))
        new_spectrum.setRT(spectrum.getRT())
        new_spectrum.setMSLevel(spectrum.getMSLevel())
        new_spectrum.setDriftTime(spectrum.getDriftTime())
        new_spectrum.setPrecursors(spectrum.getPrecursors())
        new_spectrum.setInstrumentSettings(spectrum.getInstrumentSettings())
        new_spectrum.setAcquisitionInfo(spectrum.getAcquisitionInfo())
        new_spectrum.setType(spectrum.getType())
        #print(f"[DEBUG] Spectrum {i} - First 5 corrected intensities: {corrected_intensities[:5]}")
        corrected_map.addSpectrum(new_spectrum)
        
    print(f"[DEBUG] Total intensities replaced: {n_corrected}")

    return corrected_map

def obtain_freq_from_signal(rt_array, mz_array, intensity_array, window_size=70, mz_ref=922.098):
    """
    Obtains frequency from reference mz

    Parameters
    ----------
    rt_array : TYPE
        DESCRIPTION.
    mz_array : TYPE
        DESCRIPTION.
    intensity_array : TYPE
        DESCRIPTION.
    window_size : TYPE, optional
        DESCRIPTION. The default is 70.
    mz_ref : TYPE, optional
        DESCRIPTION. The default is 922.098.

    Returns
    -------
    rt_freqs : TYPE
        DESCRIPTION.
    local_freqs : TYPE
        DESCRIPTION.
    phase : TYPE
        DESCRIPTION.

    """
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz=mz_ref)
    sampling_interval = np.mean(np.diff(rt_array))
    rt_freqs, local_freqs_ref = local_frequencies_with_fft(xic, rt_array, window_size, sampling_interval)
    phase_ref=apply_polynomial_regression(rt_array, rt_freqs, local_freqs_ref)

    return local_freqs_ref, phase_ref

def detect_oscillating_mzs(rts, mz_array, intensity_array, base_target_mzs, phase_ref, local_freqs_ref, intensity_threshold=100, osc_treshold=0.1, max_mzs_to_check=500):
    detected_mzs=extract_detected_mzs(rts, mz_array, intensity_array)
    auto_selected_mzs=[]
    
    for mz in detected_mzs:
        if mz in base_target_mzs:
            continue
        try:
            xic, modulated_signal, _=correct_oscillations(rts, mz_array, intensity_array, phase_ref, local_freqs_ref, mz)
            intensity_importance=np.var(modulated_signal)/(np.var(xic)+1e-8)
            if intensity_importance > osc_treshold:
                auto_selected_mzs.append(mz)
        except Exception as e:
            print(f"[!] Error analyzing m/z {mz}:{e}")
            
        if len(auto_selected_mzs) >= max_mzs_to_check:
            break #porque si no hay sobrecarga de memoria/CPU 
    
    return auto_selected_mzs

def extract_detected_mzs(rt_array, mz_array, intensity_array, intensity_threshold=1.0, decimal_precision=2):
    """
    Extracts all mzs with signal 

    Parameters
    ----------
    rt_array : TYPE
        array with rts from the original signal
    mz_array : list
        list of arrays of m/z 
    intensity_array : list
        list of arrays of intensities

    Returns
    -------
    List of mzs with an intensity higher than the treshold established

    """
    mz_set = set()

    
    for mzs, intensities in zip(mz_array, intensity_array):#para cada espectro sus mzs y sus intensidades 
        for mz, intensity in zip(mzs, intensities):#en cada espectro mz e intensidad
            if intensity >= intensity_threshold:
                mz_rounded = round(mz, decimal_precision)
                mz_set.add(mz_rounded)

    return sorted(mz_set)

def process_file(file_path, save_as, plot=False, verbose=False):
    start_time = time.time()
    input_map = oms.MSExperiment()

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".mzxml":
        mzml_file_path = convert_mzxml_2_mzml(file_path)
        if verbose:
            print("Converting .mzXML to .mzML...")
        time.sleep(3)
        if not os.path.exists(mzml_file_path):
            raise RuntimeError("Error: file not found")
        else:
            if verbose:
                print("Loading converted mzML file...")
            oms.MzMLFile().load(mzml_file_path, input_map)
    else:
        if verbose:
            print("Loading mzML file...")
        oms.MzMLFile().load(file_path, input_map)

    if verbose:
        print("Extracting spectra and intensities...")

    original_spectra, mz_array, intensity_array, rts, tic_original = [], [], [], [], []

    for spectrum in input_map:
        original_spectra.append(spectrum)
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(mzs)
        intensity_array.append(intensities)
        rts.append(spectrum.getRT())
        tic_original.append(np.sum(intensities))

    if verbose:
        print("Correcting oscillations...")

    # 1. Get frequency/phase of reference signal
    local_freqs_ref, phase_ref = obtain_freq_from_signal(rts, mz_array, intensity_array)

    # 2. Define and detect target m/z values
    target_mz_list = [60.07, 65.1, 95.1, 96.085, 110.1173, 922.098]
    auto_mzs = detect_oscillating_mzs(rts, mz_array, intensity_array, target_mz_list, phase_ref, local_freqs_ref)

    for mz in auto_mzs:
        if mz not in target_mz_list:
            target_mz_list.append(mz)

    cleaned_target_mzs = [float(mz) for mz in target_mz_list]#para asegurar de que todo está en formato correcto


    # 3. Apply correction to each signal
    xic_signals = {}
    modulated_signals = {}
    residual_signals = {}

    for target_mz in cleaned_target_mzs:
        xic, modulated_signal, residual_signal = correct_oscillations(
            rts, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz
        )

        if plot:
            plot_original_and_corrected(rts, xic, residual_signal)

        xic_signals[target_mz] = xic
        modulated_signals[target_mz] = modulated_signal
        residual_signals[target_mz] = residual_signal

    # 4. Apply corrected intensities to spectra
    if verbose:
        print("Applying corrections to spectra...")

    corrected_map = correct_spectra(input_map, cleaned_target_mzs, rts, residual_signals)

    # 5. Save the corrected file
    oms.MzMLFile().store(save_as, corrected_map)

    time_elapsed = time.time() - start_time

    print(" Correction completed.")
    print(f" Execution time: {time_elapsed:.3f} seconds")
    print(f" Corrected file saved to: {save_as}")

