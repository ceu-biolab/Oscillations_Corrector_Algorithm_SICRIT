# processing/processor.py
#!/usr/bin/env python

"""
This Python module implements the main pipeline for detecting and correcting 
oscillatory artifacts in mass spectrometry (MS) data using frequency-domain analysis.

@contents  :  End-to-end correction pipeline for oscillatory m/z signals.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  processor.py
@version   :  0.0.1, 18 July 2025
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    https://www.python.org/dev/peps/pep-0020/
    https://www.python.org/dev/peps/pep-0008/
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@dependencies :
    - pyopenms
    - numpy
    - sicritfix.processing.corrector
    - sicritfix.utils.frequency_analyzer

@functions :
    - detect_oscillating_mzs
    - correct_spectra
    - process_file

@notes :
    This is the central orchestrator of the SICRITfix correction logic.

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""
__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"



import pyopenms as oms
import time
import numpy as np
from collections import defaultdict

from sicritfix.processing.corrector import correct_oscillations
from sicritfix.io.io import load_file
from sicritfix.utils.frequency_analyzer import obtain_freq_from_signal
from sicritfix.utils.intensity_analyzer import build_xic
from sicritfix.validation.validator import plot_original_and_corrected


def detect_oscillating_mzs(rt_array, mz_array, intensity_array, mz_window, rt_window, min_occurrences=10, power_threshold=0.15):
    
    """
    Detects m/z values exhibiting oscillatory behavior based on their XICs using FFT analysis.
    This function scans through all m/z values across spectra, bins them to reduce variability, 
    and selects those that occur frequently enough. For each candidate m/z, it computes the extracted 
    ion chromatogram (XIC), applies FFT to identify periodic patterns, and flags m/z values as oscillating 
    if their normalized power spectrum exceeds a defined threshold.
    
    Parameters
    ----------
    rt_array : list of float
        Retention time values (in seconds) for each scan.
    
    mz_array : list of np.ndarray
        List of m/z arrays for each spectrum.
    
    intensity_array : list of np.ndarray
        List of intensity arrays corresponding to each m/z array per scan.
    
    mz_window : float, optional (default=0.01)
        Size of the bin used to group close m/z values for counting and detection.
    
    min_occurrences : int, optional (default=10)
        Minimum number of scans in which a binned m/z must appear to be considered for analysis.
    
    power_threshold : float, optional (default=0.15)
        Threshold on the normalized FFT power (excluding DC component) above which a signal is considered to exhibit oscillatory behavior.
    
      Returns
      -------
      binned_mzs : list of float
          All binned m/z values observed across the input spectra.
    
      oscillating_mzs : list of float
          Detected m/z values (rounded to 3 decimals) that exhibit oscillatory behavior.
    
      time_detect_oscillating_mzs : float
          Total execution time (in seconds) for the detection process.
      """
    mz_counts=defaultdict(int)
    start_time=time.time()
    
    #1. Binning of all m/z values across all spectra
    for mzs in mz_array:
        binned_mzs=np.round(mzs/mz_window)*mz_window
        for mz in binned_mzs:
            mz_counts[mz]+=1
            
    #2. Selection of the ones that appear in enough spectra
    candidate_mzs=[]
    for mz, count in mz_counts.items():
        if count>=min_occurrences:
            candidate_mzs.append(mz)
     
    #3. Detection of oscillating mzs
    oscillating_mzs=[]
    

        #3.1 Analysis of XIC for each m/z
    for mz in candidate_mzs:
        target_mz=mz
        xic=build_xic(mz_array, intensity_array, rt_array, target_mz, rt_window)
        if(np.sum(xic) < 1e-5):
            continue #this means signal is too weak
        
        #3.1.1 Remove baseline and compute FFT
        xic_centered=xic-np.mean(xic)
        fft_spectrum = np.fft.rfft(xic_centered)
        power_spectrum=np.abs(fft_spectrum) ** 2
        norm_power= power_spectrum /np.sum(power_spectrum)
        
        #3.1.2 Detection of xic with enough intensity power
        if np.any(norm_power[1:] > power_threshold):
           oscillating_mzs.append(round(mz, 3))#round to 3 decimals for simplification
           
    end_time=time.time()  
    time_detect_oscillating_mzs=end_time-start_time
     
    return binned_mzs, oscillating_mzs, time_detect_oscillating_mzs

def correct_spectra(input_map, oscillating_mzs, rts, residual_signals, mz_bin_size=0.001):
    """
    Applies oscillation-corrected intensity values to an MSExperiment.

    For each spectrum in the input MSExperiment, this function replaces the 
    intensities of specified oscillating m/z values with the corresponding 
    residual (corrected) values. Matching is done based on m/z proximity within 
    a specified bin size.

    Parameters
    ----------
    input_map : MSExperiment
        The original mass spectrometry experiment containing raw spectra.

    oscillating_mzs : list of float
        List of m/z values identified as oscillatory and to be corrected.

    rts : list of float
        Retention times corresponding to each spectrum in the input_map.

    residual_signals : dict
        Dictionary mapping each oscillating m/z value to a NumPy array of
        corrected intensity values, indexed by scan (i.e., spectrum position).

    mz_bin_size : float, optional (default=0.001)
        The tolerance used when matching m/z values in the spectrum to the 
        oscillating m/z values.

    Returns
    -------
    corrected_map : MSExperiment
        A new MSExperiment object where the specified m/z values have been 
        corrected with the provided residual intensities.

    time_correct_spectra : float
        Total execution time (in seconds) required to perform the correction.
    """
    corrected_map = oms.MSExperiment()
    #corrected_map.setSpectra(input_map.getSpectra())
    
    start_time=time.time()
    
    for i, spectrum in enumerate(input_map):
        mzs, intensities = spectrum.get_peaks()
        mzs = np.array(mzs)
        intensities = np.array(intensities)

        
        corrected_intensities = intensities.copy()

        for target_mz in oscillating_mzs:
            target_mz=round(float(target_mz), 3)
            corrected_intensity = residual_signals[target_mz][i]

            # Find index of closest m/z (within tolerance)
            mz_diff = np.abs(mzs - target_mz)
            idx_matches = np.where(mz_diff <= mz_bin_size)[0]#indexes within the mz_tol

            for idx in idx_matches:
                corrected_intensities[idx] = corrected_intensity
            

        # Create a new spectrum with corrected peaks
        new_spectrum = oms.MSSpectrum()
        new_spectrum.set_peaks((mzs, corrected_intensities))
        
        #Copy of the original metadata
        new_spectrum.setRT(spectrum.getRT())
        new_spectrum.setMSLevel(spectrum.getMSLevel())
        new_spectrum.setRT(spectrum.getRT())
        new_spectrum.setDriftTime(spectrum.getDriftTime())
        new_spectrum.setPrecursors(spectrum.getPrecursors())
        new_spectrum.setInstrumentSettings(spectrum.getInstrumentSettings())
        new_spectrum.setAcquisitionInfo(spectrum.getAcquisitionInfo())
        new_spectrum.setType(spectrum.getType())
    
        corrected_map.addSpectrum(new_spectrum)
        
    end_time=time.time()
    time_correct_spectra=end_time-start_time
    
    return corrected_map, time_correct_spectra
        

def process_file(file_path, save_as, plot=False, verbose=False, mz_window=0.01, rt_window=0.01):
    """
   Main pipeline for detecting and correcting oscillatory artifacts in an MS data file.

   This function performs the full correction workflow on a mass spectrometry (MS) file.
   It loads the input data (in mzML or mzXML format), detects oscillating m/z values,
   models their oscillatory patterns, corrects their intensities across all spectra,
   and saves the corrected data to a new file.

   Workflow:
   ----------
   1. Load MS data from file (converts from mzXML to mzML if needed).
   2. Extract retention times, m/z, and intensity values.
   3. Compute local frequencies and phase from a reference m/z (922.098).
   4. Detect m/z values showing oscillatory behavior using FFT-based power analysis.
   5. For each oscillating m/z:
      - Estimate signal amplitude.
      - Generate a modulated sinusoidal signal.
      - Subtract it from the original XIC to get a corrected (residual) signal.
   6. Replace the affected m/z intensities in the original spectra with the corrected values.
   7. Save the corrected MSExperiment to disk.

   Parameters
   ----------
   file_path : str
       Path to the input MS data file (must be .mzML or .mzXML).

   save_as : str
       Path where the corrected mzML file will be saved.

   Returns
   -------
   None
       The corrected mzML file is written to disk. Execution times for major steps
       are printed to the console for profiling/debugging purposes.
   """
    
    start_time=time.time()
    input_map=load_file(file_path)

        
    # 1. Load MS data from the original file (rts, mzs, and intesity values)
        
    original_spectra = []
    mz_array = []
    intensity_array=[]
    rts = []#secs
    tic_original = []
        
        
    for spectrum in input_map:
        original_spectra.append(spectrum)
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(mzs)
        intensity_array.append(intensities)
        rt = spectrum.getRT()
        rts.append(rt)
        tic_original.append(np.sum(intensities))
            
    if verbose:
        print(f"Loaded file from {file_path}")
        
    # 2. Oscillations' correction
            
        #2.1 Extract freq from signal of ref: m/z=922.098  
    try:
        local_freqs_ref, phase_ref = obtain_freq_from_signal(rts, mz_array, intensity_array)
    except ValueError:
        print(" Reference signal empty. No oscillations detected")
        oms.MzMLFile().store(save_as, input_map)
        return False
            
        #2.2 Detect mzs to correct
    binned_mzs, oscillating_mzs, time_detect_oscillating_mzs=detect_oscillating_mzs(rts, mz_array, intensity_array, mz_window, rt_window)
    #[DEBUG] PROFILING 
    #print(f" TIME detect_oscillating_mzs: {time_detect_oscillating_mzs}")
            
    if not oscillating_mzs:
        print(" File with no oscillations detected. Returning original file.")
        oms.MzMLFile().store(save_as, input_map)
        print(f" Original file saved as: {save_as}")
        return False
    
    if verbose: 
        print(" Oscillating m/z values found. Correcting...")
           
        #2.3 Call to the correcting function in corrector.py
    xic_signals = {}
    modulated_signals = {}#Dict[target_mz: float, modulated: np.ndarray]
    residual_signals = {}#Dict[target_mz: float, residual: np.ndarray]
            
            
            
    start_time_corrector=time.time()
            
    print("<<< Correcting file. ") 
    for target_mz in oscillating_mzs:
                
        xic, modulated_signal, residual_signal=correct_oscillations(rts, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz, rt_window)
                
        xic_signals[target_mz] = xic
        modulated_signals[target_mz] = modulated_signal
        residual_signals[target_mz] = residual_signal
                
        if plot:
            plot_original_and_corrected(rts, target_mz, xic, residual_signal)
            
        end_time_corrector=time.time()
            
        time_corrector=end_time_corrector-start_time_corrector
        #[DEBUG] PROFILING 
        #print(f" TIME corrector: {time_corrector}")
            
            
            
        # 3. Apply changes (corrections) to spectra
        corrected_map, time_correct_spectra=correct_spectra(input_map, oscillating_mzs, rts, residual_signals)
            
        #[DEBUG] PROFILING 
        #print(f" TIME correct_map: {time_correct_spectra}")
            
        #Computation of overall execution time
        end_time=time.time()
        time_elapsed=end_time-start_time
        
        if verbose:
            print(f" Correction done in {time_elapsed:.3f} seconds")
        
        print("<<< Correction done. ") 
        print(f"Execution time: {time_elapsed:.3f}")
            
            
            
        # 4. Save changes in mzML file
            
        oms.MzMLFile().store(save_as, corrected_map)
        
        if verbose:
            print(f"Corrected file saved: {save_as}")
            
        return True

