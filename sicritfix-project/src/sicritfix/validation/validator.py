# -*- coding: utf-8 -*-
#validation/validator.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def export_xic_signals_2_csv(rts, xic_signals, modulated_signals, residual_signals, output_csv_path):
    """
    Export XIC, modulated, and residual signals for multiple m/z values to a single CSV file.

    This function creates a unified table where each row corresponds to a retention time (RT)
    and each group of columns represents the original XIC, modulated signal, and residual 
    signal for a specific m/z value. The resulting CSV is formatted with a semicolon (';') 
    as the column separator and uses a Spanish-style decimal format.

    Parameters
    ----------
    rts : list or np.ndarray
        Array of retention time values corresponding to the time dimension of the signals.

    xic_signals : dict
        Dictionary mapping each target m/z (float) to its original extracted ion chromatogram (XIC),
        as a NumPy array aligned with `rts`.

    modulated_signals : dict
        Dictionary mapping each target m/z to its corresponding modulated sinusoidal signal.

    residual_signals : dict
        Dictionary mapping each target m/z to its corrected (residual) signal after oscillation removal.

    output_csv_path : str
        Path where the resulting CSV file will be saved.

    Returns
    -------
    None
        A CSV file is saved to disk. Each m/z has its own triplet of columns:
        - `XIC_<m/z>`
        - `Modulated_<m/z>`
        - `Residual_<m/z>`
    """
    df = pd.DataFrame({'RT': rts})

    for target_mz in xic_signals:
        mz_str = f"{target_mz:.4f}"
        df[f"XIC_{mz_str}"] = xic_signals[target_mz]
        df[f"Modulated_{mz_str}"] = modulated_signals[target_mz]
        df[f"Residual_{mz_str}"] = residual_signals[target_mz]

    df_formatted = df.map(
        lambda x: f"{x:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
        if isinstance(x, (float, int)) else x
    )

    # Saves with ';' separator
    df_formatted.to_csv(output_csv_path, index=False, sep=';')
    print(f"Exportado CSV combinado a: {output_csv_path}")
    
def plot_ms_experiment_3d(ms_experiment):
    
    """
    Plot a 3D visualization of an MSExperiment object with retention time, m/z, and intensity.

    This function creates a 3D scatter plot from a mass spectrometry experiment,
    where:
      - The x-axis represents retention time (RT),
      - The y-axis represents m/z values,
      - The z-axis and color represent signal intensity.

    It is useful for visual inspection of overall signal structure and 
    identifying patterns, trends, or anomalies across scans and m/z values.

    Parameters
    ----------
    ms_experiment : MSExperiment
        A mass spectrometry experiment containing a list of spectra.
        Each spectrum must provide m/z and intensity pairs as well as a retention time.

    Returns
    -------
    None
        Displays an interactive 3D scatter plot with color mapped to intensity.
    """
    
    rts = []
    mzs = []
    intensities = []

    for spec in ms_experiment:
        rt = spec.getRT()
        mz, inten = spec.get_peaks()

        if len(mz) == 0:
            continue  

        rts.extend([rt] * len(mz))
        mzs.extend(mz)
        intensities.extend(inten)

    rts = np.array(rts)
    mzs = np.array(mzs)
    intensities = np.array(intensities)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Crear el scatter plot con mapeo de color
    sc = ax.scatter(rts, mzs, intensities, c=intensities, cmap='viridis', marker='o', s=5, alpha=0.8)

    # Añadir colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Intensity")

    ax.set_xlabel("Retention Time (s)")
    ax.set_ylabel("m/z")
    ax.set_zlabel("Intensity")
    ax.set_title("3D MS Corrected Map (Color = Intensity)")

    plt.tight_layout()
    plt.show()

def plot_xic_from_map(ms_map, target_mz, mz_tol=0.01):
    """
    Plot the Extracted Ion Chromatogram (XIC) for a specific m/z value from an MSExperiment.

    This function scans through MS1 spectra in the given MS experiment and extracts 
    intensity values for the specified `target_mz`, using a defined tolerance. It then 
    plots the intensity as a function of retention time (RT), resulting in an XIC.

    Parameters
    ----------
    ms_map : MSExperiment
       The mass spectrometry experiment object containing spectra to be scanned.

    target_mz : float
       The m/z value for which the XIC will be extracted and plotted.

    mz_tol : float, optional (default=0.01)
       The tolerance within which m/z values are considered a match for `target_mz`.

    Returns
    -------
    None
       Displays a 2D plot of intensity vs. retention time for the selected m/z value.
   """
    rts = []
    intensities = []
    for spec in ms_map:
        if spec.getMSLevel() != 1:
            continue
        mzs, intens = spec.get_peaks()
        rt = spec.getRT()
        for mz, intensity in zip(mzs, intens):
            if abs(mz - target_mz) <= mz_tol:
                rts.append(rt)
                intensities.append(intensity)
                break
    plt.plot(rts, intensities)
    plt.xlabel("Retention Time (s)")
    plt.ylabel(f"Intensity at {target_mz} m/z")
    plt.title(f"XIC of {target_mz}")
    plt.show()

def plot_all(rts, target_mz, xic, modulated_signal, residual_signal):
    """
    Plot the original XIC, the modulated signal, and the residual signal for a specific m/z.

    This function creates a two-panel plot:
      - The first subplot displays the original XIC and the modulated signal for the given m/z.
      - The second subplot shows the residual signal, obtained by subtracting the modulated signal from the XIC.
    
    The y-axis limits are shared across both plots for visual consistency.

    Parameters
    ----------
    rts : array-like
        Retention time values corresponding to the signal arrays.

    target_mz : float
        The m/z value for which the signals are being visualized.

    xic : array-like
        Original extracted ion chromatogram for the target m/z.

    modulated_signal : array-like
        The modeled sinusoidal oscillation to be subtracted from the XIC.

    residual_signal : array-like
        The result of subtracting the modulated signal from the XIC.

    Returns
    -------
    None
        Displays the plots. Does not return any values.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # First subplot: original and modulated signal
    axs[0].plot(rts, xic, label='XIC original', linewidth=0.8, color='grey')
    axs[0].plot(rts, modulated_signal, label='Modulated signal', linestyle='--', color='orange')
    axs[0].set_ylabel("Intensity")
    axs[0].set_title(f"XIC and Modulated Signal for m/z = {target_mz}")
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: residual signal
    axs[1].plot(rts, residual_signal, label='Residual signal', color='green', linewidth=0.8)
    axs[1].set_xlabel("Retention time (s)")
    axs[1].set_ylabel("Intensity")
    axs[1].set_title("Residual Signal")
    axs[1].legend()
    axs[1].grid(True)
    
    #Same scale
    all_values = np.concatenate([xic, modulated_signal, residual_signal])
    y_min, y_max = np.min(all_values), np.max(all_values)

    
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
    
def plot_modulated_signal(rts, target_mz, modulated_signal):
    """
    Plot the modulated signal over retention time for a specific m/z.

    This function displays the sinusoidal modulation signal that was estimated
    for a given m/z, typically used to model and subtract oscillatory artifacts
    from the original signal.

    Parameters
    ----------
    rts : array-like
        Retention time values (in seconds) corresponding to the signal.

    target_mz : float
        The m/z value for which the modulated signal is being visualized.

    modulated_signal : array-like
        The generated oscillatory (modulated) signal for the target m/z.

    Returns
    -------
    None
        Displays a line plot of the modulated signal.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(rts, modulated_signal, label='Modulated signal', linestyle='--', color='orange')
    plt.xlabel("Retention time (s)")
    plt.ylabel("Intesity")
    plt.title(f"Modulated signal for m/z = {target_mz}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residual_signal(rt_array, target_mz, residual_signal):
    """
    Plot the residual signal (after modulation removal) versus retention time.

    This function visualizes how the residual signal behaves over time for a specific m/z,
    helping to assess the effectiveness of the modulation correction.

    Parameters
    ----------
    rt_array : array-like
        Retention time values (in seconds).

    target_mz : float
        Target m/z value (included in the plot title for reference).

    residual_signal : array-like
        Signal after subtracting the modulated (oscillatory) component.

    Returns
    -------
    None
        Displays the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rt_array[:40], residual_signal[:40], label='Residual signal', color='blue', linewidth=0.9)

    plt.xlabel("Retention time (s)")
    plt.ylabel("Residual intensity")
    title = "Residual Signal"
    if target_mz is not None:
        title += f" for m/z = {target_mz}"
    plt.title(title)
    
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_original_and_modulated(rts, target_mz, xic, modulated_signal):
    """
    Plot the original XIC signal alongside the modulated signal over retention time.

    This visualization helps compare the raw signal with the modeled oscillation,
    and is useful to verify how well the modulation fits the original data.

    Parameters
    ----------
    rts : array-like
        Retention time values (in seconds).

    target_mz : float
        Target m/z value (included in the plot title for context).

    xic : array-like
        Original extracted ion chromatogram (XIC) signal.

    modulated_signal : array-like
        Generated oscillatory signal to be subtracted from the XIC.

    Returns
    -------
    None
        Displays the plot.
    """
    
    
    plt.figure(8,4)
    plt.plot(rts, xic, label='Original XIC signal', color='black', linewidth=0.8)
    plt.plot(rts, modulated_signal, labbel='Modulated signal', color='blue', linewidth=0.8)
    plt.xlabel("Retention time (s)")
    plt.ylabel("Intesity")
    plt.title(f"XIC original signal vs Modulated signal for m/z = {target_mz}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_original_and_corrected(rts, target_mz, xic, residual_signal):
    """
    Plot the original XIC signal and the corrected (residual) signal over retention time for a specific m/z.

    This function helps visualize the effect of oscillation correction by comparing the raw extracted
    ion chromatogram (XIC) to its corrected counterpart (residual signal).

    Parameters
    ----------
    rts : array-like
        Retention time values (in seconds).

    target_mz : float
        The m/z value for which the comparison is plotted.

    xic : array-like
        Original extracted ion chromatogram for the target m/z.

    residual_signal : array-like
        Corrected (residual) signal after subtracting the oscillatory component.

    Returns
    -------
    None
        Displays the plot showing original vs. corrected signal.
    """
    plt.figure(8,4)
    plt.plot(rts, xic, label='Original XIC signal', color='black', linewidth=0.8)
    plt.plot(rts, residual_signal, label='Corrected signal', color='blue', linewidth=0.8)
    plt.xlabel("Retention time (s)")
    plt.ylabel("Intesity")
    plt.title(f"XIC original signal vs Corrected signal for m/z = {target_mz}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    