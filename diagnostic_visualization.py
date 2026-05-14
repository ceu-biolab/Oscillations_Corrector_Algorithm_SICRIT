#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic visualization script to analyze oscillation correction performance.

This script analyzes 4 example m/z values and produces detailed visualizations showing:
1. Original signal (XIC)
2. Modulated signals (for all amplitude calculation methods)
3. Output signal (residual after correction)

This helps identify why amplitude corrections might be minimal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "sicritfix-project" / "src"))

from sicritfix.io.io import load_file
from sicritfix.processing.processor import detect_oscillating_mzs
from sicritfix.utils.frequency_analyzer import (
    local_frequencies_with_fft,
    apply_polynomial_regression,
)
from sicritfix.utils.intensity_analyzer import (
    build_xic,
    get_amplitude_by_q75,
    get_amplitude_by_q90,
    get_amplitude_by_local_robust_detrended,
    get_amplitude_by_global_trimmed_detrended,
)
from sicritfix.processing.corrector import correct_oscillations, fit_phase_offset


def analyze_example_mzs(
    file_path,
    mz_window=0.1,
    rt_window=5,
    amplitude_multiplier=1.0,
    n_examples=4,
    example_mzs=None,
    ref_mz=922.098,
    output_dir="diagnostic_plots",
):
    """
    Analyzes and visualizes correction for example m/z values.
    
    Parameters
    ----------
    file_path : str
        Path to input mzML file
    mz_window : float
        M/z window for binning
    rt_window : float
        RT window for XIC smoothing
    amplitude_multiplier : float
        Amplitude multiplier to apply
    n_examples : int
        Number of example m/z to analyze
    output_dir : str
        Directory to save plots
    """
    
    print(f"Loading file: {file_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    exp = load_file(file_path)
    rt_array = np.array([spectrum.getRT() for spectrum in exp])
    
    mz_list = []
    intensity_list = []
    for spectrum in exp:
        mz, intensity = spectrum.get_peaks()
        mz_list.append(np.array(mz))
        intensity_list.append(np.array(intensity))
    
    print(f"Loaded {len(exp)} spectra")
    print(f"RT range: {rt_array.min():.2f} - {rt_array.max():.2f} seconds")
    print(f"Number of scans: {len(rt_array)}")
    print(f"Mean RT interval: {np.mean(np.diff(rt_array)):.4f} seconds")
    
    oscillating_mzs = []
    if example_mzs is None:
        print(f"\nDetecting oscillating m/z values with mz_window={mz_window}, rt_window={rt_window}...")
        binned_mzs, oscillating_mzs, detection_time = detect_oscillating_mzs(
            rt_array=rt_array,
            mz_array=mz_list,
            intensity_array=intensity_list,
            mz_window=mz_window,
            rt_window=rt_window,
            min_occurrences=10,
            power_threshold=0.15,
        )
        
        print(f"Detected {len(oscillating_mzs)} oscillating m/z values in {detection_time:.2f} seconds")
        
        if len(oscillating_mzs) == 0:
            print("No oscillating m/z values detected!")
            return
    
    # Select example m/z values
    if example_mzs is None:
        if len(oscillating_mzs) < n_examples:
            example_mzs = oscillating_mzs
            print(f"Warning: Only {len(oscillating_mzs)} oscillating m/z values detected, analyzing all")
        else:
            indices = np.linspace(0, len(oscillating_mzs) - 1, n_examples, dtype=int)
            example_mzs = [oscillating_mzs[i] for i in indices]
    else:
        example_mzs = [float(mz) for mz in example_mzs]
        print("\nSkipping global oscillating-m/z detection because example_mzs were provided explicitly.")
    
    print(f"\nSelected example m/z values: {example_mzs}")
    
    # Obtain frequency and phase from the same fixed reference mass used in production.
    print(f"\nObtaining reference frequency and phase from m/z {ref_mz}...")
    sampling_interval = np.mean(np.diff(rt_array))

    ref_xic = build_xic(
        mz_array=mz_list,
        intensity_array=intensity_list,
        rt_array=rt_array,
        target_mz=ref_mz,
        rt_window=rt_window,
        mz_window=0.1,
    )
    rt_freqs, local_freqs_ref = local_frequencies_with_fft(
        ref_xic,
        rt_array,
        window_scan_size=70,
        sampling_interval=sampling_interval,
    )
    phase_ref = apply_polynomial_regression(rt_array, rt_freqs, local_freqs_ref)
    
    print(f"Reference frequency range: {local_freqs_ref.min():.4f} - {local_freqs_ref.max():.4f} Hz")
    print(f"Mean reference frequency: {local_freqs_ref.mean():.4f} Hz")
    
    def calc_centered_reduction_pct(original, residual, reducer):
        centered_original = original - np.mean(original)
        centered_residual = residual - np.mean(residual)
        baseline = reducer(centered_original)
        if baseline <= 0:
            return 0.0
        return 100.0 * (baseline - reducer(centered_residual)) / baseline

    # Analyze each example m/z
    results = []
    for mz in example_mzs:
        print(f"\n{'='*70}")
        print(f"Analyzing m/z: {mz}")
        print(f"{'='*70}")
        
        # Extract XIC
        xic = build_xic(
            mz_array=mz_list,
            intensity_array=intensity_list,
            rt_array=rt_array,
            target_mz=mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )
        
        print(f"XIC range: {xic.min():.2e} - {xic.max():.2e}")
        print(f"XIC mean: {xic.mean():.2e}")
        print(f"XIC std: {xic.std():.2e}")
        
        # Calculate amplitudes using all methods
        amp_q75 = get_amplitude_by_q75(xic, local_freqs_ref, sampling_interval)
        amp_q90 = get_amplitude_by_q90(xic, local_freqs_ref, sampling_interval)
        amp_local_robust = get_amplitude_by_local_robust_detrended(
            xic, local_freqs_ref, sampling_interval
        )
        amp_global_trimmed = get_amplitude_by_global_trimmed_detrended(
            xic, local_freqs_ref, sampling_interval
        )
        
        print(f"\nAmplitude estimates:")
        print(f"  Q75:                   {amp_q75:.4e}")
        print(f"  Q90:                   {amp_q90:.4e}")
        print(f"  Local Robust Detrended: {amp_local_robust:.4e}")
        print(f"  Global Trimmed Detrended: {amp_global_trimmed:.4e}")
        
        # Apply multiplier
        amp_q75_mult = amp_q75 * amplitude_multiplier
        amp_q90_mult = amp_q90 * amplitude_multiplier
        amp_local_robust_mult = amp_local_robust * amplitude_multiplier
        amp_global_trimmed_mult = amp_global_trimmed * amplitude_multiplier
        
        print(f"\nAmplitude estimates (with multiplier={amplitude_multiplier}):")
        print(f"  Q75:                   {amp_q75_mult:.4e}")
        print(f"  Q90:                   {amp_q90_mult:.4e}")
        print(f"  Local Robust Detrended: {amp_local_robust_mult:.4e}")
        print(f"  Global Trimmed Detrended: {amp_global_trimmed_mult:.4e}")
        
        # Generate modulated and residual signals using the current pipeline,
        # including the per-m/z phase-offset fitting.
        _, mod_q75, res_q75 = correct_oscillations(
            rt_array,
            mz_list,
            intensity_list,
            phase_ref,
            local_freqs_ref,
            target_mz=mz,
            rt_window=rt_window,
            mz_tol=mz_window,
            amplitude_method="q75",
            amplitude_percentile=75,
            amplitude_multiplier=amplitude_multiplier,
        )
        _, mod_q90, res_q90 = correct_oscillations(
            rt_array,
            mz_list,
            intensity_list,
            phase_ref,
            local_freqs_ref,
            target_mz=mz,
            rt_window=rt_window,
            mz_tol=mz_window,
            amplitude_method="q90",
            amplitude_percentile=75,
            amplitude_multiplier=amplitude_multiplier,
        )
        _, mod_local_robust, res_local_robust = correct_oscillations(
            rt_array,
            mz_list,
            intensity_list,
            phase_ref,
            local_freqs_ref,
            target_mz=mz,
            rt_window=rt_window,
            mz_tol=mz_window,
            amplitude_method="local_robust_detrended",
            amplitude_percentile=75,
            amplitude_multiplier=amplitude_multiplier,
        )
        _, mod_global_trimmed, res_global_trimmed = correct_oscillations(
            rt_array,
            mz_list,
            intensity_list,
            phase_ref,
            local_freqs_ref,
            target_mz=mz,
            rt_window=rt_window,
            mz_tol=mz_window,
            amplitude_method="global_trimmed_detrended",
            amplitude_percentile=75,
            amplitude_multiplier=amplitude_multiplier,
        )

        phase_offset = fit_phase_offset(xic, phase_ref, local_freqs_ref, sampling_interval)

        print(f"\nCorrected signal ranges:")
        print(f"  Q75 residual:           {res_q75.min():.2e} - {res_q75.max():.2e}")
        print(f"  Q90 residual:           {res_q90.min():.2e} - {res_q90.max():.2e}")
        print(f"  Local Robust residual:  {res_local_robust.min():.2e} - {res_local_robust.max():.2e}")
        print(f"  Global Trimmed residual: {res_global_trimmed.min():.2e} - {res_global_trimmed.max():.2e}")
        print(f"  Fitted phase offset:     {phase_offset:.4f} rad")
        
        # Calculate correction percentages
        def calc_correction_pct(original, residual):
            if np.ptp(original) > 0:
                return 100 * (np.ptp(original) - np.ptp(residual)) / np.ptp(original)
            return 0
        
        print(f"\nCorrection as % of peak-to-peak:")
        print(f"  Q75:                   {calc_correction_pct(xic, res_q75):.2f}%")
        print(f"  Q90:                   {calc_correction_pct(xic, res_q90):.2f}%")
        print(f"  Local Robust Detrended: {calc_correction_pct(xic, res_local_robust):.2f}%")
        print(f"  Global Trimmed Detrended: {calc_correction_pct(xic, res_global_trimmed):.2f}%")
        print(f"\nCentered correction metrics:")
        print(f"  Q75 std reduction:      {calc_centered_reduction_pct(xic, res_q75, np.std):.2f}%")
        print(f"  Q90 std reduction:      {calc_centered_reduction_pct(xic, res_q90, np.std):.2f}%")
        print(f"  Local Robust std red.:  {calc_centered_reduction_pct(xic, res_local_robust, np.std):.2f}%")
        print(f"  Global Trimmed std red.: {calc_centered_reduction_pct(xic, res_global_trimmed, np.std):.2f}%")
        
        results.append({
            'mz': mz,
            'xic': xic,
            'mod_q75': mod_q75,
            'mod_q90': mod_q90,
            'mod_local_robust': mod_local_robust,
            'mod_global_trimmed': mod_global_trimmed,
            'res_q75': res_q75,
            'res_q90': res_q90,
            'res_local_robust': res_local_robust,
            'res_global_trimmed': res_global_trimmed,
            'amp_q75': amp_q75,
            'amp_q90': amp_q90,
            'amp_local_robust': amp_local_robust,
            'amp_global_trimmed': amp_global_trimmed,
            'phase_offset': phase_offset,
            'q75_ptp_reduction_pct': calc_correction_pct(xic, res_q75),
            'q90_ptp_reduction_pct': calc_correction_pct(xic, res_q90),
            'local_robust_ptp_reduction_pct': calc_correction_pct(xic, res_local_robust),
            'global_trimmed_ptp_reduction_pct': calc_correction_pct(xic, res_global_trimmed),
            'q75_std_reduction_pct': calc_centered_reduction_pct(xic, res_q75, np.std),
            'q90_std_reduction_pct': calc_centered_reduction_pct(xic, res_q90, np.std),
            'local_robust_std_reduction_pct': calc_centered_reduction_pct(xic, res_local_robust, np.std),
            'global_trimmed_std_reduction_pct': calc_centered_reduction_pct(xic, res_global_trimmed, np.std),
        })
    
    # Create comprehensive visualization
    print(f"\n{'='*70}")
    print("Creating comprehensive visualization...")
    print(f"{'='*70}")
    
    create_visualization(results, rt_array, output_path)
    
    # Print diagnostic information
    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS SUMMARY")
    print("="*70)
    print(f"File: {file_path}")
    print(f"MZ Window: {mz_window} Da")
    print(f"RT Window: {rt_window} seconds")
    print(f"Amplitude Multiplier: {amplitude_multiplier}")
    print(f"Sampling Interval: {sampling_interval:.6f} seconds")
    print(f"Signal Duration: {rt_array[-1] - rt_array[0]:.2f} seconds")
    print(f"Total Points: {len(rt_array)}")
    print(f"Reference Frequency: {local_freqs_ref.mean():.4f} Hz")
    print(f"Expected Period: {1/local_freqs_ref.mean():.4f} seconds")
    print(f"Expected Period in points: {1/(local_freqs_ref.mean() * sampling_interval):.1f}")

    summary_path = output_path / "diagnostic_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "file": str(file_path),
                "mz_window": mz_window,
                "rt_window": rt_window,
                "amplitude_multiplier": amplitude_multiplier,
                "ref_mz": ref_mz,
                "sampling_interval": float(sampling_interval),
                "example_mzs": example_mzs,
                "reference_frequency_mean_hz": float(local_freqs_ref.mean()),
                "results": [
                    {
                        "mz": float(result["mz"]),
                        "amp_q75": float(result["amp_q75"]),
                        "amp_q90": float(result["amp_q90"]),
                        "amp_local_robust": float(result["amp_local_robust"]),
                        "amp_global_trimmed": float(result["amp_global_trimmed"]),
                        "phase_offset_rad": float(result["phase_offset"]),
                        "q75_ptp_reduction_pct": float(result["q75_ptp_reduction_pct"]),
                        "q90_ptp_reduction_pct": float(result["q90_ptp_reduction_pct"]),
                        "local_robust_ptp_reduction_pct": float(result["local_robust_ptp_reduction_pct"]),
                        "global_trimmed_ptp_reduction_pct": float(result["global_trimmed_ptp_reduction_pct"]),
                        "q75_std_reduction_pct": float(result["q75_std_reduction_pct"]),
                        "q90_std_reduction_pct": float(result["q90_std_reduction_pct"]),
                        "local_robust_std_reduction_pct": float(result["local_robust_std_reduction_pct"]),
                        "global_trimmed_std_reduction_pct": float(result["global_trimmed_std_reduction_pct"]),
                    }
                    for result in results
                ],
            },
            handle,
            indent=2,
        )
    print(f"Saved summary to: {summary_path}")
    
    # Hypothesis analysis
    print("\n" + "="*70)
    print("POTENTIAL ISSUES ANALYSIS")
    print("="*70)
    
    avg_amp = np.mean([r['amp_local_robust'] for r in results])
    print(f"\nAverage amplitude (local_robust_detrended): {avg_amp:.4e}")
    
    if avg_amp < 1e3:
        print("  ⚠ ISSUE: Amplitude is very small compared to signal scale")
        print("  Possible causes:")
        print("    1. XIC baseline is too high relative to oscillation amplitude")
        print("    2. MZ window is capturing too much background noise")
        print("    3. RT window smoothing is suppressing the oscillations")
    
    if np.mean(np.diff(rt_array)) * 70 > rt_array[-1] - rt_array[0]:
        print("\n  ⚠ ISSUE: Window size for frequency analysis is large compared to signal duration")
        print(f"    Window size: ~{np.mean(np.diff(rt_array)) * 70:.2f} seconds")
        print(f"    Signal duration: {rt_array[-1] - rt_array[0]:.2f} seconds")
    
    # Print successful completion
    print(f"\n✓ Plots saved to: {output_path}")
    print(f"✓ Analysis complete!")


def create_visualization(results, rt_array, output_path):
    """Create comprehensive visualization of correction results."""
    
    n_mz = len(results)
    fig = plt.figure(figsize=(20, 5 * n_mz))
    
    for idx, result in enumerate(results):
        mz = result['mz']
        xic = result['xic']
        rt_array_plot = rt_array
        
        # Create grid for this m/z
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=gridspec.GridSpec(n_mz, 1)[idx]
        )
        
        # Plot 1: Original XIC
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(rt_array_plot, xic, 'b-', linewidth=2, label='Original XIC')
        ax1.set_title(f'm/z = {mz}\nOriginal Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2-5: Modulated signals (all methods)
        methods = [
            ('Q75', result['mod_q75']),
            ('Q90', result['mod_q90']),
            ('Local Robust', result['mod_local_robust']),
            ('Global Trimmed', result['mod_global_trimmed']),
        ]
        
        for col, (method_name, mod_signal) in enumerate(methods):
            ax = fig.add_subplot(gs[0, col])
            ax.plot(rt_array_plot, mod_signal, 'r-', linewidth=2, label=f'Modulated ({method_name})')
            ax.set_title(f'Signal to Subtract\n({method_name})', fontsize=10, fontweight='bold')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Plot 6-9: Residual signals (all methods)
        residuals = [
            ('Q75', result['res_q75']),
            ('Q90', result['res_q90']),
            ('Local Robust', result['res_local_robust']),
            ('Global Trimmed', result['res_global_trimmed']),
        ]
        
        for col, (method_name, res_signal) in enumerate(residuals):
            ax = fig.add_subplot(gs[1, col])
            ax.plot(rt_array_plot, res_signal, 'g-', linewidth=2, label=f'Corrected ({method_name})')
            ax.set_title(f'Output Signal (Residual)\n({method_name})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Retention Time (s)')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Add annotation with amplitude information
        amp_text = (
            f"Amplitudes (with multiplier):\n"
            f"Q75: {result['amp_q75']:.2e}\n"
            f"Q90: {result['amp_q90']:.2e}\n"
            f"Local Robust: {result['amp_local_robust']:.2e}\n"
            f"Global Trimmed: {result['amp_global_trimmed']:.2e}\n"
            f"Phase offset: {result['phase_offset']:.3f} rad"
        )
        fig.text(
            0.02, 0.95 - idx * (1 / (n_mz + 1)), amp_text,
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    output_file = output_path / "comprehensive_correction_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Create individual plots for each m/z
    for result in results:
        mz = result['mz']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'm/z = {mz} - Detailed Correction Analysis', fontsize=14, fontweight='bold')
        
        # Original vs all corrections
        ax = axes[0, 0]
        ax.plot(rt_array, result['xic'], 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(rt_array, result['res_q75'], 'r--', linewidth=1.5, label='Q75', alpha=0.7)
        ax.plot(rt_array, result['res_q90'], 'g--', linewidth=1.5, label='Q90', alpha=0.7)
        ax.plot(rt_array, result['res_local_robust'], 'orange', linestyle='--', linewidth=1.5, label='Local Robust', alpha=0.7)
        ax.plot(rt_array, result['res_global_trimmed'], 'purple', linestyle='--', linewidth=1.5, label='Global Trimmed', alpha=0.7)
        ax.set_title('Original vs All Corrections', fontweight='bold')
        ax.set_xlabel('Retention Time (s)')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # All modulated signals
        ax = axes[0, 1]
        ax.plot(rt_array, result['mod_q75'], 'r-', linewidth=1.5, label='Q75', alpha=0.7)
        ax.plot(rt_array, result['mod_q90'], 'g-', linewidth=1.5, label='Q90', alpha=0.7)
        ax.plot(rt_array, result['mod_local_robust'], 'orange', linewidth=1.5, label='Local Robust', alpha=0.7)
        ax.plot(rt_array, result['mod_global_trimmed'], 'purple', linewidth=1.5, label='Global Trimmed', alpha=0.7)
        ax.set_title('All Modulated Signals', fontweight='bold')
        ax.set_xlabel('Retention Time (s)')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Amplitude comparison
        ax = axes[1, 0]
        methods = ['Q75', 'Q90', 'Local\nRobust', 'Global\nTrimmed']
        amplitudes = [
            result['amp_q75'], result['amp_q90'],
            result['amp_local_robust'], result['amp_global_trimmed']
        ]
        bars = ax.bar(methods, amplitudes, color=['red', 'green', 'orange', 'purple'], alpha=0.7)
        ax.set_title('Amplitude Estimates', fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, amp in zip(bars, amplitudes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{amp:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Correction percentage
        ax = axes[1, 1]
        corrections = [
            result['q75_ptp_reduction_pct'],
            result['q90_ptp_reduction_pct'],
            result['local_robust_ptp_reduction_pct'],
            result['global_trimmed_ptp_reduction_pct'],
        ]
        bars = ax.bar(methods, corrections, color=['red', 'green', 'orange', 'purple'], alpha=0.7)
        ax.set_title('Correction as % of Peak-to-Peak', fontweight='bold')
        ax.set_ylabel('Correction %')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, corr in zip(bars, corrections):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{corr:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_file = output_path / f"mz_{mz:.3f}_detailed_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


if __name__ == "__main__":
    # Find demo data
    demo_data_dir = Path(__file__).parent / "demo_data"
    
    # Look for mzML files
    mzml_files = list(demo_data_dir.glob("*.mzML"))
    if not mzml_files:
        print(f"No mzML files found in {demo_data_dir}")
        print("Available files:")
        for f in demo_data_dir.iterdir():
            print(f"  {f.name}")
        sys.exit(1)
    
    # Use first file
    file_path = mzml_files[0]
    print(f"Using file: {file_path}")
    
    # Run analysis
    analyze_example_mzs(
        file_path=str(file_path),
        mz_window=0.1,
        rt_window=5,
        amplitude_multiplier=1.0,
        n_examples=4,
        output_dir="diagnostic_plots",
    )
