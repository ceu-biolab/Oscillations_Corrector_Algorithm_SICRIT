#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Targeted diagnostic plots for selected m/z values.

For each target mass this script saves:
1. A combined signal plot with original, subtract, and corrected traces for each amplitude method.
2. A bar plot with the absolute amplitude estimated by each method.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent / "sicritfix-project" / "src"))

from sicritfix.io.io import load_file
from sicritfix.processing.corrector import correct_oscillations
from sicritfix.utils.frequency_analyzer import (
    apply_polynomial_regression,
    local_frequencies_with_fft,
)
from sicritfix.utils.intensity_analyzer import build_xic, get_amplitude


TARGET_MASSES = {
    "L-Tryptophan": 205.09771,
    "L-Histidine": 156.07731,
    "Gly-Leu": 189.12392,
    "Methionine sulfone": 182.04871,
    "Another 121.050873": 121.050873,
    "Another 922.009798": 922.009798,
}

PHASE_SOURCE_MZ = 922.009798

METHODS = [
    "q75",
    "q90",
    "global_trimmed_detrended",
    "local_robust_detrended",
]

METHOD_COLORS = {
    "q75": "#b22222",
    "q90": "#0f766e",
    "global_trimmed_detrended": "#7c3aed",
    "local_robust_detrended": "#ea580c",
}


def load_arrays(file_path: str):
    exp = load_file(file_path)
    rt_array = np.array([spectrum.getRT() for spectrum in exp], dtype=float)
    mz_array = []
    intensity_array = []
    for spectrum in exp:
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(np.asarray(mzs, dtype=float))
        intensity_array.append(np.asarray(intensities, dtype=float))
    return exp, rt_array, mz_array, intensity_array


def get_phase_reference(
    rt_array,
    mz_array,
    intensity_array,
    phase_source_mz,
    rt_window,
    mz_window,
    window_scan_size,
):
    sampling_interval = float(np.mean(np.diff(rt_array)))
    ref_xic = build_xic(
        mz_array,
        intensity_array,
        rt_array,
        target_mz=phase_source_mz,
        rt_window=rt_window,
        mz_window=mz_window,
    )
    rt_freqs, local_freqs_ref = local_frequencies_with_fft(
        ref_xic,
        rt_array,
        window_scan_size=window_scan_size,
        sampling_interval=sampling_interval,
    )
    phase_ref = apply_polynomial_regression(rt_array, rt_freqs, local_freqs_ref)
    return local_freqs_ref, phase_ref, sampling_interval


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def run_targeted_analysis(
    file_path: str,
    output_dir: str,
    mz_window: float = 0.1,
    rt_window: float = 5.0,
    amplitude_multiplier: float = 1.0,
    window_scan_size: int = 70,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, rt_array, mz_array, intensity_array = load_arrays(file_path)
    local_freqs_ref, phase_ref, sampling_interval = get_phase_reference(
        rt_array,
        mz_array,
        intensity_array,
        phase_source_mz=PHASE_SOURCE_MZ,
        rt_window=rt_window,
        mz_window=mz_window,
        window_scan_size=window_scan_size,
    )
    rt_minutes = rt_array / 60.0
    major_tick_minutes = 100.0 / 60.0

    summary = {
        "file": file_path,
        "phase_source_mz": PHASE_SOURCE_MZ,
        "window_scan_size": window_scan_size,
        "mz_window": mz_window,
        "rt_window": rt_window,
        "amplitude_multiplier": amplitude_multiplier,
        "targets": {},
    }

    for target_name, target_mz in TARGET_MASSES.items():
        original_xic = build_xic(
            mz_array,
            intensity_array,
            rt_array,
            target_mz=target_mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )

        method_results = {}
        amplitudes = []
        labels = []

        for method in METHODS:
            xic, modulated_signal, residual_signal = correct_oscillations(
                rt_array,
                mz_array,
                intensity_array,
                phase_ref,
                local_freqs_ref,
                target_mz=target_mz,
                rt_window=rt_window,
                mz_tol=mz_window,
                amplitude_method=method,
                amplitude_multiplier=amplitude_multiplier,
            )
            amplitude = float(
                get_amplitude(
                    xic,
                    local_freqs_ref,
                    sampling_interval,
                    method=method,
                )
                * amplitude_multiplier
            )
            method_results[method] = {
                "amplitude": amplitude,
                "modulated_signal": modulated_signal,
                "residual_signal": residual_signal,
                "ptp_reduction_pct": float(
                    100.0 * (np.ptp(original_xic) - np.ptp(residual_signal)) / max(1e-12, np.ptp(original_xic))
                ),
                "std_reduction_pct": float(
                    100.0
                    * (
                        np.std(original_xic - np.mean(original_xic))
                        - np.std(residual_signal - np.mean(residual_signal))
                    )
                    / max(1e-12, np.std(original_xic - np.mean(original_xic)))
                ),
            }
            amplitudes.append(amplitude)
            labels.append(method)

        fig1, axes1 = plt.subplots(len(METHODS), 1, figsize=(16, 12), sharex=True)
        fig1.suptitle(
            f"{target_name} ({target_mz:.5f}) - Original, Subtract, and Corrected Signals",
            fontsize=14,
        )
        for ax, method in zip(axes1, METHODS):
            result = method_results[method]
            ax.plot(rt_minutes, original_xic, color="black", linewidth=1.0, label="Original signal")
            ax.plot(
                rt_minutes,
                result["modulated_signal"],
                color=METHOD_COLORS[method],
                linewidth=1.0,
                linestyle="--",
                label=f"Signal to subtract ({method})",
            )
            ax.plot(
                rt_minutes,
                result["residual_signal"],
                color=METHOD_COLORS[method],
                linewidth=1.0,
                alpha=0.8,
                label=f"Corrected signal ({method})",
            )
            ax.set_ylabel("Intensity")
            ax.set_title(
                f"{method} | amplitude={result['amplitude']:.3e} | "
                f"PTP reduction={result['ptp_reduction_pct']:.2f}% | "
                f"STD reduction={result['std_reduction_pct']:.2f}%"
            )
            ax.grid(True, alpha=0.2)
            ax.xaxis.set_major_locator(MultipleLocator(major_tick_minutes))
            ax.legend(loc="upper right", fontsize=8)
        axes1[-1].set_xlabel("Retention time (min; major ticks every 100 s)")
        fig1.tight_layout()
        fig1.subplots_adjust(top=0.96)
        combined_plot_path = outdir / f"{sanitize_name(target_name)}_01_combined_signals.png"
        fig1.savefig(combined_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved combined signals plot to: {combined_plot_path}")

        fig3, ax3 = plt.subplots(figsize=(11, 6))
        x = np.arange(len(labels), dtype=float)
        bars = ax3.bar(x, amplitudes, color=[METHOD_COLORS[m] for m in labels], alpha=0.85)
        ax3.set_xticks(x, labels, rotation=15, ha="right")
        ax3.set_ylabel("Absolute amplitude")
        ax3.set_title(f"{target_name} ({target_mz:.5f}) - Amplitude by method")
        ax3.grid(True, alpha=0.2, axis="y")
        for bar, amp in zip(bars, amplitudes):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{amp:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig3.tight_layout()
        amplitude_plot_path = outdir / f"{sanitize_name(target_name)}_02_amplitudes.png"
        fig3.savefig(amplitude_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved amplitude plot to: {amplitude_plot_path}")

        summary["targets"][target_name] = {
            "target_mz": target_mz,
            "methods": {
                method: {
                    "amplitude": float(method_results[method]["amplitude"]),
                    "ptp_reduction_pct": float(method_results[method]["ptp_reduction_pct"]),
                    "std_reduction_pct": float(method_results[method]["std_reduction_pct"]),
                }
                for method in METHODS
            },
        }

    summary_path = outdir / "targeted_mass_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    demo_file = Path(__file__).parent / "demo_data" / "MSCONVERT_CTRL_103_01_c_afterreboot_original.mzML"
    run_targeted_analysis(
        file_path=str(demo_file),
        output_dir="diagnostic_plots/targeted_mass_signal_analysis_2026_05_19",
        mz_window=0.1,
        rt_window=5.0,
        amplitude_multiplier=1.0,
        window_scan_size=70,
    )
