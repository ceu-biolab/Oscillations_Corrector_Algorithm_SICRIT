#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reference-mass phase diagnostic for SICRITfix.

This script does not change correction behavior. It compares the current
reference-phase estimate against alternative phase-start/frequency alignments so
we can see where phase disagreement happens.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sicritfix-project" / "src"))

from sicritfix.io.io import load_file
from sicritfix.utils.frequency_analyzer import (
    apply_polynomial_regression,
    build_dual_reference_phase,
    find_stable_signal_range,
    local_frequencies_with_fft,
)
from sicritfix.utils.intensity_analyzer import build_xic, get_amplitude


def load_arrays(file_path: str):
    exp = load_file(file_path)
    rt_array = np.array([spectrum.getRT() for spectrum in exp], dtype=float)
    mz_array = []
    intensity_array = []
    for spectrum in exp:
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(np.asarray(mzs, dtype=float))
        intensity_array.append(np.asarray(intensities, dtype=float))
    return rt_array, mz_array, intensity_array


def moving_average(signal, window_points):
    signal = np.asarray(signal, dtype=float)
    window_points = int(max(1, min(window_points, signal.size)))
    if window_points <= 1:
        return signal.copy()
    kernel = np.ones(window_points, dtype=float) / window_points
    return np.convolve(signal, kernel, mode="same")


def rolling_quantile(signal, window_points, quantile):
    signal = np.asarray(signal, dtype=float)
    window_points = int(max(1, min(window_points, signal.size)))
    if window_points <= 1:
        return signal.copy()
    if window_points % 2 == 0:
        window_points = max(1, window_points - 1)
    half = window_points // 2
    padded = np.pad(signal, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_points)
    return np.quantile(windows, quantile, axis=-1)


def estimate_baseline(signal, window_points, method):
    method = str(method).strip().lower()
    if method == "moving_average":
        return moving_average(signal, window_points)
    if method == "rolling_median":
        return rolling_quantile(signal, window_points, 0.5)
    raise ValueError("baseline_method must be one of: moving_average, rolling_median")


def first_consecutive_above(signal, threshold, min_consecutive=1):
    signal = np.asarray(signal, dtype=float)
    if threshold is None or threshold <= 0:
        return 0
    min_consecutive = int(max(1, min_consecutive))
    above = signal >= threshold
    if not np.any(above):
        return 0

    run_start = None
    run_length = 0
    for idx, is_above in enumerate(above):
        if is_above:
            if run_start is None:
                run_start = idx
            run_length += 1
            if run_length >= min_consecutive:
                return int(run_start)
        else:
            run_start = None
            run_length = 0
    return int(np.argmax(above))


def build_analysis_mask(signal_size, start_idx, end_idx=None):
    mask = np.zeros(int(signal_size), dtype=bool)
    if signal_size:
        if end_idx is None:
            end_idx = signal_size
        start_idx = int(max(0, min(start_idx, signal_size)))
        end_idx = int(max(start_idx, min(end_idx, signal_size)))
        mask[start_idx:end_idx] = True
    return mask


def robust_scale(signal):
    signal = np.asarray(signal, dtype=float)
    q25, q75 = np.percentile(signal, [25, 75])
    scale = (q75 - q25) / 1.349
    if scale <= 0 or not np.isfinite(scale):
        scale = np.std(signal)
    if scale <= 0 or not np.isfinite(scale):
        return 1.0
    return float(scale)


def normalize_signal(signal):
    return (np.asarray(signal, dtype=float) - np.median(signal)) / robust_scale(signal)


def weighted_corr(x, y, weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0
    x_mean = float(np.sum(weights * x) / weight_sum)
    y_mean = float(np.sum(weights * y) / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.sqrt(np.sum(weights * x_centered**2) * np.sum(weights * y_centered**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * x_centered * y_centered) / denom)


def weighted_corr_masked(x, y, weights, mask):
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    if np.sum(valid) < 3:
        return 0.0
    return weighted_corr(np.asarray(x)[valid], np.asarray(y)[valid], np.asarray(weights)[valid])


def fit_offset_lstsq(oscillation, phase, weights=None):
    if weights is None:
        weights = np.ones_like(oscillation, dtype=float)
    design = np.column_stack([np.sin(phase), np.cos(phase)])
    sqrt_weights = np.sqrt(weights)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            oscillation * sqrt_weights,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.arctan2(coeffs[1], coeffs[0]))


def refine_offset_by_correlation(oscillation, phase, weights, center, radius=np.pi, steps=721):
    offsets = np.linspace(center - radius, center + radius, int(steps))
    scores = np.array(
        [weighted_corr(oscillation, np.sin(phase + offset), weights) for offset in offsets],
        dtype=float,
    )
    idx = int(np.nanargmax(scores))
    return float(offsets[idx]), float(scores[idx])


def estimate_fft_frequency(segment, sampling_interval, method, zero_pad_factor):
    centered = np.asarray(segment, dtype=float) - np.mean(segment)
    if centered.size < 4 or np.allclose(centered, 0):
        return 0.0

    if method in {"hann", "hann_quadratic"}:
        centered = centered * np.hanning(centered.size)

    n_fft = centered.size
    if method in {"zeropad", "zeropad_quadratic"}:
        n_fft = int(centered.size * max(1, zero_pad_factor))

    fft_result = np.fft.rfft(centered, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=sampling_interval)
    magnitudes = np.abs(fft_result)
    if magnitudes.size <= 1:
        return 0.0

    magnitudes[0] = 0.0
    peak_idx = int(np.argmax(magnitudes))
    if peak_idx <= 0:
        return 0.0

    if method in {"quadratic", "hann_quadratic", "zeropad_quadratic"} and peak_idx < magnitudes.size - 1:
        eps = np.finfo(float).eps
        alpha = np.log(magnitudes[peak_idx - 1] + eps)
        beta = np.log(magnitudes[peak_idx] + eps)
        gamma = np.log(magnitudes[peak_idx + 1] + eps)
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > eps:
            delta = 0.5 * (alpha - gamma) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            return float((peak_idx + delta) * (freqs[1] - freqs[0]))

    return float(freqs[peak_idx])


def local_frequencies_with_fft_diagnostic(
    xic,
    rts,
    window_scan_size,
    sampling_interval,
    frequency_method="current_bin",
    zero_pad_factor=16,
):
    if frequency_method == "current_bin":
        return local_frequencies_with_fft(
            xic,
            rts,
            window_scan_size=window_scan_size,
            sampling_interval=sampling_interval,
        )

    local_freqs = []
    rt_freqs = []
    step = window_scan_size // 2

    for i in range(0, len(xic) - window_scan_size, step):
        segment = xic[i : i + window_scan_size]
        rt_segment = rts[i : i + window_scan_size]
        dominant_freq = estimate_fft_frequency(
            segment,
            sampling_interval,
            frequency_method,
            zero_pad_factor,
        )
        local_freqs.append(dominant_freq)
        rt_freqs.append(np.mean(rt_segment))

    return np.array(rt_freqs), np.array(local_freqs)


def fit_frequency_scale_and_offset(oscillation, phase, weights, scales):
    best = None
    for scale in scales:
        scaled_phase = phase * scale
        start_offset = fit_offset_lstsq(oscillation, scaled_phase, weights)
        offset, score = refine_offset_by_correlation(
            oscillation,
            scaled_phase,
            weights,
            start_offset,
            radius=0.35,
            steps=181,
        )
        if best is None or score > best["score"]:
            best = {"scale": float(scale), "offset": offset, "score": score}
    return best


def score_variant(name, oscillation, model, weights, rt_array, score_mask=None, rolling_seconds=300.0):
    if score_mask is None:
        score_mask = np.ones_like(oscillation, dtype=bool)
    score_mask = np.asarray(score_mask, dtype=bool)
    valid = score_mask & np.isfinite(oscillation) & np.isfinite(model) & np.isfinite(weights)

    denom = float(np.sum(weights[valid] * model[valid] ** 2)) if np.any(valid) else 0.0
    amplitude = float(np.sum(weights[valid] * oscillation[valid] * model[valid]) / denom) if denom > 0 else 0.0
    fitted_model = amplitude * model
    fitted_model[~score_mask] = 0.0
    corrected_oscillation = oscillation - fitted_model

    corr = weighted_corr(oscillation[valid], model[valid], weights[valid]) if np.any(valid) else 0.0
    sign_mask = valid & (np.sign(oscillation) != 0) & (np.sign(model) != 0)
    sign_agreement = float(np.mean(np.sign(oscillation[sign_mask]) == np.sign(model[sign_mask]))) if np.any(sign_mask) else 0.0
    rmse = float(
        np.sqrt(
            np.average(
                (normalize_signal(oscillation[valid]) - normalize_signal(model[valid])) ** 2,
                weights=weights[valid],
            )
        )
    ) if np.any(valid) else 0.0
    original_std = float(np.std(oscillation[valid])) if np.any(valid) else 0.0
    corrected_std = float(np.std(corrected_oscillation[valid])) if np.any(valid) else 0.0
    original_ptp = float(np.ptp(oscillation[valid])) if np.any(valid) else 0.0
    corrected_ptp = float(np.ptp(corrected_oscillation[valid])) if np.any(valid) else 0.0
    std_reduction_pct = 100.0 * (original_std - corrected_std) / original_std if original_std > 0 else 0.0
    ptp_reduction_pct = 100.0 * (original_ptp - corrected_ptp) / original_ptp if original_ptp > 0 else 0.0

    sampling_interval = float(np.mean(np.diff(rt_array)))
    rolling_points = max(3, int(round(rolling_seconds / sampling_interval)))
    rolling_corr = np.full_like(oscillation, np.nan, dtype=float)
    half = rolling_points // 2
    for i in range(oscillation.size):
        start = max(0, i - half)
        end = min(oscillation.size, i + half + 1)
        local_mask = score_mask[start:end]
        if end - start >= 5 and np.sum(local_mask) >= 5:
            rolling_corr[i] = weighted_corr(
                oscillation[start:end][local_mask],
                model[start:end][local_mask],
                weights[start:end][local_mask],
            )

    return {
        "name": name,
        "weighted_corr": corr,
        "sign_agreement": sign_agreement,
        "rmse_norm": rmse,
        "amplitude": amplitude,
        "fitted_model": fitted_model,
        "corrected_oscillation": corrected_oscillation,
        "std_reduction_pct": std_reduction_pct,
        "ptp_reduction_pct": ptp_reduction_pct,
        "rolling_corr": rolling_corr,
    }


def build_phase_current(
    rt_array,
    xic,
    window_scan_size,
    frequency_method="current_bin",
    zero_pad_factor=16,
    start_idx=0,
    end_idx=None,
):
    sampling_interval = float(np.mean(np.diff(rt_array)))
    if end_idx is None:
        end_idx = len(rt_array)
    rt_segment = np.asarray(rt_array, dtype=float)[start_idx:end_idx]
    xic_segment = np.asarray(xic, dtype=float)[start_idx:end_idx]
    rt_freqs, local_freqs = local_frequencies_with_fft_diagnostic(
        xic_segment,
        rt_segment,
        window_scan_size=window_scan_size,
        sampling_interval=sampling_interval,
        frequency_method=frequency_method,
        zero_pad_factor=zero_pad_factor,
    )
    phase = np.zeros_like(np.asarray(rt_array, dtype=float))
    if rt_segment.size:
        phase[start_idx:end_idx] = apply_polynomial_regression(rt_segment, rt_freqs, local_freqs)
    return rt_freqs, local_freqs, phase


def build_phase_shifted_fit(rt_array, rt_freqs, local_freqs, start_idx=0, end_idx=None):
    rts = np.asarray(rt_array, dtype=float)
    if end_idx is None:
        end_idx = rts.size
    rt_freqs = np.asarray(rt_freqs, dtype=float)
    local_freqs = np.asarray(local_freqs, dtype=float)
    phase = np.zeros_like(rts, dtype=float)
    if rt_freqs.size == 0 or local_freqs.size == 0:
        return phase
    stable_rts = rts[start_idx:end_idx]
    if stable_rts.size == 0:
        return phase
    t = stable_rts - stable_rts[0]
    freq_interp = np.interp(stable_rts, rt_freqs, local_freqs)
    fit = np.polyfit(t, freq_interp, 2)
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)
    phase[start_idx:end_idx] = 2.0 * np.pi * np.concatenate([[0.0], np.cumsum((f_t[1:] + f_t[:-1]) * 0.5 * np.diff(t))])
    return phase


def build_constant_frequency_phase(rt_array, rt_freqs, local_freqs, start_idx=0, end_idx=None):
    rts = np.asarray(rt_array, dtype=float)
    if end_idx is None:
        end_idx = rts.size
    phase = np.zeros_like(rts, dtype=float)
    valid_freqs = np.asarray(local_freqs, dtype=float)
    valid_freqs = valid_freqs[np.isfinite(valid_freqs) & (valid_freqs > 0)]
    if valid_freqs.size == 0 or start_idx >= end_idx:
        return phase, 0.0
    constant_freq = float(np.median(valid_freqs))
    t = rts[start_idx:end_idx] - rts[start_idx]
    phase[start_idx:end_idx] = 2.0 * np.pi * constant_freq * t
    return phase, constant_freq


def frequency_curve_current(rt_array, rt_freqs, local_freqs):
    rts = np.asarray(rt_array, dtype=float)
    if len(rt_freqs) == 0:
        return np.zeros_like(rts)
    t = rts - rts[0]
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit = np.polyfit(rts, freq_interp, 2)
    return np.poly1d(fit)(t)


def frequency_curve_shifted(rt_array, rt_freqs, local_freqs):
    rts = np.asarray(rt_array, dtype=float)
    if len(rt_freqs) == 0:
        return np.zeros_like(rts)
    t = rts - rts[0]
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit = np.polyfit(t, freq_interp, 2)
    return np.poly1d(fit)(t)


def run_reference_diagnostic(
    input_file,
    output_dir,
    corrected_input=None,
    reference_mz=922.009798,
    secondary_reference_mz=None,
    mz_window=0.1,
    rt_window=5.0,
    window_scan_size=70,
    baseline_seconds=300.0,
    frequency_method="current_bin",
    zero_pad_factor=16,
    ignore_until_intensity=5e6,
    ignore_consecutive_scans=5,
    baseline_method="rolling_median",
    amplitude_percentile=75.0,
    quality_window_minutes=4.0,
    quality_step_minutes=2.0,
    quality_corr_threshold=0.5,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rt_array, mz_array, intensity_array = load_arrays(input_file)
    sampling_interval = float(np.mean(np.diff(rt_array)))
    rt_minutes = rt_array / 60.0
    tick_minutes = 100.0 / 60.0

    xic = build_xic(
        mz_array,
        intensity_array,
        rt_array,
        target_mz=reference_mz,
        rt_window=rt_window,
        mz_window=mz_window,
    )

    stable_start_idx, stable_end_idx = find_stable_signal_range(
        xic,
        intensity_threshold=ignore_until_intensity,
        min_consecutive_scans=ignore_consecutive_scans,
    )
    score_mask = build_analysis_mask(xic.size, stable_start_idx, stable_end_idx)
    stable_start_rt = float(rt_array[stable_start_idx]) if rt_array.size else 0.0
    stable_end_rt = float(rt_array[stable_end_idx - 1]) if stable_end_idx > stable_start_idx else 0.0
    stable_start_min = stable_start_rt / 60.0
    stable_end_min = stable_end_rt / 60.0

    baseline_points = max(3, int(round(baseline_seconds / sampling_interval)))
    baseline = xic.copy()
    if stable_start_idx < stable_end_idx:
        stable_baseline = estimate_baseline(
            xic[stable_start_idx:stable_end_idx],
            baseline_points,
            baseline_method,
        )
        baseline[stable_start_idx:stable_end_idx] = stable_baseline
    oscillation = xic - baseline
    oscillation_norm = normalize_signal(oscillation)

    weight_points = max(3, baseline_points // 4)
    weights = moving_average(np.abs(oscillation), weight_points)
    high_weight = float(np.percentile(weights, 95))
    weights = np.clip(weights / high_weight, 0.05, 1.0) if high_weight > 0 else np.ones_like(oscillation)
    weights[~score_mask] = 0.05

    rt_freqs, local_freqs, phase_current = build_phase_current(
        rt_array,
        xic,
        window_scan_size,
        frequency_method=frequency_method,
        zero_pad_factor=zero_pad_factor,
        start_idx=stable_start_idx,
        end_idx=stable_end_idx,
    )
    phase_shifted_fit = build_phase_shifted_fit(
        rt_array,
        rt_freqs,
        local_freqs,
        start_idx=stable_start_idx,
        end_idx=stable_end_idx,
    )
    phase_constant, constant_freq = build_constant_frequency_phase(
        rt_array,
        rt_freqs,
        local_freqs,
        start_idx=stable_start_idx,
        end_idx=stable_end_idx,
    )
    dual_reference_metadata = {}
    phase_dual = None
    dual_local_freqs = local_freqs
    secondary_xic = None
    if secondary_reference_mz is not None:
        secondary_xic = build_xic(
            mz_array,
            intensity_array,
            rt_array,
            target_mz=secondary_reference_mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )
        _, dual_local_freqs, phase_dual, dual_reference_metadata = build_dual_reference_phase(
            rt_array,
            xic,
            secondary_xic,
            window_scan_size=window_scan_size,
            sampling_interval=sampling_interval,
            stable_start_idx=stable_start_idx,
            stable_end_idx=stable_end_idx,
            quality_window_minutes=quality_window_minutes,
            quality_step_minutes=quality_step_minutes,
            quality_corr_threshold=quality_corr_threshold,
            baseline_seconds=baseline_seconds,
        )
    f_current = frequency_curve_current(rt_array, rt_freqs, local_freqs)
    f_shifted = frequency_curve_shifted(rt_array, rt_freqs, local_freqs)

    variants = []
    candidates = {
        "current_phase": phase_current,
        "current_plus_pi": phase_current + np.pi,
        "shifted_time_fit": phase_shifted_fit,
        "constant_frequency": phase_constant,
    }
    if phase_dual is not None:
        candidates["dual_reference_phase"] = phase_dual

    for base_name, phase in candidates.items():
        model = np.sin(phase)
        variants.append((base_name, phase, model, {"offset": 0.0, "scale": 1.0}))

        offset = fit_offset_lstsq(oscillation[score_mask], phase[score_mask])
        variants.append((f"{base_name}_global_offset", phase + offset, np.sin(phase + offset), {"offset": offset, "scale": 1.0}))

        weighted_offset = fit_offset_lstsq(oscillation[score_mask], phase[score_mask], weights[score_mask])
        refined_offset, refined_score = refine_offset_by_correlation(
            oscillation[score_mask],
            phase[score_mask],
            weights[score_mask],
            weighted_offset,
        )
        variants.append(
            (
                f"{base_name}_weighted_refined_offset",
                phase + refined_offset,
                np.sin(phase + refined_offset),
                {"offset": refined_offset, "scale": 1.0, "refine_score": refined_score},
            )
        )

    scale_grid = np.linspace(0.9, 1.1, 81)
    scale_fit = fit_frequency_scale_and_offset(
        oscillation[score_mask],
        phase_current[score_mask],
        weights[score_mask],
        scale_grid,
    )
    scaled_phase = phase_current * scale_fit["scale"] + scale_fit["offset"]
    variants.append(
        (
            "current_phase_scale_refined",
            scaled_phase,
            np.sin(scaled_phase),
            scale_fit,
        )
    )

    scored = []
    for name, phase, model, metadata in variants:
        score = score_variant(name, oscillation, model, weights, rt_array, score_mask=score_mask)
        scored.append({**score, "phase": phase, "model": model, **metadata})
    scored.sort(key=lambda item: item["weighted_corr"], reverse=True)

    amplitude_phase_item = scored[0]
    if phase_dual is not None:
        for item in scored:
            if item["name"] == "dual_reference_phase_weighted_refined_offset":
                amplitude_phase_item = item
                break

    amplitude_methods = ["q75", "q90", "global_trimmed_detrended", "local_robust_detrended"]
    amplitude_method_scores = []
    stable_xic = xic[stable_start_idx:stable_end_idx]
    best_model = amplitude_phase_item["model"]
    valid = score_mask & np.isfinite(oscillation) & np.isfinite(best_model)
    for method in amplitude_methods:
        amplitude = get_amplitude(
            stable_xic,
            dual_local_freqs,
            sampling_interval,
            method=method,
            summary_percentile=amplitude_percentile,
        )
        fitted_model = np.zeros_like(xic, dtype=float)
        if stable_start_idx < stable_end_idx:
            fitted_model[stable_start_idx:stable_end_idx] = amplitude * best_model[stable_start_idx:stable_end_idx]
        corrected_oscillation = oscillation - fitted_model
        original_std = float(np.std(oscillation[valid])) if np.any(valid) else 0.0
        corrected_std = float(np.std(corrected_oscillation[valid])) if np.any(valid) else 0.0
        original_ptp = float(np.ptp(oscillation[valid])) if np.any(valid) else 0.0
        corrected_ptp = float(np.ptp(corrected_oscillation[valid])) if np.any(valid) else 0.0
        amplitude_method_scores.append(
            {
                "method": method,
                "amplitude": float(amplitude),
                "subtraction_std": float(np.std(fitted_model[valid])) if np.any(valid) else 0.0,
                "std_reduction_pct": 100.0 * (original_std - corrected_std) / original_std if original_std > 0 else 0.0,
                "ptp_reduction_pct": 100.0 * (original_ptp - corrected_ptp) / original_ptp if original_ptp > 0 else 0.0,
                "fitted_model": fitted_model,
                "corrected_xic": xic - fitted_model,
            }
        )

    amplitude_method_scores.append(
        {
            "method": "diagnostic_least_squares_reference",
            "amplitude": float(amplitude_phase_item["amplitude"]),
            "subtraction_std": float(np.std(amplitude_phase_item["fitted_model"][valid])) if np.any(valid) else 0.0,
            "std_reduction_pct": float(amplitude_phase_item["std_reduction_pct"]),
            "ptp_reduction_pct": float(amplitude_phase_item["ptp_reduction_pct"]),
            "fitted_model": amplitude_phase_item["fitted_model"],
            "corrected_xic": xic - amplitude_phase_item["fitted_model"],
        }
    )

    corrected_xic_from_file = None
    corrected_delta_stats = None
    if corrected_input:
        corrected_rt_array, corrected_mz_array, corrected_intensity_array = load_arrays(corrected_input)
        corrected_xic_from_file = build_xic(
            corrected_mz_array,
            corrected_intensity_array,
            corrected_rt_array,
            target_mz=reference_mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )
        if corrected_xic_from_file.shape == xic.shape:
            delta = xic - corrected_xic_from_file
            valid = score_mask & np.isfinite(delta)
            best_subtraction = scored[0]["fitted_model"]
            corrected_delta_stats = {
                "max_abs_original_minus_saved": float(np.max(np.abs(delta[valid]))) if np.any(valid) else 0.0,
                "mean_abs_original_minus_saved": float(np.mean(np.abs(delta[valid]))) if np.any(valid) else 0.0,
                "median_abs_original_minus_saved": float(np.median(np.abs(delta[valid]))) if np.any(valid) else 0.0,
                "original_minus_saved_std": float(np.std(delta[valid])) if np.any(valid) else 0.0,
                "diagnostic_subtraction_std": float(np.std(best_subtraction[valid])) if np.any(valid) else 0.0,
                "original_minus_saved_vs_diagnostic_subtraction_corr": (
                    weighted_corr(delta[valid], best_subtraction[valid], weights[valid]) if np.any(valid) else 0.0
                ),
            }
        else:
            corrected_delta_stats = {
                "error": "Corrected mzML XIC length differs from original XIC length.",
                "original_points": int(xic.size),
                "corrected_points": int(corrected_xic_from_file.size),
            }

    csv_path = output_path / "phase_variant_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "weighted_corr",
                "sign_agreement",
                "rmse_norm",
                "amplitude",
                "std_reduction_pct",
                "ptp_reduction_pct",
                "offset",
                "scale",
                "refine_score",
            ],
        )
        writer.writeheader()
        for item in scored:
            writer.writerow(
                {
                    "name": item["name"],
                    "weighted_corr": item["weighted_corr"],
                    "sign_agreement": item["sign_agreement"],
                    "rmse_norm": item["rmse_norm"],
                    "amplitude": item["amplitude"],
                    "std_reduction_pct": item["std_reduction_pct"],
                    "ptp_reduction_pct": item["ptp_reduction_pct"],
                    "offset": item.get("offset", ""),
                    "scale": item.get("scale", ""),
                    "refine_score": item.get("refine_score", ""),
                }
            )

    amplitude_csv_path = output_path / "amplitude_method_scores.csv"
    with amplitude_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "amplitude",
                "subtraction_std",
                "std_reduction_pct",
                "ptp_reduction_pct",
            ],
        )
        writer.writeheader()
        for item in amplitude_method_scores:
            writer.writerow(
                {
                    "method": item["method"],
                    "amplitude": item["amplitude"],
                    "subtraction_std": item["subtraction_std"],
                    "std_reduction_pct": item["std_reduction_pct"],
                    "ptp_reduction_pct": item["ptp_reduction_pct"],
                }
            )

    quality_csv_path = output_path / "dual_reference_quality_windows.csv"
    if dual_reference_metadata:
        with quality_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "start_min",
                    "end_min",
                    "corr",
                    "is_bad",
                ],
            )
            writer.writeheader()
            for item in dual_reference_metadata.get("quality_windows", []):
                writer.writerow(
                    {
                        "start_min": item["start_rt"] / 60.0,
                        "end_min": item["end_rt"] / 60.0,
                        "corr": item["corr"],
                        "is_bad": item["corr"] < quality_corr_threshold,
                    }
                )

    source_csv_path = output_path / "dual_reference_source_intervals.csv"
    if dual_reference_metadata:
        with source_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "start_min",
                    "end_min",
                    "source",
                    "main_min_corr",
                    "secondary_corr",
                    "frequency_hz",
                ],
            )
            writer.writeheader()
            for item in dual_reference_metadata.get("source_intervals", []):
                writer.writerow(
                    {
                        "start_min": item["start_min"],
                        "end_min": item["end_min"],
                        "source": item["source"],
                        "main_min_corr": item["main_min_corr"],
                        "secondary_corr": item["secondary_corr"],
                        "frequency_hz": item["frequency_hz"],
                    }
                )

    summary = {
        "input_file": input_file,
        "reference_mz": reference_mz,
        "secondary_reference_mz": secondary_reference_mz,
        "mz_window": mz_window,
        "rt_window": rt_window,
        "window_scan_size": window_scan_size,
        "frequency_method": frequency_method,
        "zero_pad_factor": zero_pad_factor,
        "ignore_until_intensity": ignore_until_intensity,
        "ignore_consecutive_scans": ignore_consecutive_scans,
        "stable_start_index": stable_start_idx,
        "stable_end_index": stable_end_idx,
        "stable_start_rt_seconds": stable_start_rt,
        "stable_end_rt_seconds": stable_end_rt,
        "stable_start_rt_minutes": stable_start_min,
        "stable_end_rt_minutes": stable_end_min,
        "baseline_method": baseline_method,
        "amplitude_percentile": amplitude_percentile,
        "quality_window_minutes": quality_window_minutes,
        "quality_step_minutes": quality_step_minutes,
        "quality_corr_threshold": quality_corr_threshold,
        "amplitude_phase_variant": amplitude_phase_item["name"],
        "dual_reference_metadata": dual_reference_metadata,
        "sampling_interval_seconds": sampling_interval,
        "baseline_seconds": baseline_seconds,
        "constant_frequency_hz": constant_freq,
        "local_frequency_mean_hz": float(np.mean(local_freqs)) if local_freqs.size else None,
        "local_frequency_median_hz": float(np.median(local_freqs)) if local_freqs.size else None,
        "corrected_input": corrected_input,
        "corrected_delta_stats": corrected_delta_stats,
        "best_variant": scored[0]["name"],
        "scores": [
            {
                "name": item["name"],
                "weighted_corr": item["weighted_corr"],
                "sign_agreement": item["sign_agreement"],
                "rmse_norm": item["rmse_norm"],
                "amplitude": item["amplitude"],
                "std_reduction_pct": item["std_reduction_pct"],
                "ptp_reduction_pct": item["ptp_reduction_pct"],
                "offset": item.get("offset"),
                "scale": item.get("scale"),
            }
            for item in scored
        ],
        "amplitude_method_scores": [
            {
                "method": item["method"],
                "amplitude": item["amplitude"],
                "subtraction_std": item["subtraction_std"],
                "std_reduction_pct": item["std_reduction_pct"],
                "ptp_reduction_pct": item["ptp_reduction_pct"],
            }
            for item in amplitude_method_scores
        ],
    }
    summary_path = output_path / "reference_phase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top = scored[:4]

    def mark_ignored(ax):
        if stable_start_idx > 0:
            ax.axvspan(rt_minutes[0], stable_start_min, color="#f97316", alpha=0.12, label="Ignored insertion ramp")
            ax.axvline(stable_start_min, color="#f97316", linewidth=1.0, alpha=0.8)
        if stable_end_idx < rt_minutes.size:
            tail_start_min = rt_minutes[stable_end_idx]
            ax.axvspan(tail_start_min, rt_minutes[-1], color="#f97316", alpha=0.12, label="Ignored end ramp")
            ax.axvline(tail_start_min, color="#f97316", linewidth=1.0, alpha=0.8)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    mark_ignored(axes[0])
    axes[0].plot(rt_minutes, xic, color="black", linewidth=0.9, label="Reference XIC")
    axes[0].plot(rt_minutes, baseline, color="#2563eb", linewidth=1.0, label="Moving baseline")
    axes[0].axhline(ignore_until_intensity, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.8, label="Ignore threshold")
    axes[0].set_title(
        f"Reference m/z {reference_mz:.6f} XIC | stable region {stable_start_min:.2f}-{stable_end_min:.2f} min"
    )
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="upper right")

    mark_ignored(axes[1])
    axes[1].plot(rt_minutes, oscillation_norm, color="black", linewidth=0.8, label="Detrended XIC (normalized)")
    for item in top:
        axes[1].plot(rt_minutes, normalize_signal(item["model"]), linewidth=0.8, label=item["name"])
    axes[1].set_title("Detrended reference signal vs candidate sine models")
    axes[1].set_ylabel("Normalized")
    axes[1].legend(loc="upper right", fontsize=8)

    mark_ignored(axes[2])
    for item in top:
        axes[2].plot(rt_minutes, np.unwrap(item["phase"]), linewidth=0.8, label=item["name"])
    axes[2].set_title("Candidate phase traces")
    axes[2].set_ylabel("Phase (rad)")
    axes[2].legend(loc="upper right", fontsize=8)

    mark_ignored(axes[3])
    for item in top:
        axes[3].plot(rt_minutes, item["rolling_corr"], linewidth=0.8, label=item["name"])
    axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[3].set_title("Rolling weighted correlation vs detrended reference XIC")
    axes[3].set_ylabel("Rolling corr")
    axes[3].set_xlabel("Retention time (min; major ticks every 100 s)")
    axes[3].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(MultipleLocator(tick_minutes))

    fig.tight_layout()
    overview_path = output_path / "reference_phase_overview.png"
    fig.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    mark_ignored(ax)
    ax.plot(rt_freqs / 60.0, local_freqs, "o-", markersize=3, linewidth=0.8)
    ax.plot(rt_minutes, f_current, linewidth=1.2, label="Current polynomial frequency")
    ax.plot(rt_minutes, f_shifted, linewidth=1.2, linestyle="--", label="Shifted-time polynomial frequency")
    if constant_freq > 0:
        ax.axhline(constant_freq, color="#16a34a", linewidth=1.2, linestyle=":", label="Stable median constant frequency")
    ax.set_title(
        f"Local FFT frequencies, window_scan_size={window_scan_size}, "
        f"method={frequency_method}, zero_pad_factor={zero_pad_factor}"
    )
    ax.set_xlabel("Retention time (min; major ticks every 100 s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(MultipleLocator(tick_minutes))
    ax.legend(loc="upper right", fontsize=8)
    freq_path = output_path / "reference_local_frequencies.png"
    fig.savefig(freq_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(top), 1, figsize=(18, 4.2 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]
    for ax, item in zip(axes, top):
        corrected_xic = xic - item["fitted_model"]
        mark_ignored(ax)
        ax.plot(rt_minutes, xic, color="black", linewidth=0.8, label="Original reference XIC")
        ax.plot(rt_minutes, corrected_xic, color="#2563eb", linewidth=0.8, label="Diagnostic corrected XIC")
        ax.plot(rt_minutes, baseline, color="#64748b", linewidth=0.8, alpha=0.8, label="Moving baseline")
        ax.plot(
            rt_minutes,
            baseline + item["fitted_model"],
            color="#dc2626",
            linewidth=0.8,
            alpha=0.9,
            label="Baseline + signal to subtract",
        )
        ax_sub = ax.twinx()
        ax_sub.plot(
            rt_minutes,
            item["fitted_model"],
            color="#b91c1c",
            linewidth=0.65,
            alpha=0.55,
            label="Signal to subtract",
        )
        ax_sub.set_ylabel("Subtraction signal")
        ax.set_title(
            f"{item['name']} | corr={item['weighted_corr']:.3f} | "
            f"STD reduction={item['std_reduction_pct']:.2f}% | "
            f"PTP reduction={item['ptp_reduction_pct']:.2f}%"
        )
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(MultipleLocator(tick_minutes))
        handles, labels = ax.get_legend_handles_labels()
        handles_sub, labels_sub = ax_sub.get_legend_handles_labels()
        ax.legend(handles + handles_sub, labels + labels_sub, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Retention time (min; major ticks every 100 s)")
    fig.tight_layout()
    phase_correction_path = output_path / "reference_phase_variants_original_vs_corrected.png"
    fig.savefig(phase_correction_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(amplitude_method_scores), 1, figsize=(18, 4.0 * len(amplitude_method_scores)), sharex=True)
    if len(amplitude_method_scores) == 1:
        axes = [axes]
    for ax, item in zip(axes, amplitude_method_scores):
        mark_ignored(ax)
        ax.plot(rt_minutes, xic, color="black", linewidth=0.8, label="Original reference XIC")
        ax.plot(rt_minutes, item["corrected_xic"], color="#2563eb", linewidth=0.8, label="Corrected XIC")
        ax.plot(rt_minutes, baseline, color="#64748b", linewidth=0.8, alpha=0.8, label="Moving baseline")
        ax.plot(
            rt_minutes,
            baseline + item["fitted_model"],
            color="#dc2626",
            linewidth=0.8,
            alpha=0.9,
            label="Baseline + signal to subtract",
        )
        ax_sub = ax.twinx()
        ax_sub.plot(
            rt_minutes,
            item["fitted_model"],
            color="#b91c1c",
            linewidth=0.65,
            alpha=0.55,
            label="Signal to subtract",
        )
        ax_sub.set_ylabel("Subtraction signal")
        ax.set_title(
            f"{item['method']} | amplitude={item['amplitude']:.1f} | "
            f"STD reduction={item['std_reduction_pct']:.2f}% | "
            f"PTP reduction={item['ptp_reduction_pct']:.2f}%"
        )
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(MultipleLocator(tick_minutes))
        handles, labels = ax.get_legend_handles_labels()
        handles_sub, labels_sub = ax_sub.get_legend_handles_labels()
        ax.legend(handles + handles_sub, labels + labels_sub, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Retention time (min; major ticks every 100 s)")
    fig.tight_layout()
    correction_path = output_path / "reference_original_vs_corrected.png"
    amplitude_plot_path = output_path / "reference_amplitude_methods_original_vs_corrected.png"
    fig.savefig(correction_path, dpi=180, bbox_inches="tight")
    fig.savefig(amplitude_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Best variant: {scored[0]['name']} corr={scored[0]['weighted_corr']:.4f}")
    print(f"Saved overview plot to: {overview_path}")
    print(f"Saved frequency plot to: {freq_path}")
    print(f"Saved phase-variant original-vs-corrected plot to: {phase_correction_path}")
    print(f"Saved original-vs-corrected plot to: {correction_path}")
    print(f"Saved amplitude-method plot to: {amplitude_plot_path}")

    if corrected_xic_from_file is not None:
        fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
        mark_ignored(axes[0])
        axes[0].plot(rt_minutes, xic, color="black", linewidth=0.8, label="Original reference XIC")
        axes[0].plot(rt_minutes, corrected_xic_from_file, color="#7c3aed", linewidth=0.8, label="Saved mzML corrected XIC")
        axes[0].plot(rt_minutes, xic - scored[0]["fitted_model"], color="#2563eb", linewidth=0.8, label="Diagnostic best corrected XIC")
        axes[0].set_title("Original vs saved corrected mzML vs diagnostic corrected")
        axes[0].set_ylabel("Intensity")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(loc="upper right", fontsize=8)

        mark_ignored(axes[1])
        axes[1].plot(rt_minutes, xic - corrected_xic_from_file, color="#7c3aed", linewidth=0.8, label="Original - saved mzML corrected")
        axes[1].plot(rt_minutes, scored[0]["fitted_model"], color="#dc2626", linewidth=0.8, label="Diagnostic signal to subtract")
        axes[1].set_title("Saved mzML subtraction vs diagnostic subtraction")
        axes[1].set_ylabel("Intensity difference")
        axes[1].set_xlabel("Retention time (min; major ticks every 100 s)")
        axes[1].grid(True, alpha=0.2)
        axes[1].legend(loc="upper right", fontsize=8)
        for ax in axes:
            ax.xaxis.set_major_locator(MultipleLocator(tick_minutes))
        fig.tight_layout()
        saved_compare_path = output_path / "reference_saved_mzml_comparison.png"
        fig.savefig(saved_compare_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved corrected mzML comparison plot to: {saved_compare_path}")

    print(f"Saved scores to: {csv_path}")
    print(f"Saved amplitude method scores to: {amplitude_csv_path}")
    if dual_reference_metadata:
        print(f"Saved dual-reference quality windows to: {quality_csv_path}")
        print(f"Saved dual-reference source intervals to: {source_csv_path}")
    print(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose reference-mass phase alignment.")
    parser.add_argument("--input", required=True, help="Input mzML file.")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots and score tables.")
    parser.add_argument("--corrected_input", default=None, help="Optional corrected mzML file to compare against the original.")
    parser.add_argument("--reference_mz", type=float, default=922.009798)
    parser.add_argument("--secondary_reference_mz", type=float, default=None)
    parser.add_argument("--mz_window", type=float, default=0.1)
    parser.add_argument("--rt_window", type=float, default=5.0)
    parser.add_argument("--window_scan_size", type=int, default=70)
    parser.add_argument("--baseline_seconds", type=float, default=300.0)
    parser.add_argument("--amplitude_percentile", type=float, default=75.0)
    parser.add_argument("--quality_window_minutes", type=float, default=4.0)
    parser.add_argument("--quality_step_minutes", type=float, default=2.0)
    parser.add_argument("--quality_corr_threshold", type=float, default=0.5)
    parser.add_argument(
        "--ignore_until_intensity",
        type=float,
        default=5e6,
        help="Ignore scans before the reference XIC reaches this intensity. Use 0 to disable.",
    )
    parser.add_argument(
        "--ignore_consecutive_scans",
        type=int,
        default=5,
        help="Number of consecutive scans above --ignore_until_intensity required to mark the stable start.",
    )
    parser.add_argument(
        "--baseline_method",
        choices=["moving_average", "rolling_median"],
        default="rolling_median",
        help="Baseline estimator used before fitting the subtraction signal.",
    )
    parser.add_argument(
        "--frequency_method",
        choices=[
            "current_bin",
            "quadratic",
            "zeropad",
            "zeropad_quadratic",
            "hann",
            "hann_quadratic",
        ],
        default="current_bin",
        help="Diagnostic-only local frequency estimator.",
    )
    parser.add_argument("--zero_pad_factor", type=int, default=16)
    args = parser.parse_args()

    run_reference_diagnostic(
        input_file=args.input,
        output_dir=args.output_dir,
        corrected_input=args.corrected_input,
        reference_mz=args.reference_mz,
        secondary_reference_mz=args.secondary_reference_mz,
        mz_window=args.mz_window,
        rt_window=args.rt_window,
        window_scan_size=args.window_scan_size,
        baseline_seconds=args.baseline_seconds,
        frequency_method=args.frequency_method,
        zero_pad_factor=args.zero_pad_factor,
        ignore_until_intensity=args.ignore_until_intensity,
        ignore_consecutive_scans=args.ignore_consecutive_scans,
        baseline_method=args.baseline_method,
        amplitude_percentile=args.amplitude_percentile,
        quality_window_minutes=args.quality_window_minutes,
        quality_step_minutes=args.quality_step_minutes,
        quality_corr_threshold=args.quality_corr_threshold,
    )


if __name__ == "__main__":
    main()
