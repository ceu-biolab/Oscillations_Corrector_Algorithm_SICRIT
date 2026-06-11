# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This Python module contains the unit tests for the corrector.py module in the SICRITfix project.

@contents  :  Unit tests for sinusoidal signal generation and oscillation correction.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  test_corrector.py
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@version   :  0.0.1, 24 July 2025
@information :
    The Zen of Python
        https://www.python.org/dev/peps/pep-0020/
    Style Guide for Python Code
        https://www.python.org/dev/peps/pep-0008/
    Example NumPy Style Python Docstrings
        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    doctest – Testing through documentation
        https://pymotw.com/2/doctest/

@dependencies :
    - numpy
    - unittest
    - sicritfix.processing.corrector
    - sicritfix.utils.intensity_analyzer

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""

__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"

import unittest
import numpy as np
from sicritfix.processing.corrector import generate_modulated_signal, correct_oscillations

class TestCorrector(unittest.TestCase):

    def test_generate_modulated_signal_scalar(self):
        amplitude = 1.0
        phase = np.pi / 2
        result = generate_modulated_signal(amplitude, phase)
        self.assertAlmostEqual(result, 1.0, delta=1e-6)

    def test_generate_modulated_signal_array(self):
        amplitude = np.array([1, 2, 3])
        phase = np.array([0, np.pi/2, np.pi])
        expected = amplitude * np.sin(phase)
        result = generate_modulated_signal(amplitude, phase)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_correct_oscillations_outputs(self):
        rt_array = np.linspace(0, 10, 100)
        mz_array = [np.array([100.0, 200.0]) for _ in range(100)]
        intensity_array = [np.array([0.0, 10.0 + 5 * np.sin(2 * np.pi * 0.5 * t)]) for t in rt_array]
        phase_ref = 2 * np.pi * 0.5 * rt_array
        local_freqs_ref = np.full_like(rt_array, 0.5)

        xic, modulated_signal, residual_signal = correct_oscillations(
            rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz=200.0, rt_window=1
        )

        self.assertEqual(len(xic), len(rt_array))
        self.assertEqual(len(modulated_signal), len(rt_array))
        self.assertEqual(len(residual_signal), len(rt_array))

        # Check signal shape
        self.assertTrue(np.allclose(xic, modulated_signal + residual_signal, rtol=1e-4))

    def test_correct_oscillations_applies_amplitude_multiplier(self):
        rt_array = np.linspace(0, 10, 100)
        mz_array = [np.array([100.0, 200.0]) for _ in range(100)]
        intensity_array = [np.array([0.0, 10.0 + 5 * np.sin(2 * np.pi * 0.5 * t)]) for t in rt_array]
        phase_ref = 2 * np.pi * 0.5 * rt_array
        local_freqs_ref = np.full_like(rt_array, 0.5)

        _, modulated_signal_base, _ = correct_oscillations(
            rt_array,
            mz_array,
            intensity_array,
            phase_ref,
            local_freqs_ref,
            target_mz=200.0,
            rt_window=1,
            amplitude_multiplier=1.0,
        )
        _, modulated_signal_scaled, _ = correct_oscillations(
            rt_array,
            mz_array,
            intensity_array,
            phase_ref,
            local_freqs_ref,
            target_mz=200.0,
            rt_window=1,
            amplitude_multiplier=2.0,
        )

        np.testing.assert_allclose(modulated_signal_scaled, 2.0 * modulated_signal_base, rtol=1e-4, atol=1e-4)

    def test_correct_oscillations_ignores_ramp_before_analysis_start(self):
        rt_array = np.linspace(0, 40, 400)
        mz_array = [np.array([100.0, 200.0]) for _ in range(400)]
        phase_ref = 2 * np.pi * 0.2 * rt_array
        intensity_array = [np.array([0.0, 10.0 + 5 * np.sin(phase_ref[i])]) for i in range(400)]
        local_freqs_ref = np.full_like(rt_array, 0.2)

        analysis_start_idx = 50
        _, modulated_signal, residual_signal = correct_oscillations(
            rt_array,
            mz_array,
            intensity_array,
            phase_ref,
            local_freqs_ref,
            target_mz=200.0,
            rt_window=0.01,
            amplitude_method="q90",
            analysis_start_idx=analysis_start_idx,
        )

        np.testing.assert_allclose(modulated_signal[:analysis_start_idx], 0.0, atol=1e-12)
        self.assertGreater(np.max(np.abs(modulated_signal[analysis_start_idx:])), 0.0)
        xic_before_start = np.array([values[1] for values in intensity_array[:analysis_start_idx]])
        np.testing.assert_allclose(residual_signal[:analysis_start_idx], xic_before_start, rtol=1e-6)

    def test_correct_oscillations_ignores_ramp_after_analysis_end(self):
        rt_array = np.linspace(0, 40, 400)
        mz_array = [np.array([100.0, 200.0]) for _ in range(400)]
        phase_ref = 2 * np.pi * 0.2 * rt_array
        intensity_array = [np.array([0.0, 10.0 + 5 * np.sin(phase_ref[i])]) for i in range(400)]
        local_freqs_ref = np.full_like(rt_array, 0.2)

        analysis_start_idx = 50
        analysis_end_idx = 350
        _, modulated_signal, residual_signal = correct_oscillations(
            rt_array,
            mz_array,
            intensity_array,
            phase_ref,
            local_freqs_ref,
            target_mz=200.0,
            rt_window=0.01,
            amplitude_method="q90",
            analysis_start_idx=analysis_start_idx,
            analysis_end_idx=analysis_end_idx,
        )

        np.testing.assert_allclose(modulated_signal[:analysis_start_idx], 0.0, atol=1e-12)
        np.testing.assert_allclose(modulated_signal[analysis_end_idx:], 0.0, atol=1e-12)
        self.assertGreater(np.max(np.abs(modulated_signal[analysis_start_idx:analysis_end_idx])), 0.0)
        xic_after_end = np.array([values[1] for values in intensity_array[analysis_end_idx:]])
        np.testing.assert_allclose(residual_signal[analysis_end_idx:], xic_after_end, rtol=1e-6)

    def test_correct_oscillations_fits_phase_offset_per_mz(self):
        rt_array = np.linspace(0, 40, 400)
        mz_array = [np.array([100.0, 200.0]) for _ in range(400)]
        phase_ref = 2 * np.pi * 0.2 * rt_array
        phase_shift = 0.7
        true_oscillation = 5.0 * np.sin(phase_ref + phase_shift)
        intensity_array = [np.array([0.0, 10.0 + true_oscillation[i]]) for i in range(400)]
        local_freqs_ref = np.full_like(rt_array, 0.2)

        _, modulated_signal, _ = correct_oscillations(
            rt_array,
            mz_array,
            intensity_array,
            phase_ref,
            local_freqs_ref,
            target_mz=200.0,
            rt_window=0.01,
            amplitude_method="q90",
        )

        corr = np.corrcoef(modulated_signal, true_oscillation)[0, 1]
        self.assertGreater(corr, 0.95)

if __name__ == "__main__":
    unittest.main()
