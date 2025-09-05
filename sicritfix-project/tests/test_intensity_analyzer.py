# -*- coding: utf-8 -*-

"""
Unit tests for intensity_analyzer.py

@contents : Tests for FFT-based frequency analysis and polynomial smoothing.
@project  : SICRITfix – Oscillation Correction in Mass Spectrometry Data
@file     : test_intensity_analyzer.py
@author   : Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)
@version  : 0.0.1, 24 July 2025

@license  : GNU AFFERO GENERAL PUBLIC LICENSE v3
"""



import unittest
import numpy as np
from sicritfix.utils.intensity_analyzer import build_xic, get_amplitude


class TestIntensityAnalyzer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        self.n_scans = 100
        self.mz_points = 1000
        self.rt_array = np.linspace(0, 99, self.n_scans)
        self.target_mz = 922.098
        self.mz_tol = 0.2
        self.sampling_interval = np.mean(np.diff(self.rt_array))
        self.local_freqs = np.full(20, 0.2)  # constant freq (Hz)

        # Synthetic mz_array and intensity_array
        self.mz_array = []
        self.intensity_array = []
        for _ in range(self.n_scans):
            mzs = np.linspace(800, 1000, self.mz_points)
            intensities = np.random.normal(0, 0.5, self.mz_points)
            # Add a Gaussian peak around target_mz
            intensities += 100 * np.exp(-0.5 * ((mzs - self.target_mz) / 0.1) ** 2)
            self.mz_array.append(mzs)
            self.intensity_array.append(intensities)

    def test_build_xic_shape(self):
        xic = build_xic(self.mz_array, self.intensity_array, self.rt_array, self.target_mz, self.mz_tol)
        self.assertEqual(xic.shape, self.rt_array.shape)

    def test_build_xic_contains_signal(self):
        xic = build_xic(self.mz_array, self.intensity_array, self.rt_array, self.target_mz, self.mz_tol)
        self.assertTrue(np.any(xic > 0), "XIC should contain non-zero signal near the target m/z")

    def test_get_amplitude_positive(self):
        xic = build_xic(self.mz_array, self.intensity_array, self.rt_array, self.target_mz, self.mz_tol)
        amplitude = get_amplitude(self.target_mz, xic, self.rt_array, self.local_freqs, self.sampling_interval)
        self.assertGreater(amplitude, 0, "Amplitude should be greater than zero for meaningful signal")

    def test_get_amplitude_stability(self):
        xic = build_xic(self.mz_array, self.intensity_array, self.rt_array, self.target_mz, self.mz_tol)
        amp1 = get_amplitude(self.target_mz, xic, self.rt_array, self.local_freqs, self.sampling_interval)
        amp2 = get_amplitude(self.target_mz, xic, self.rt_array, self.local_freqs, self.sampling_interval)
        self.assertAlmostEqual(amp1, amp2, places=5, msg="Amplitude should be stable across identical input")


if __name__ == "__main__":
    unittest.main()
