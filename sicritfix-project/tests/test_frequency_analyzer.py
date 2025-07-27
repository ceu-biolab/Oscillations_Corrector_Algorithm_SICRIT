# -*- coding: utf-8 -*-

#!/usr/bin/env python

"""
Unit tests for frequency_analyzer.py

@contents : Tests for FFT-based frequency analysis and polynomial smoothing.
@project  : SICRITfix – Oscillation Correction in Mass Spectrometry Data
@file     : test_frequency_analyzer.py
@author   : Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)
@version  : 0.0.1, 24 July 2025

@license  : GNU AFFERO GENERAL PUBLIC LICENSE v3
"""

import os
import sys
import unittest
import numpy as np

# Ensure correct import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sicritfix.utils.frequency_analyzer import (
    calculate_freq,
    local_frequencies_with_fft,
    apply_polynomial_regression,
)

class TestFrequencyAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic sinusoidal signal (frequency = 0.2 Hz)
        self.sampling_interval = 1.0
        self.time = np.arange(0, 100, self.sampling_interval)
        self.true_freq = 0.2  # Hz
        self.signal = np.sin(2 * np.pi * self.true_freq * self.time)
    
    def test_calculate_freq(self):
        fft_freqs, fft_magnitude, main_freq = calculate_freq(self.signal, self.sampling_interval)
        self.assertAlmostEqual(main_freq, self.true_freq, delta=0.01)

    def test_local_frequencies_with_fft(self):
        window_size = 20
        rts = self.time
        times, freqs = local_frequencies_with_fft(self.signal, rts, window_size, self.sampling_interval)
        self.assertEqual(len(times), len(freqs))
        self.assertTrue(np.allclose(freqs, self.true_freq, atol=0.05))

    def test_apply_polynomial_regression(self):
        rts = self.time
        step = 10
        rt_freqs = rts[::step]
        local_freqs = np.full_like(rt_freqs, fill_value=self.true_freq)
        phase = apply_polynomial_regression(rts, rt_freqs, local_freqs, freq_deg=2)

        # Assert increasing phase trend
        self.assertEqual(phase.shape, rts.shape)
        self.assertTrue(np.all(np.diff(phase) >= 0))

if __name__ == "__main__":
    unittest.main()
