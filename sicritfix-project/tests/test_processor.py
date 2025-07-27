# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This Python module contains unit tests for the processor.py module of the SICRITfix project.

@contents :  Unit tests for oscillation detection and correction logic.
@project  :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program  :  N/A
@file     :  test_processor.py
@version  :  0.0.1, 18 July 2025
@author   :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    The Zen of Python
      https://www.python.org/dev/peps/pep-0020/
    Style Guide for Python Code
      https://www.python.org/dev/peps/pep-0008/
    Example NumPy Style Python Docstrings
      http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""

__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"

import unittest
import numpy as np
import pyopenms as oms
from sicritfix.processing.processor import detect_oscillating_mzs, correct_spectra

class TestProcessor(unittest.TestCase):

    def setUp(self):
        self.rt_array = np.linspace(0, 10, 50)
        self.osc_mz = 100.123

        # Create artificial oscillating XIC
        osc_signal = np.sin(2 * np.pi * 3 * self.rt_array) + np.random.normal(0, 0.1, 50)
        flat_signal = np.zeros(50)

        self.mz_array = [np.array([self.osc_mz, 150.0]) for _ in range(50)]
        self.intensity_array = [np.array([osc_signal[i], flat_signal[i]]) for i in range(50)]

        self.input_map = oms.MSExperiment()
        for i, rt in enumerate(self.rt_array):
            spec = oms.MSSpectrum()
            spec.setRT(rt)
            spec.setMSLevel(1)
            spec.set_peaks((self.mz_array[i], self.intensity_array[i]))
            self.input_map.addSpectrum(spec)

    def test_detect_oscillating_mzs(self):
        _, oscillating_mzs, _ = detect_oscillating_mzs(
            self.rt_array, self.mz_array, self.intensity_array,
            mz_bin_size=0.01, min_occurrences=5, power_threshold=0.05
        )
        self.assertIn(round(self.osc_mz, 3), oscillating_mzs)

    def test_correct_spectra_structure(self):
        dummy_residuals = {
            round(self.osc_mz, 3): np.ones(len(self.rt_array)) * 0.5
        }
        corrected_map, exec_time = correct_spectra(
            self.input_map, [self.osc_mz], self.rt_array, dummy_residuals, mz_bin_size=0.01
        )
        self.assertIsInstance(corrected_map, oms.MSExperiment)
        self.assertEqual(len(corrected_map), len(self.input_map))
        self.assertTrue(exec_time > 0)

if __name__ == "__main__":
    unittest.main()


