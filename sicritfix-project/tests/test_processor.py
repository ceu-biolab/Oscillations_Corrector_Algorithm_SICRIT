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
import tempfile
import os
from sicritfix.processing.processor import detect_oscillating_mzs, correct_spectra, process_file


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
        self.assertIn(round(self.osc_mz, 2), oscillating_mzs)

    def test_correct_spectra_structure(self):
        dummy_residuals = {
            round(self.osc_mz, 3): np.ones(len(self.rt_array)) * 0.5
        }
        corrected_map, exec_time = correct_spectra(
            self.input_map, [self.osc_mz], self.rt_array, dummy_residuals, mz_bin_size=0.01
        )
        self.assertIsInstance(corrected_map, oms.MSExperiment)
        self.assertEqual(corrected_map.getNrSpectra(), self.input_map.getNrSpectra())
        #self.assertTrue(exec_time > 0)
        
    def test_process_file_no_oscillations(self):
        """Ensure process_file returns original file if no oscillations are detected."""
        # To create an MSExperiment with no oscillations
        input_map = oms.MSExperiment()
        for rt in self.rt_array:
            spec = oms.MSSpectrum()
            spec.setRT(rt)
            spec.setMSLevel(1)
            spec.set_peaks((np.array([200.0, 300.0]), np.zeros(2)))
            input_map.addSpectrum(spec)

        # Save to a temporary mzML file
        tmp_dir = tempfile.mkdtemp()
        input_file = os.path.join(tmp_dir, "flat.mzML")
        output_file = os.path.join(tmp_dir, "flat_corrected.mzML")
        oms.MzMLFile().store(input_file, input_map)

        # Run process_file
        file_corrected=process_file(input_file, output_file)
        self.assertFalse(file_corrected, "Should return False when no oscillations are detected")


        # Reload output
        result_map = oms.MSExperiment()
        self.assertTrue(os.path.exists(output_file), "Output file was not created")

        oms.MzMLFile().load(output_file, result_map)

        # Assertions: same number of spectra, identical m/z and intensities
        self.assertEqual(result_map.getNrSpectra(), input_map.getNrSpectra())
        for spec_in, spec_out in zip(input_map, result_map):
            mz_in, intens_in = spec_in.get_peaks()
            mz_out, intens_out = spec_out.get_peaks()
            np.testing.assert_array_equal(mz_in, mz_out)
            np.testing.assert_array_equal(intens_in, intens_out)

if __name__ == "__main__":
    unittest.main()


