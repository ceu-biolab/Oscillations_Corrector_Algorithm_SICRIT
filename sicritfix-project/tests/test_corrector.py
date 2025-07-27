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
            rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz=200.0
        )

        self.assertEqual(len(xic), len(rt_array))
        self.assertEqual(len(modulated_signal), len(rt_array))
        self.assertEqual(len(residual_signal), len(rt_array))

        # Check signal shape
        self.assertTrue(np.allclose(xic, modulated_signal + residual_signal, rtol=1e-4))

if __name__ == "__main__":
    unittest.main()

