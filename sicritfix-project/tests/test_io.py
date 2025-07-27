# -*- coding: utf-8 -*-

#!/usr/bin/env python

"""
This Python module contains unit tests for the module `io_utils`.

@contents  :  Unit tests for functions in io_utils.py
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  test_io.py
@version   :  0.0.1, 24 July 2025
@autor     :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

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
import os
import subprocess
import tempfile
import shutil
import pyopenms as oms
from unittest import mock
from sicritfix.io.io import load_file, convert_mzxml_2_mzml


class TestIOUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_mzml = os.path.join(self.temp_dir, "test.mzML")

        # Create a dummy mzML file
        exp = oms.MSExperiment()
        oms.MzMLFile().store(self.fake_mzml, exp)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_mzml_file_successfully(self):
        result = load_file(self.fake_mzml)
        self.assertIsInstance(result, oms.MSExperiment)
        self.assertEqual(len(result.getSpectra()), 0)  # Dummy file is empty

    @mock.patch("subprocess.run")
    @mock.patch("os.path.exists")
    def test_convert_mzxml_2_mzml_success(self, mock_exists, mock_run):
        fake_input = os.path.join(self.temp_dir, "file.mzXML")
        expected_output = os.path.join(self.temp_dir, "file.mzML")

        # Simulate msconvert command success
        mock_run.return_value = None
        mock_exists.side_effect = lambda path: path == expected_output

        result = convert_mzxml_2_mzml(fake_input)
        self.assertEqual(result, expected_output)

    @mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "msconvert"))
    def test_convert_mzxml_2_mzml_failure(self, mock_run):
        with self.assertRaises(RuntimeError):
            convert_mzxml_2_mzml("fake.mzXML")

    @mock.patch("sicritfix.io.io.convert_mzxml_2_mzml")
    @mock.patch("os.path.exists", return_value=True)
    #@mock.patch("pyopenms.MzMLFile.load")
    def test_load_file_with_conversion(self, mock_exists, mock_convert):
        # Setup mocks
        mock_convert.return_value = self.fake_mzml
        #mock_load.return_value = None  # No return, it fills input_map in-place

        # Call function
        result = load_file("some_file.mzXML")

        self.assertIsInstance(result, oms.MSExperiment)
        mock_convert.assert_called_once()
        #mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
