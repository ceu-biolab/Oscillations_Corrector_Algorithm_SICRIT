# io/io_utils.py

#!/usr/bin/env python

"""
This Python module provides utilities for reading and converting mass spectrometry files,
specifically mzML and mzXML formats, using OpenMS and ProteoWizard tools.

@contents  :  File I/O utilities for mzML and mzXML mass spectrometry data.
@project   :  SICRITfix – Oscillation Correction in Mass Spectrometry Data
@program   :  N/A
@file      :  io_utils.py
@version   :  0.0.1, 18 July 2025
@author    :  Maite Gómez del Rio Vinuesa (maite.gomezriovinuesa@gmail.com)

@information :
    The Zen of Python
        https://www.python.org/dev/peps/pep-0020/
    Style Guide for Python Code
        https://www.python.org/dev/peps/pep-0008/
    Example NumPy Style Python Docstrings
        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

@dependencies :
    - pyopenms
    - ProteoWizard (msconvert) must be installed and accessible in system PATH

@raises :
    - RuntimeError if file conversion or loading fails

@copyright :
    Copyright 2025 GNU AFFERO GENERAL PUBLIC LICENSE.
    All rights reserved. Reproduction in whole or in part is prohibited
    without the written consent of the copyright owner.
"""
__author__    = "Maite Gómez del Rio Vinuesa"
__copyright__ = "GPL License version 3"



import os
import subprocess
import time
import pyopenms as oms

def load_file(file_path):
    
    """
    Loads a mass spectrometry file in mzML or mzXML format.

    If the provided file is in mzXML format, it is automatically converted to mzML
    before being loaded. The mzML file is then loaded using OpenMS's MSExperiment class.

    Args:
        file_path (str): Path to the mzML or mzXML file.

    Returns:
        oms.MSExperiment: An object representing the loaded mass spectrometry experiment,
        ready for further processing.

    Raises:
        RuntimeError: If the converted mzML file is not found after conversion.
    """
    
    input_map = oms.MSExperiment()
    
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".mzxml":
        mzml_file_path = convert_mzxml_2_mzml(file_path)
        print("Converting to mzML...")
        time.sleep(3)
        if not os.path.exists(mzml_file_path):
            raise RuntimeError("Error: file not found")
        else:
            print("Loading mzML file...")
            oms.MzMLFile().load(mzml_file_path, input_map)
            print(f"File: {os.path.basename(file_path)} loaded correctly")
    else:
        oms.MzMLFile().load(file_path, input_map)
    
    return input_map
    
def convert_mzxml_2_mzml(file_path):
    """
    Converts a .mzXML file to .mzML format using ProteoWizard's msconvert tool.

    Parameters
    ----------
    file_path : str
        Path to the input .mzXML file.

    Returns
    -------
    str
        Path to the newly converted .mzML file.

    Raises
    ------
    RuntimeError
        If msconvert fails or the expected output file is not generated.
    """
    output_folder = os.path.dirname(file_path)
    mzml_file_path = os.path.join(
        output_folder, os.path.basename(file_path).replace(".mzXML", ".mzML")
    )

    try:
        subprocess.run([
            "msconvert", file_path, "--mzML", "--outdir", output_folder,
            "--64", "--zlib", "--filter", "peakPicking true 1-"
        ], check=True)

        if os.path.exists(mzml_file_path):
            return mzml_file_path
        else:
            raise RuntimeError(f"MSConvert did not generate the expected file: {mzml_file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error while running MSConvert: {e}")
        raise RuntimeError("Conversion with ProteoWizard (msconvert) failed.")

