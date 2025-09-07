# Oscillations_Corrector_Algorithm_SICRIT
Suppressing Signal Artifacts in CE-SICRIT-MS via Oscillation Processing

This repository contains a Python tool to **detect and correct oscillations** that appear in mass spectrometry (MS) spectra acquired with the SICRIT ionization source.
The program reads .mzXML or .mzML files, identifies oscillatory m/z signals via FFT analysis, corrects their intensities, and outputs cleaned .mzML files.

The tool includes a **command-line interface (CLI)** to process single files or whole folders.

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing a Single File](#processing-a-single-file)
  - [Processing a Folder of Files](#processing-a-folder-of-files)
  - [Command-Line Options](#command-line-options)
- [Detection and Correction Workflow](#detection-and-correction-workflow)
- [Examples](#examples)
- [License](#license)

---

## Introduction
This Python module provides a pipeline for correcting oscillatory artifacts in CE-SICRIT-MS spectra.  
It includes:
- Automatic loading of `.mzML` and `.mzXML` files.  
- Conversion of `.mzXML` to `.mzML` using [ProteoWizard MSConvert](https://proteowizard.sourceforge.io/downloads.shtml).  
- Detection of oscillatory m/z values using FFT-based power analysis.  
- Correction of intensity oscillations via residual signal reconstruction.  
- Saving corrected spectra in `.mzML` format.

---

## Installation
Clone the repository and install dependencies inside a conda environment:

```bash
git clone https://github.com/ceu-biolab/Oscillations_Corrector_Algorithm_SICRIT.git
cd Oscillations_Corrector_Algorithm_SICRIT

conda create -n sicritfix python=3.12
conda activate sicritfix
pip install -e .
```
---

## Usage
You can run the program from the command line after installation.  
Both single files and folders are supported.

### Processing a Single File
Run the tool with an input file:

```bash
python sicritfix.py input_file.mzXML --output corrected.mzML --overwrite
```
If no --output is provided, the tool will automatically generate a filename by appending _corrected.mzML.

### Processing a Folder of Files
To process an entire folder of `.mzXML` files, simply pass the folder path as input.  
Each file will be automatically converted to `.mzML`, processed, and saved.

```bash
python CLI.py /path/to/folder --overwrite --verbose
```
During execution you’ll see messages like:

```bash
file: sample1.mzXML loaded correctly
file: sample2.mzXML loaded correctly
```
### Command-Line Options
- `--output` : Output path (optional).  
- `--overwrite` : Overwrite existing files.  
- `--plot` : Show diagnostic plots.  
- `--verbose` : Print detailed execution logs.  
- `--mz_window` : m/z bin size for oscillation detection (default: `0.01`).  
- `--rt_window` : Retention time window for XIC smoothing (default: `0.01`).  

---

## Detection and Correction Workflow
1. Load the input file(s).  
   - If `.mzXML` is detected, it is converted to `.mzML`.  

2. Extract relevant MS data.  
   - Retention times (RT), m/z values, and intensities.  

3. Detect oscillating m/z values.  
   - Uses FFT-based power spectrum analysis.  

4. Correct oscillations.  
   - Subtracts a modeled sinusoidal component to recover residual signal.  

5. Update spectra.  
   - Replace oscillating intensities with corrected values.  

6. Save results.  
   - Corrected file is written to disk in `.mzML` format.

---

## Examples
Process a single file:
```bash
sicritfix data/sample.mzXML --plot
```
or
```bash
python CLI.py data/sample.mzXML --plot
```
Process all files in a folder:
```bash
sicritfix data/sample.mzXML --overwrite
```
or
```bash
python CLI.py data/sample.mzXML --overwrite
```
Verbose mode:
```bash
sicritfix data/sample.mzXML --verbose
```
or
```bash
python CLI.py data/sample.mzML --verbose
```
---

## License
