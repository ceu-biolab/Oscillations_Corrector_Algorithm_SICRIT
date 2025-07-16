# Oscillations_Corrector_Algorithm_SICRIT
SUPPRESSING SIGNAL ARTIFACTS IN CE-SICRIT-MS VIA OSCILLATION PROCESSING


This repository contains an algorithm designed to **correct oscillations that appear in mass spectrometry (MS) spectra** after applying the **SICRIT ionization source**.

The tool is implemented in Python and can be executed from the **command-line interface (CLI)** after installation.

---

## 🚀 Key Features

- Automatic correction of oscillations in post-SICRIT MS data.
- Ready to be used as a command-line tool.
- Modular and extensible codebase.
- Easy to integrate into analytical data processing pipelines.

---

## 📦 Installation

### Optional: Using Conda environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd SICRIT_Oscillations_corrector_algorithm
conda env create -f conda_env.yml
conda activate oscilations_env

### Execution- CLI usage
sicritfix --input path/to/input_file.mzML --output path/to/output_folder/
###For full CLI options: sicritfix --help
