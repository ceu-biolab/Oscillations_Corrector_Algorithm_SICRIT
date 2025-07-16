# io/io_utils.py
import os
import subprocess

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

