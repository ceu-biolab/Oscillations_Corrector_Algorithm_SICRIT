# -*- coding: utf-8 -*-
# src/sicritfix/main.py
import argparse
import os
from sicritfix.processing.processor import process_file

def main():
    parser = argparse.ArgumentParser(
        description="Correct oscillations in an mzML/mzXML file and output the corrected mzML."
    )

    parser.add_argument(
        "--input", required=True,
        help="Path to input mzML/mzXML file to process or Path to a folder containing multiple mzML/mzXML files to process"
        )

    parser.add_argument(
        "--output",
        help="(Optional) Path to output corrected mzML file. "
             "If not provided, '_corrected.mzML' will be added to the input filename."
    )

    
    parser.add_argument(
        "--mz_window", type=float, default=0.1,
        help="MZ window to calculate the different amplitude in each mz window in Da (e.g., 0.1)"
    )

    parser.add_argument(
        "--rt_window", type=float, default=5,
        help="RT window to calculate the frequency of the oscillations in seconds. "
    )
    
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plots for corrected signals"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f" Input file not found: {args.input}")
        return

    input_path = args.input

    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return

    files_to_process = []

    # Case 1: single file
    if os.path.isfile(input_path):

        if not input_path.lower().endswith((".mzml", ".mzxml")):
            print("Input file must be .mzML or .mzXML")
            return

        files_to_process = [input_path]

    # Case 2: directory
    elif os.path.isdir(input_path):

        files_to_process = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".mzml", ".mzxml"))
        ]

        if not files_to_process:
            print("No mzML/mzXML files found in directory.")
            return

    else:
        print("Input path must be a file or directory.")
        return

    if args.verbose:
        print(" Starting processing")
        print(f" Input: {args.input}")

    if args.plot:
        print(" Plotting is ENABLED")

    for file_path in files_to_process:

        if args.output and len(files_to_process) == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(file_path)
            if os.path.isfile(input_path):
                output_path = base + "_corrected" + ext
            else:
                output_path = base + "_corrected.mzML"

        if os.path.exists(output_path) and not args.overwrite:
            print(f"Output exists: {output_path}")
            print("Use --overwrite to replace it.")
            continue

        if os.path.exists(output_path) and args.overwrite:
            print(f" Removing existing file: {output_path}")
            os.remove(output_path)

        if args.verbose:
            print("Processing file:")
            print(f"  Input : {file_path}")
            print(f"  Output: {output_path}")

        file_corrected = process_file(
            file_path=file_path,
            save_as=output_path,
            plot=args.plot,
            verbose=args.verbose,
            mz_window=args.mz_window,
            rt_window=args.rt_window,
        )

        if file_corrected:
            print(f"Oscillations detected and corrected → {output_path}")
        else:
            print(f"No oscillations detected → {output_path}")

if __name__ == "__main__":
    main()
