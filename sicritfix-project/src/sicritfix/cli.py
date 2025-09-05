# -*- coding: utf-8 -*-
# src/sicritfix/main.py
import argparse
import os
from sicritfix.processing.processor import process_file

def main():
    parser = argparse.ArgumentParser(
        description="Correct oscillations in an mzML/mzXML file and output the corrected mzML."
    )

    parser.add_argument("input", help="Path to input mzML file")

    parser.add_argument(
        "output",
        nargs="?",
        help="(Optional) Path to output corrected mzML file. "
             "If not provided, '_corrected.mzML' will be added to the input filename."
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

    # Auto-generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = base + "_corrected" + ext

    if os.path.exists(output_path) and not args.overwrite:
        print(f" Output file exists: {output_path}")
        print(" Use --overwrite to allow replacing it.")
        return

    if os.path.exists(output_path) and args.overwrite:
        print(f" Removing existing file: {output_path}")
        os.remove(output_path)

    if args.verbose:
        print(" Starting processing")
        print(f" Input: {args.input}")
        print(f" Output: {output_path}")
        
    if args.plot:
        print(" Plotting is ENABLED")

    # Run the processing function
    file_corrected=process_file(
        file_path=args.input,
        save_as=output_path,
        plot=args.plot,
        verbose=args.verbose,
    )
    
    if file_corrected:
        print(f" Oscillations were detected and corrected. Corrected file saved to: {output_path}")
    else:
        print(f" No oscillations detected. Original file saved to: {output_path}")