# -*- coding: utf-8 -*-
# src/sicritfix/main.py
import argparse
import os

def _percentile_argument(value):
    percentile = float(value)
    if percentile < 0 or percentile > 100:
        raise argparse.ArgumentTypeError("--amplitude_percentile must be between 0 and 100.")
    return percentile


def _multiplier_argument(value):
    multiplier = float(value)
    if multiplier < 0:
        raise argparse.ArgumentTypeError("--amplitude_multiplier must be greater than or equal to 0.")
    return multiplier


def _mz_argument(value):
    mz = float(value)
    if mz <= 0:
        raise argparse.ArgumentTypeError("--reference_mz must be greater than 0.")
    return mz


def _positive_int_argument(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0.")
    return int_value


def main():
    from sicritfix.processing.processor import process_file
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
        "--reference_mz",
        type=_mz_argument,
        default=922.098,
        help="Reference m/z used to estimate the oscillation frequency and phase."
    )

    parser.add_argument(
        "--window_scan_size",
        type=_positive_int_argument,
        default=70,
        help=(
            "Number of scans in each local FFT window used to estimate reference "
            "local frequencies and phase (default: 70)."
        ),
    )

    parser.add_argument(
        "--amplitude_method",
        type=str,
        default="local_robust_detrended",
        choices=["q75", "q90", "percentile", "global_trimmed_detrended", "local_robust_detrended", "all"],
        help=(
            "Amplitude estimation method. "
            "'q75': local IQR amplitudes summarized by 75th percentile (conservative). "
            "'q90': local IQR amplitudes summarized by 90th percentile (more aggressive). "
            "'percentile': local IQR amplitudes summarized by the percentile set with --amplitude_percentile. "
            "'global_trimmed_detrended': baseline-detrended global trimmed range estimator. "
            "'local_robust_detrended': baseline-detrended robust local-window estimator. "
            "'all': run the four fixed built-in methods and save one corrected mzML per method."
        ),
    )

    parser.add_argument(
        "--amplitude_percentile",
        type=_percentile_argument,
        default=75.0,
        help=(
            "Percentile used when --amplitude_method percentile is selected "
            "(for example 75 or 90)."
        ),
    )

    parser.add_argument(
        "--amplitude_multiplier",
        type=_multiplier_argument,
        default=1.0,
        help=(
            "Multiplication factor applied to the estimated amplitude for any method "
            "(for example 1.2 to increase correction strength by 20%%)."
        ),
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

    if args.amplitude_method == "all":
        methods_to_run = ["q75", "q90", "global_trimmed_detrended", "local_robust_detrended"]
    else:
        methods_to_run = [args.amplitude_method]

    for file_path in files_to_process:

        explicit_single_output = bool(args.output and len(files_to_process) == 1 and len(methods_to_run) == 1)

        if args.output and len(files_to_process) == 1:
            requested_base, requested_ext = os.path.splitext(args.output)
            if not requested_ext:
                requested_ext = ".mzML"
            base_output = requested_base
            output_ext = requested_ext
        else:
            file_base, file_ext = os.path.splitext(file_path)
            if os.path.isfile(input_path):
                base_output = file_base
                output_ext = file_ext if file_ext else ".mzML"
            else:
                base_output = file_base
                output_ext = ".mzML"

        for method in methods_to_run:
            method_label = method
            if method == "percentile":
                method_label = f"percentile{args.amplitude_percentile:g}"

            if explicit_single_output:
                output_path = args.output if os.path.splitext(args.output)[1] else f"{args.output}.mzML"
            elif len(methods_to_run) == 1:
                output_path = f"{base_output}_corrected{output_ext}"
            else:
                output_path = f"{base_output}_{method_label}_corrected{output_ext}"

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
                if method == "percentile":
                    print(f"  Amplitude method: {method} ({args.amplitude_percentile:g})")
                else:
                    print(f"  Amplitude method: {method}")
                print(f"  Amplitude multiplier: {args.amplitude_multiplier:g}")
                print(f"  Reference m/z: {args.reference_mz:g}")
                print(f"  Window scan size: {args.window_scan_size}")

            file_corrected = process_file(
                file_path=file_path,
                save_as=output_path,
                plot=args.plot,
                verbose=args.verbose,
                mz_window=args.mz_window,
                rt_window=args.rt_window,
                amplitude_method=method,
                amplitude_percentile=args.amplitude_percentile,
                amplitude_multiplier=args.amplitude_multiplier,
                reference_mz=args.reference_mz,
                window_scan_size=args.window_scan_size,
            )

            if file_corrected:
                print(f"Oscillations detected and corrected → {output_path}")
            else:
                print(f"No oscillations detected → {output_path}")

if __name__ == "__main__":
    main()
