# -*- coding: utf-8 -*-

import csv
import os
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyopenms as oms

from sicritfix.utils.frequency_analyzer import obtain_freq_from_signal
from sicritfix.utils.intensity_analyzer import (
    build_xic,
    get_amplitude,
)


class TestAmplitudeMethodsComparison(unittest.TestCase):
    def test_compare_amplitude_methods_on_demo_data(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        demo_path = os.path.join(
            root_dir,
            "demo_data",
            "MSCONVERT_CTRL_103_01_c_afterreboot_original.mzML",
        )
        self.assertTrue(os.path.exists(demo_path), f"Demo file not found: {demo_path}")

        output_dir = os.path.join(root_dir, "output_jpg")
        os.makedirs(output_dir, exist_ok=True)

        exp = oms.MSExperiment()
        oms.MzMLFile().load(demo_path, exp)

        rts = []
        mz_array = []
        intensity_array = []
        for spectrum in exp:
            mzs, intensities = spectrum.get_peaks()
            rts.append(spectrum.getRT())
            mz_array.append(np.asarray(mzs, dtype=float))
            intensity_array.append(np.asarray(intensities, dtype=float))

        rts = np.asarray(rts, dtype=float)
        dt = float(np.mean(np.diff(rts)))

        # Keep frequency context aligned with production code.
        local_freqs_ref, _ = obtain_freq_from_signal(
            rts,
            mz_array,
            intensity_array,
            rt_window=5,
            mz_window=0.1,
        )

        # Use most intense peaks from first scan to select representative m/z values.
        first_mzs = mz_array[0]
        first_ints = intensity_array[0]
        top_idx = np.argsort(first_ints)[-12:]
        mz_candidates = np.unique(np.round(first_mzs[top_idx], 2))
        mz_candidates = np.sort(mz_candidates)

        methods = [
            "q75",
            "q90",
            "global_trimmed_detrended",
            "local_robust_detrended",
        ]

        metrics = {m: [] for m in methods}
        for mz in mz_candidates:
            xic = build_xic(
                mz_array,
                intensity_array,
                rts,
                target_mz=float(mz),
                rt_window=5,
                mz_window=0.1,
            )
            for method in methods:
                amp = get_amplitude(xic, local_freqs_ref, dt, method=method)
                metrics[method].append(float(amp))

        # Basic sanity assertions.
        for method in methods:
            vals = np.asarray(metrics[method], dtype=float)
            self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite amplitudes in {method}")
            self.assertTrue(np.any(vals > 0), f"No positive amplitudes found in {method}")

        # Save metrics CSV.
        csv_path = os.path.join(output_dir, "test_amplitude_methods_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mz"] + methods)
            for i, mz in enumerate(mz_candidates):
                writer.writerow([float(mz)] + [metrics[m][i] for m in methods])

        # Plot: amplitudes across m/z for each method.
        x = np.arange(len(mz_candidates), dtype=float)
        width = 0.2
        plt.figure(figsize=(14, 6))
        for j, method in enumerate(methods):
            plt.bar(x + (j - 1.5) * width, metrics[method], width=width, label=method)

        plt.xticks(x, [f"{mz:.2f}" for mz in mz_candidates], rotation=30, ha="right")
        plt.xlabel("m/z")
        plt.ylabel("Estimated amplitude")
        plt.title("Amplitude method comparison on demo mzML")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "test_amplitude_methods_comparison.jpg")
        plt.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close()

        self.assertTrue(os.path.exists(csv_path), f"Metrics CSV not created: {csv_path}")
        self.assertTrue(os.path.exists(plot_path), f"Comparison plot not created: {plot_path}")


if __name__ == "__main__":
    unittest.main()
