#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare XIC traces between an original mzML file and a corrected mzML file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sicritfix-project" / "src"))

from sicritfix.utils.intensity_analyzer import build_xic


DEFAULT_TARGET_MASSES = {
    "L-Tryptophan": 205.09771,
    "L-Histidine": 156.07731,
    "Gly-Leu": 189.12392,
    "Methionine sulfone": 182.04871,
    "Another 121.050873": 121.050873,
    "Another 922.09798": 922.09798,
}


def parse_target_mz(values: list[str] | None) -> dict[str, float]:
    if not values:
        return DEFAULT_TARGET_MASSES.copy()

    targets = {}
    for value in values:
        if "=" in value:
            name, mz_text = value.split("=", 1)
            name = name.strip()
            mz = float(mz_text.strip())
            if not name:
                name = f"m/z {mz:g}"
        else:
            mz = float(value)
            name = f"m/z {mz:g}"
        targets[name] = mz
    return targets


def load_arrays(file_path: str):
    from sicritfix.io.io import load_file

    exp = load_file(file_path)
    rt_array = np.array([spectrum.getRT() for spectrum in exp], dtype=float)
    mz_array = []
    intensity_array = []

    for spectrum in exp:
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(np.asarray(mzs, dtype=float))
        intensity_array.append(np.asarray(intensities, dtype=float))

    return rt_array, mz_array, intensity_array


def plot_xic_comparison(
    original_file: str,
    corrected_file: str,
    targets: dict[str, float],
    output: str,
    mz_window: float,
    rt_window: float,
    show: bool,
):
    original_rt, original_mzs, original_intensities = load_arrays(original_file)
    corrected_rt, corrected_mzs, corrected_intensities = load_arrays(corrected_file)

    n_targets = len(targets)
    fig_height = max(3.0, 2.4 * n_targets)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, fig_height), sharex=False)
    if n_targets == 1:
        axes = [axes]

    for ax, (target_name, target_mz) in zip(axes, targets.items()):
        original_xic = build_xic(
            original_mzs,
            original_intensities,
            original_rt,
            target_mz=target_mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )
        corrected_xic = build_xic(
            corrected_mzs,
            corrected_intensities,
            corrected_rt,
            target_mz=target_mz,
            rt_window=rt_window,
            mz_window=mz_window,
        )

        ax.plot(original_rt, original_xic, color="black", linewidth=1.0, label="Original")
        ax.plot(corrected_rt, corrected_xic, color="#2563eb", linewidth=1.0, label="Corrected")
        ax.set_title(f"{target_name} ({target_mz:.5f} m/z)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Retention time (s)")
    fig.suptitle(
        f"XIC comparison | m/z window +/- {mz_window:g} Da | RT window {rt_window:g} s",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved XIC comparison plot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay XICs from an original mzML and a corrected mzML for selected m/z values."
    )
    parser.add_argument("--original", required=True, help="Path to the original mzML file.")
    parser.add_argument("--corrected", required=True, help="Path to the corrected mzML file.")
    parser.add_argument(
        "--mz",
        action="append",
        help=(
            "Target m/z to plot. Use multiple times. Accepts either 'Name=205.09771' "
            "or a bare value like '205.09771'. If omitted, the built-in target masses are used."
        ),
    )
    parser.add_argument("--mz_window", type=float, default=0.1, help="m/z tolerance in Da.")
    parser.add_argument(
        "--rt_window",
        type=float,
        default=0.01,
        help="RT smoothing window in seconds. The default 0.01 keeps point-by-point XICs.",
    )
    parser.add_argument(
        "--output",
        default="xic_original_vs_corrected.png",
        help="Output image path.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window.")

    args = parser.parse_args()
    targets = parse_target_mz(args.mz)

    plot_xic_comparison(
        original_file=args.original,
        corrected_file=args.corrected,
        targets=targets,
        output=args.output,
        mz_window=args.mz_window,
        rt_window=args.rt_window,
        show=args.show,
    )


if __name__ == "__main__":
    main()
