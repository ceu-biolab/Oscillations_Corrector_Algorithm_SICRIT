# -*- coding: utf-8 -*-
import importlib
import os
import sys
import tempfile
import unittest
import types
from unittest import mock


class TestCli(unittest.TestCase):

    def test_main_processes_single_file_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            fake_processor = types.SimpleNamespace(process_file=mock.MagicMock(return_value=True))
            with mock.patch.dict(
                sys.modules,
                {"pyopenms": mock.MagicMock(), "sicritfix.processing.processor": fake_processor},
            ):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(sys, "argv", ["sicritfix", "--input", input_file]):
                    cli.main()

                fake_processor.process_file.assert_called_once_with(
                    file_path=input_file,
                    save_as=os.path.join(tmp_dir, "input_corrected.mzML"),
                    plot=False,
                    verbose=False,
                    mz_window=0.1,
                    rt_window=5,
                    amplitude_method="local_robust_detrended",
                    amplitude_percentile=75.0,
                    amplitude_multiplier=1.0,
                    reference_mz=922.098,
                    window_scan_size=70,
                )

    def test_main_processes_all_amplitude_methods(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            fake_processor = types.SimpleNamespace(process_file=mock.MagicMock(return_value=True))
            with mock.patch.dict(
                sys.modules,
                {"pyopenms": mock.MagicMock(), "sicritfix.processing.processor": fake_processor},
            ):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(
                    sys,
                    "argv",
                    ["sicritfix", "--input", input_file, "--amplitude_method", "all"],
                ):
                    cli.main()

                self.assertEqual(fake_processor.process_file.call_count, 4)
                called_methods = [call.kwargs["amplitude_method"] for call in fake_processor.process_file.call_args_list]
                self.assertEqual(
                    called_methods,
                    ["q75", "q90", "global_trimmed_detrended", "local_robust_detrended"],
                )

    def test_main_processes_percentile_99_method(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            fake_processor = types.SimpleNamespace(process_file=mock.MagicMock(return_value=True))
            with mock.patch.dict(
                sys.modules,
                {"pyopenms": mock.MagicMock(), "sicritfix.processing.processor": fake_processor},
            ):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "sicritfix",
                        "--input",
                        input_file,
                        "--amplitude_method",
                        "percentile",
                        "--amplitude_percentile",
                        "99",
                    ],
                ):
                    cli.main()

                fake_processor.process_file.assert_called_once_with(
                    file_path=input_file,
                    save_as=os.path.join(tmp_dir, "input_corrected.mzML"),
                    plot=False,
                    verbose=False,
                    mz_window=0.1,
                    rt_window=5,
                    amplitude_method="percentile",
                    amplitude_percentile=99.0,
                    amplitude_multiplier=1.0,
                    reference_mz=922.098,
                    window_scan_size=70,
                )

    def test_main_processes_amplitude_multiplier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            fake_processor = types.SimpleNamespace(process_file=mock.MagicMock(return_value=True))
            with mock.patch.dict(
                sys.modules,
                {"pyopenms": mock.MagicMock(), "sicritfix.processing.processor": fake_processor},
            ):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "sicritfix",
                        "--input",
                        input_file,
                        "--amplitude_method",
                        "percentile",
                        "--amplitude_percentile",
                        "99",
                        "--amplitude_multiplier",
                        "1.5",
                    ],
                ):
                    cli.main()

                fake_processor.process_file.assert_called_once_with(
                    file_path=input_file,
                    save_as=os.path.join(tmp_dir, "input_corrected.mzML"),
                    plot=False,
                    verbose=False,
                    mz_window=0.1,
                    rt_window=5,
                    amplitude_method="percentile",
                    amplitude_percentile=99.0,
                    amplitude_multiplier=1.5,
                    reference_mz=922.098,
                    window_scan_size=70,
                )

    def test_main_processes_custom_reference_mz(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            fake_processor = types.SimpleNamespace(process_file=mock.MagicMock(return_value=True))
            with mock.patch.dict(
                sys.modules,
                {"pyopenms": mock.MagicMock(), "sicritfix.processing.processor": fake_processor},
            ):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "sicritfix",
                        "--input",
                        input_file,
                        "--reference_mz",
                        "121.050873",
                    ],
                ):
                    cli.main()

                fake_processor.process_file.assert_called_once_with(
                    file_path=input_file,
                    save_as=os.path.join(tmp_dir, "input_corrected.mzML"),
                    plot=False,
                    verbose=False,
                    mz_window=0.1,
                    rt_window=5,
                    amplitude_method="local_robust_detrended",
                    amplitude_percentile=75.0,
                    amplitude_multiplier=1.0,
                    reference_mz=121.050873,
                    window_scan_size=70,
                )


if __name__ == "__main__":
    unittest.main()
