# -*- coding: utf-8 -*-
import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock


class TestCli(unittest.TestCase):

    def test_main_processes_single_file_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "input.mzML")
            with open(input_file, "w", encoding="utf-8"):
                pass

            with mock.patch.dict(sys.modules, {"pyopenms": mock.MagicMock()}):
                cli = importlib.import_module("sicritfix.cli")

                with mock.patch.object(cli, "process_file", return_value=True) as mock_process_file:
                    with mock.patch.object(sys, "argv", ["sicritfix", "--input", input_file]):
                        cli.main()

                    mock_process_file.assert_called_once_with(
                        file_path=input_file,
                        save_as=os.path.join(tmp_dir, "input_corrected.mzML"),
                        plot=False,
                        verbose=False,
                        mz_window=0.1,
                        rt_window=5,
                    )


if __name__ == "__main__":
    unittest.main()
