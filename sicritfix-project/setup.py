# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="sicritfix",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "sicritfix = sicritfix.cli:main"
        ]
    },
    install_requires=[
        "numpy",
        "pandas",
        "massql",
        "pyopenms",
        "pyteomics",
    ],
)
