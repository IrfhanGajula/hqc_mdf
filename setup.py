from setuptools import setup, find_packages

setup(
    name="hqc_mdf",
    version="1.0.0",
    description="Hybrid Quantum Medical Diagnostics Framework",
    author="Siddhartha Gummadi",
    packages=find_packages(),
    install_requires=[
        "pennylane",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "joblib",
        "fpdf2"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hqc-mdf=hqc_mdf.cli:launch_runner",
        ],
    },
)
