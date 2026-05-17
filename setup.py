"""Setup script for Monte Carlo Simulation Engine."""
from setuptools import setup, find_packages

setup(
    name="montecarlo-engine",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.9",
)
