UQ‑KGAT: Uncertainty‑Quantified Knowledge‑Graph Attention Network for Material Defect Prediction
https://img.shields.io/badge/python-3.9-blue.svg
https://img.shields.io/badge/PyTorch-1.13.1%2520%252B%2520CUDA%252011.7-ee4c2c.svg
https://img.shields.io/badge/License-Apache%25202.0-blue.svg

UQ‑KGAT is a Python package that integrates graph attention networks (GATs) with physical‑metallurgy knowledge graphs for robust defect prediction in additive manufacturing. It provides uncertainty quantification (epistemic and aleatoric) and supports multi‑objective alloy design optimisation via NSGA‑II.

Overview
The framework consists of two main scripts:

UQ‑KGAT model.py – defines the GAT model, training routines, Monte Carlo dropout uncertainty estimation, and utility functions for data loading and graph construction.

Sorting algorithm NSGA‑II.py – loads pre‑trained surrogate models, chains them into an integrated prediction pipeline, and runs a multi‑objective genetic algorithm (NSGA‑II) to optimise alloy composition for minimal predicted defect area fraction and uncertainty.

The system was validated on Ni‑based superalloy and aluminium alloy datasets across L‑DED and LPBF processes, but is designed to be transferable to other alloy systems and AM modalities.

System Requirements
Hardware
A standard computer with sufficient RAM to hold the training data and graph structures. GPU acceleration is required for training and optimisation (CUDA‑capable NVIDIA GPU).

Software
OS: Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10+
Python: 3.9 (recommended, as tested)
PyTorch: 1.13.1 with CUDA 11.7 (pytorch-cuda=11.7)
Core dependencies:
numpy, scipy, pandas
scikit‑learn
geatpy (genetic algorithm framework)
matplotlib, seaborn
joblib, openpyxl, csv (built‑in)

