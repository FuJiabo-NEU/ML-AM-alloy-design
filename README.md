# UQ‑KGAT: Uncertainty‑Quantified Knowledge‑Graph Attention Network for Material Defect Prediction

https://img.shields.io/badge/python-3.9-blue.svg

https://img.shields.io/badge/PyTorch-1.13.1%2520%252B%2520CUDA%252011.7-ee4c2c.svg

https://img.shields.io/badge/License-Apache%25202.0-blue.svg

**UQ‑KGAT** is a Python package that integrates **graph attention networks (GATs) with physical‑metallurgy knowledge graphs** for robust defect prediction in additive manufacturing. It provides uncertainty quantification (epistemic and aleatoric) and supports multi‑objective alloy design optimisation via NSGA‑II.

## Overview
The framework consists of two main scripts:

`UQ‑KGAT model.py` – defines the GAT model, training routines, Monte Carlo dropout uncertainty estimation, and utility functions for data loading and graph construction.

`Sorting algorithm NSGA‑II.py` – loads pre‑trained surrogate models, chains them into an integrated prediction pipeline, and runs a multi‑objective genetic algorithm (NSGA‑II) to optimise alloy composition for minimal predicted defect area fraction and uncertainty.

The system was validated on Ni‑based superalloy and aluminium alloy datasets across L‑DED and LPBF processes, but is designed to be transferable to other alloy systems and AM modalities.

## System Requirements

## Hardware

A standard computer with sufficient RAM to hold the training data and graph structures. **GPU acceleration is required for training and optimisation** (CUDA‑capable NVIDIA GPU).

## Software

OS: Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10+;
Python: 3.9 (recommended, as tested);
PyTorch: 1.13.1 with CUDA 11.7 (pytorch-cuda=11.7);

Core dependencies:
numpy, scipy, pandas,
scikit‑learn,
geatpy (genetic algorithm framework),
matplotlib, seaborn,
joblib, openpyxl, csv (built‑in);

## Installation

## 1. Clone the repository

```
git clone https://github.com/your‑org/UQ‑KGAT.git
cd UQ‑KGAT
```

## 2. Create environment and install PyTorch (exact tested versions)

```
conda create -n uqkgat python=3.9
conda activate uqkgat
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch
```

## 3. Install remaining packages

```
pip install numpy scipy pandas scikit-learn geatpy matplotlib seaborn joblib openpyxl
```

## Quick Start

## 1. Data preparation

Place your dataset `AM_Data.xlsx` and the knowledge‑graph adjacency matrix `data_matrix.csv` in the project root directory. The spreadsheet must contain feature columns (composition + process parameters) and the target defect area fraction as the last column.

## 2: Train the UQ‑KGAT model

Run the training script:

```python "Codes/UQ-KGAT model.py"```

This will:

Normalise the input features.

Build graph datasets (GraphTrainSet, GraphTestSet).

Train the multi‑head GAT with the heteroscedastic loss.

Save the model checkpoint as Results/model state/gat_state.pth.

You can adjust hyperparameters (epochs, learning rate, dropout) directly in the script.

## 3: Train intermediate surrogate models

The NSGA‑II optimisation script expects pre‑trained models for coarsening rate, COMSOL outputs, volume energy, and hardness (CNN‑based). You must train these separately using your own data or the same dataset.

## 4: Run multi‑objective optimisation

```python "Codes/Sorting algorithm NSGA-II.py"```

This will:

Load all pre‑trained models.

Define 13 decision variables (alloy composition and process parameters) with given bounds.

Execute NSGA‑II for 1500 generations (population size 50).

Save Pareto‑optimal solutions and all intermediate predictions to Results/.

You can modify the variable bounds and algorithm parameters inside the MyProblem class and the main loop.

## Repository Structure

```
UQ‑KGAT/
├── Codes/
│   ├── UQ-KGAT model.py          # GAT training & uncertainty estimation
│   └── Sorting algorithm NSGA-II.py  # Multi‑objective optimisation pipeline
├── data_matrix.csv               # Adjacency matrix of the physical knowledge graph
├── AM_Data.xlsx                  # Example dataset (not included; request from authors)
├── Results/
│   ├── model state/
│   │   └── gat_state.pth         # Trained UQ‑KGAT weights
│   ├── GA_dAta/                  # Optimisation history
│   └── all targets *.csv         # Pareto front solutions
├── README.md
└── requirements.txt (optional)
```

## Usage Notes

Both scripts automatically detect CUDA and fall back to CPU if unavailable, but GPU is strongly recommended due to the Monte Carlo dropout sampling and genetic algorithm.

The GAT class returns both prediction mean and log‑variance; variance is learned via the loss function in Statistical_loss.

For pure inference (without labels), use GAT_Set instead of GraphTrainSet – this is already done in the optimisation script.

## License

This project is distributed under the **Apache License 2.0**. See the LICENSE file for details.
