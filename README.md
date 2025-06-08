# Tumor Growth Modeling with ODEs and Machine Learning

## Project Overview

This project combines mathematical biology and machine learning to model and predict tumor growth dynamics. Inspired by models used in oncology and biophysics, we simulate tumor growth using ordinary differential equations (ODEs), then train regression-based machine learning models to forecast tumor volume from temporal data.

## Motivation

Mathematical models like the Gompertz and logistic equations are widely used in biomedical research to capture tumor growth dynamics under various treatment conditions. This project explores how machine learning models compare to—and complement—traditional modeling approaches.

This project was developed as part of a portfolio to demonstrate applied mathematical modeling skills and to support graduate school applications in computational mathematics and data science.

## Tools & Concepts Used

- **ODE-based modeling:** Logistic and Gompertz growth models
- **Dimensional analysis:** Nondimensionalizing growth equations
- **Numerical simulation:** `scipy.integrate.solve_ivp`
- **ML Regression Models:** Linear Regression, Random Forest, XGBoost, and LSTM
- **Data Generation:** Synthetic datasets and optional real data (e.g. TCIA)
- **Comparison metrics:** MSE, MAE, R² score

## Structure

```
tumor-growth-modeling-ml/
├── data/                       # 📦 Datasets
│   ├── real/                  # → Placeholder for real tumor datasets
│   └── synthetic/             # → Synthetic data generated from ODEs
│
├── notebooks/                 # 📓 Jupyter Notebooks
│   ├── 01_ode_modeling.ipynb         # → Derive & simulate ODEs (logistic, Gompertz)
│   ├── 02_ml_regression.ipynb        # → ML model trained on tumor data
│   └── 03_analysis_and_plots.ipynb   # → Results, visualizations, and comparison
│
├── src/                      # 🧠 Core Python modules
│   ├── ode_models.py         # → ODEs and simulation functions
│   ├── generate_data.py      # → Utilities for generating synthetic tumor data
│   └── ml_models.py          # → ML model definitions and training functions
│
├── results/                  # 📊 Output plots and analysis
│   └── figures/              # → Visualizations (Matplotlib PNGs, PDFs)
│
├── tests/                    # 🧪 Optional: Unit tests for model code
│   └── test_ode_models.py
│
├── report/                   # 📄 Optional: Final writeup or PDF summary
│   ├── tumor_modeling_report.pdf
│   └── tumor_modeling_report.tex
│
├── README.md                 # 📘 Project overview and setup
├── requirements.txt          # 📦 Python dependencies
├── .gitignore                # 🚫 Ignored files (e.g., `.ipynb_checkpoints`)
└── LICENSE                   # 📜 MIT or other license
```

## Getting Started

This lab requires installation of the following libraries:

- For Windows Powershell Users:
```bash
pip install numpy scipy pandas matplotlib
```

- For Anaconda Powershell Users:
```bash
conda install numpy scipy pandas matplotlib
```

## Credits

Credits
Created by Troy Chin — CS major, math & data science minor at California State University, Fullerton. This project reflects interdisciplinary research interests in mathematical modeling, machine learning, and biological systems.

## GNU License
[GNU LICENSE](https://github.com/troyc03/CSUF-Tumor-Growth/blob/main/LICENSE)
