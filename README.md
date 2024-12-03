# BCNN-SCP

Implementation for "Bayesian Machine Learning 2024" course project titled "Bayesian Convolution Neural Network with Spatially Correlated Priors"

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [License](#license)

## Project Structure

```bash
BCNN-SCP/
├── configs/                      # Configs for the project
├── notebooks/                    # Jupyter notebooks for running experiments and analysis
│   └── bcnn.baseline.ipynb       # Baseline experiment notebook for initial testing and validation
├── src/                          # Source code for the project
│   ├── models/                   # Contains model-related modules
│   │   ├── __init__.py           # Makes the 'models' directory a Python package
│   │   ├── kernels.py            # Contains kernel functions for the model architecture
│   │   ├── layers.py             # Custom layers for the deep learning models
│   │   ├── losses.py             # Custom loss functions for training the models
│   │   └── networks.py           # Definitions of neural network architectures
│   │── __init__.py               # Makes the 'src' directory a Python package
│   │── main.py                   # Main script to run the project (training, evaluation, etc.)
│   └── utils.py                  # Utility functions for common tasks (e.g., data preprocessing, logging)
├── requirements.txt              # List of Python dependencies required for the project
├── README.md                     # Documentation describing the project, its purpose, and how to use it
└── LICENSE                       # Licensing details for the project code

```

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd BCNN-SCP/
python -m src.main configs/<config_name>.yaml
```
