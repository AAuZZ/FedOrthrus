# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype

## Project Overview

FedOrthrus is a federated learning framework that addresses cross-domain challenges through dual-prototype architecture. It enables effective model training across different data distributions while maintaining privacy and reducing communication overhead.

## Project Structure

```
FedOrthrus/
├── data/                 # Dataset directory
│   ├── digit/            # Digit datasets (MNIST, MNIST_M, SVHN, etc.)
│   │   ├── MNIST/        # MNIST dataset
│   │   ├── MNIST_M/      # MNIST-M dataset
│   │   ├── SVHN/         # SVHN dataset
│   │   ├── SynthDigits/  # SynthDigits dataset
│   │   └── USPS/         # USPS dataset
│   ├── office/           # Office datasets (amazon, caltech, dslr, webcam)
│   └── domain/           # Domain adaptation datasets (download required)
├── models/               # Model definitions
│   └── resnet.py         # ResNet10 model implementation
├── utils/                # Utility functions
│   ├── options.py        # Command line arguments configuration
│   ├── update.py         # Local training and testing implementation
│   ├── util.py           # Prototype aggregation, clustering and other utilities
│   ├── data_util.py      # Data preparation and preprocessing
│   ├── dataset.py        # Dataset processing utilities
│   └── finch.py          # FINCH clustering algorithm implementation
├── main.py               # Main entry point of the project
├── acc.npy               # Experimental results storage file
└── LICENSE               # License file
```

## Environment Setup

### Create Virtual Environment

```bash
# Create virtual environment with conda
conda create -n fedorthrus python=3.9 -y
conda activate fedorthrus

# Or create virtual environment with venv
python -m venv fedorthrus_venv
# Activate on Windows
fedorthrus_venv\Scripts\activate
# Activate on Linux/Mac
source fedorthrus_venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Datasets

FedOrthrus has been evaluated on three datasets:

### Digit
Digit datasets (MNIST, MNIST-M, SVHN, SynthDigits, USPS) are included in the repository.

### Office
Office datasets (amazon, caltech, dslr, webcam) are included in the repository.

### Domain
For the domain adaptation datasets, please download from the following link:
[Domain Adaptation Datasets](https://drive.google.com/file/d/1pj7pk73OYeGhYXp9Nptmpw8nSxUe4dEY/view?usp=sharing)

After downloading, extract the contents to the `data/domain/` directory.

## Usage

### Basic Running Commands

```bash
# Run on digit dataset with 5 clients and 50 rounds
python main.py --dataset digit --num_clients 5 --rounds 50 --lamb 50

# Run on office dataset with 4 clients and 80 rounds
python main.py --dataset office --num_clients 4 --rounds 80 --lamb 10

# Run on domain dataset with 6 clients and 200 rounds
python main.py --dataset domain --num_clients 6 --rounds 200 --lamb 1 --n_per_class 30
```



