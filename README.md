# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype


## Project Structure

```
FedOrthrus/
├── data/                 # Dataset directory
│   └── digit/            # Digit datasets (MNIST, MNIST_M, SVHN, etc.)
│       ├── MNIST/        # MNIST dataset
│       ├── MNIST_M/      # MNIST-M dataset
│       ├── SVHN/         # SVHN dataset
│       ├── SynthDigits/  # SynthDigits dataset
│       └── USPS/         # USPS dataset
├── models/               # Model definitions
│   └── resnet.py         # ResNet10 model implementation
├── utils/                # Utility functions
│   ├── options.py        # Command-line parameter configuration
│   ├── update.py         # Local training and testing implementation
│   ├── util.py           # Prototype aggregation, clustering, and other utility functions
│   ├── data_util.py      # Data preparation and preprocessing
│   ├── dataset.py        # Dataset processing tools
│   └── finch.py          # FINCH clustering algorithm implementation
├── main.py               # Project main entry
├── acc.npy               # Experiment results storage file
└── LICENSE               # License file
```

## Environment Setup

### Create Virtual Environment

```bash
# Using conda to create virtual environment
conda create -n fedorthrus python=3.9 -y
conda activate fedorthrus

# Or using venv to create virtual environment
python -m venv fedorthrus_venv
# Activate on Windows
fedorthrus_venv\Scripts\activate
# Activate on Linux/Mac
# source fedorthrus_venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Running Command

```bash
python main.py --dataset digit --num_clients 5 --rounds 50 --N 316
```

### Main Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | cuda | Running device (cpu, cuda, or others) |
| `--gpu` | int | 0 | GPU index |
| `--seed` | int | 0 | Random seed |
| `--dataset` | str | digit | Dataset type (digit, office, domain) |
| `--num_clients` | int | 5 | Number of clients |
| `--rounds` | int | 50 | Number of training rounds |
| `--N` | int | 316 | Number of aggregation dimensions |
| `--lr` | float | 0.01 | Learning rate |
| `--local_bs` | int | 32 | Local batch size |
| `--train_ep` | int | 2 | Number of local epochs |
| `--lamb` | float | 50 | CE loss weight parameter |
| `--tau` | float | 0.07 | Loss temperature |

### Recommended Parameters for Different Datasets

| Dataset | num_clients | rounds | N | lamb |
|---------|-------------|--------|---|------|
| digit | 5 | 50 | 316 | 50 |
| office | 4 | 80 | 192 | 10 |
| domain | 6 | 200 | 192 | 1 |

### Running Examples

```bash
# Run on digit dataset, 5 clients, 50 rounds
python main.py --dataset digit --num_clients 5 --rounds 50 --N 316 --lamb 50

# Run on office dataset, 4 clients, 80 rounds
python main.py --dataset office --num_clients 4 --rounds 80 --N 192 --lamb 10

# Run on domain dataset, 6 clients, 200 rounds
python main.py --dataset domain --num_clients 6 --rounds 200 --N 192 --lamb 1
```



