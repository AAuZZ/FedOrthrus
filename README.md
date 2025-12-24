# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype


## Project Structure

```
FedOrthrus/
├── data/
│   ├── digit/
│   ├── office/
│   └── domain/
├── models/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
# 1. Create conda environment
conda create -n fedorthrus python=3.9 -y

# 2. Activate the environment
conda activate fedorthrus

# 3. Install dependencies
pip install -r requirements.txt
```

## Datasets

FedOrthrus has been evaluated on three datasets:

### Digit
The digit dataset (MNIST, MNIST-M, SVHN, SynthDigits, USPS) can be downloaded from:
[Digit Dataset](https://drive.google.com/drive/folders/1gWrTqBhsuhi6DeP6s9POUg3-ZBf7YMaO?usp=sharing)

After downloading, extract the files to `FedOrthrus/data/digit/`. The expected directory structure is:
```
FedOrthrus/data/digit/
├── MNIST/
├── MNIST_M/
├── SVHN/
├── SynthDigits/
└── USPS/
```

### Office
The office dataset (amazon, caltech, dslr, webcam) can be downloaded from:
[Office Dataset](https://drive.google.com/drive/folders/1OKFcnBL-ijq9-IlB9H2-NMStEDSXU98f?usp=sharing)

After downloading, extract the files to `FedOrthrus/data/office/`. The expected directory structure is:
```
FedOrthrus/data/office/
├── amazon/
├── caltech/
├── caltech_manual/
├── dslr/
├── webcam/
├── amazon_train.pkl
├── amazon_test.pkl
├── caltech_train.pkl
├── ...
```

### Domain
For the domain adaptation datasets, please download from the following link:
[Domain Adaptation Datasets](https://drive.google.com/file/d/1pj7pk73OYeGhYXp9Nptmpw8nSxUe4dEY/view?usp=sharing)

After downloading, extract the contents to the `FedOrthrus/data/domain/` directory. The expected directory structure is:
```
FedOrthrus/data/domain/
├── clipart/
├── infograph/
├── painting/
├── quickdraw/
├── real/
├── sketch/
├── clipart_test.pkl
├── clipart_train.pkl
├── infograph_test.pkl
├── ...
```

**Note:** All datasets should be placed in the `FedOrthrus/data/` folder according to the project structure shown above.

## Usage

### Quick Start (Experiment 1)
This is the minimal reproducible example corresponding to **Table 1** in the paper:
```bash
python main.py
```

### Complete Running Commands

```bash
# Experiment 1 (Table 1): Digit dataset
python main.py --dataset digit --num_clients 5 --rounds 50 --lamb 50 --num_exps 5

# Experiment 2 (Table 2): Office dataset
python main.py --dataset office --num_clients 4 --rounds 80 --lamb 10 --num_exps 5

# Experiment 3 (Table 6): Domain dataset
python main.py --dataset domain --num_clients 6 --rounds 200 --lamb 1 --n_per_class 30 --num_exps 5
```

### Performance Notes
On a device with RTX 4090:
- Digit dataset experiment: approximately 4 minutes
- Office dataset experiment: approximately 15 minutes
- Domain dataset experiment: approximately 35 minutes



