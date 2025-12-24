# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype



## 项目结构

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

## 环境配置

```bash
# 1. 创建conda环境
conda create -n fedorthrus python=3.9 -y

# 2. 激活环境
conda activate fedorthrus

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据集

FedOrthrus在三个数据集上进行了实验：

### Digit
数字数据集（MNIST、MNIST-M、SVHN、SynthDigits、USPS）可从以下链接下载：
[数字数据集](https://drive.google.com/drive/folders/1gWrTqBhsuhi6DeP6s9POUg3-ZBf7YMaO?usp=sharing)

下载后，将文件内容解压到 `FedOrthrus/data/digit/` 目录中。预期的目录结构为：
```
FedOrthrus/data/digit/
├── MNIST/
├── MNIST_M/
├── SVHN/
├── SynthDigits/
└── USPS/
```

### Office
办公用品数据集（amazon、caltech、dslr、webcam）可从以下链接下载：
[办公用品数据集](https://drive.google.com/drive/folders/1OKFcnBL-ijq9-IlB9H2-NMStEDSXU98f?usp=sharing)

下载后，将文件内容解压到 `FedOrthrus/data/office/` 目录中。预期的目录结构为：
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
对于域适应数据集，请从以下链接下载：
[域适应数据集](https://drive.google.com/file/d/1pj7pk73OYeGhYXp9Nptmpw8nSxUe4dEY/view?usp=sharing)

下载后，将文件内容解压到 `FedOrthrus/data/domain/` 目录中。预期的目录结构为：
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

**注意：** 所有数据集应按照上述项目结构放置在根目录的FedOrthrus/data文件夹中。

## 使用方法

### 快速开始（实验一）
这是最小可复现示例，对应论文的**表一**：
```bash
python main.py
```

### 完整运行命令

```bash
# 实验一（表一）：Digit数据集
python main.py --dataset digit --num_clients 5 --rounds 50 --lamb 50

# 实验二（表二）：Office数据集
python main.py --dataset office --num_clients 4 --rounds 80 --lamb 10

# 实验三（表六）：Domain数据集
python main.py --dataset domain --num_clients 6 --rounds 200 --lamb 1 --n_per_class 30
```

### 性能说明
在配备RTX 4090的设备上：
- digit数据集实验：大约4分钟
- office数据集实验：大约15分钟
- domain数据集实验：大约35分钟

