# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype



## 项目结构

```
FedOrthrus/
├── data/                 # 数据集目录
│   ├── digit/            # 数字数据集（MNIST、MNIST_M、SVHN等）
│   │   ├── MNIST/        # MNIST数据集
│   │   ├── MNIST_M/      # MNIST-M数据集
│   │   ├── SVHN/         # SVHN数据集
│   │   ├── SynthDigits/  # SynthDigits数据集
│   │   └── USPS/         # USPS数据集
│   ├── office/           # 办公用品数据集（amazon、caltech、dslr、webcam）
│   └── domain/           # 域适应数据集（需要下载）
├── models/               # 模型定义
│   └── resnet.py         # ResNet10模型实现
├── utils/                # 工具函数
│   ├── options.py        # 命令行参数配置
│   ├── update.py         # 本地训练和测试实现
│   ├── util.py           # 原型聚合、聚类等工具函数
│   ├── data_util.py      # 数据准备和预处理
│   ├── dataset.py        # 数据集处理工具
│   └── finch.py          # FINCH聚类算法实现
├── main.py               # 项目主入口
├── acc.npy               # 实验结果存储文件
└── LICENSE               # 许可证文件
```

## 环境配置

### 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n fedorthrus python=3.9 -y
conda activate fedorthrus

# 或使用venv创建虚拟环境
python -m venv fedorthrus_venv

```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

FedOrthrus在三个数据集上进行了实验：

### Digit
数字数据集（MNIST、MNIST-M、SVHN、SynthDigits、USPS）可从以下链接下载：
[数字数据集](https://drive.google.com/drive/folders/1gWrTqBhsuhi6DeP6s9POUg3-ZBf7YMaO?usp=sharing)

下载后，将文件内容解压到 `data/digit/` 目录中。

### Office
办公用品数据集（amazon、caltech、dslr、webcam）可从以下链接下载：
[办公用品数据集](https://drive.google.com/drive/folders/1OKFcnBL-ijq9-IlB9H2-NMStEDSXU98f?usp=sharing)

下载后，将文件内容解压到 `data/office/` 目录中。

### Domain
对于域适应数据集，请从以下链接下载：
[域适应数据集](https://drive.google.com/file/d/1pj7pk73OYeGhYXp9Nptmpw8nSxUe4dEY/view?usp=sharing)

下载后，将文件内容解压到 `data/domain/` 目录中。

**注意：** 所有数据集应按照上述项目结构放置在根目录的data文件夹中。

## 使用方法

### 快速开始
您可以直接运行main文件，使用默认设置开始digit实验：
```bash
python main.py
```

### 运行示例(参数设置与论文相同)

```bash
# 在digit数据集上运行，5个客户端，50轮训练
python main.py --dataset digit --num_clients 5 --rounds 50 --lamb 50

# 在office数据集上运行，4个客户端，80轮训练
python main.py --dataset office --num_clients 4 --rounds 80 --lamb 10

# 在domain数据集上运行，6个客户端，200轮训练
python main.py --dataset domain --num_clients 6 --rounds 200 --lamb 1 --n_per_class 30
```

### 性能说明
在配备RTX 4090的设备上：
- digit数据集实验：大约4分钟
- office数据集实验：大约15分钟
- domain数据集实验：大约35分钟

