# Two Heads Are Better Than One: Generalized Cross-Domain Federated Learning via Dual-Prototype



## 项目结构

```
FedOrthrus/
├── data/                 # 数据集目录
│   └── digit/            # 数字数据集（MNIST、MNIST_M、SVHN等）
│       ├── MNIST/        # MNIST数据集
│       ├── MNIST_M/      # MNIST-M数据集
│       ├── SVHN/         # SVHN数据集
│       ├── SynthDigits/  # SynthDigits数据集
│       └── USPS/         # USPS数据集
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
# Windows激活
fedorthrus_venv\Scripts\activate
# Linux/Mac激活
# source fedorthrus_venv/bin/activate
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本运行命令

```bash
python main.py --dataset digit --num_clients 5 --rounds 50 --N 316
```

### 主要参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--device` | str | cuda | 运行设备（cpu或cuda） |
| `--gpu` | int | 0 | GPU索引 |
| `--seed` | int | 0 | 随机种子 |
| `--dataset` | str | digit | 数据集类型（digit、office、domain） |
| `--num_clients` | int | 5 | 客户端数量 |
| `--rounds` | int | 50 | 训练轮数 |
| `--N` | int | 316 | 聚合维度数 |
| `--lr` | float | 0.01 | 学习率 |
| `--local_bs` | int | 32 | 本地批次大小 |
| `--train_ep` | int | 2 | 本地训练轮数 |
| `--lamb` | float | 50 | CE损失权重参数 |
| `--tau` | float | 0.07 | InfoNCE损失温度参数 |

### 不同数据集的推荐参数

| 数据集 | num_clients | rounds | N | lamb |
|--------|-------------|--------|---|------|
| digit | 5 | 50 | 316 | 50 |
| office | 4 | 80 | 192 | 10 |
| domain | 6 | 200 | 192 | 1 |

### 运行示例

```bash
# 在digit数据集上运行，5个客户端，50轮训练
python main.py --dataset digit --num_clients 5 --rounds 50 --N 316 --lamb 50

# 在office数据集上运行，4个客户端，80轮训练
python main.py --dataset office --num_clients 4 --rounds 80 --N 192 --lamb 10

# 在domain数据集上运行，6个客户端，200轮训练
python main.py --dataset domain --num_clients 6 --rounds 200 --N 192 --lamb 1
```

