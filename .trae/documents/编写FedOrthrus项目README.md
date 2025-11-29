# FedOrthrus项目README编写计划

## 1. 项目概述
- 项目名称：FedOrthrus
- 项目类型：联邦学习框架
- 核心功能：处理跨域/多源数据的联邦分类任务
- 应用场景：数字识别、办公场景图像分类等

## 2. 项目结构
```
FedOrthrus/
├── data/                 # 数据集目录
│   └── digit/            # 数字数据集（MNIST、MNIST_M、SVHN等）
├── models/               # 模型定义
│   └── resnet.py         # ResNet10模型
├── utils/                # 工具函数
│   ├── options.py        # 参数配置
│   ├── update.py         # 本地训练和测试
│   ├── util.py           # 原型聚合、聚类等工具
│   ├── data_util.py      # 数据准备
│   ├── dataset.py        # 数据集处理
│   └── finch.py          # 聚类算法
├── main.py               # 项目入口
└── LICENSE               # 许可证
```

## 3. 核心算法
- 双原型机制：全局首原型（global_N_protos）和全局聚类次原型（global_cluster_N_M_protos）
- 特征分割与聚合：将特征向量分为两部分进行不同方式的聚合
- 本地聚类与全局聚类相结合
- 使用InfoNCE损失和修正损失进行模型训练

## 4. 支持的数据集
- digit：MNIST、MNIST_M、SVHN、SynthDigits、USPS
- office：办公场景数据集
- domain：其他域适应数据集

## 5. 使用方法
- 基本运行命令
- 参数配置说明
- 实验结果记录

## 6. 实验结果
- 不同数据集上的准确率
- 与其他方法的比较

## 7. 参考文献
- KDD接收论文引用

## 8. 作者信息
- 项目维护者
- 联系方式

## 9. 许可证
- 项目许可证类型

现在我将编写完整的README.md文件。