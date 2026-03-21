# 全连接卷积神经网络（FCN）
# 用深度学习做语义分割的 开创性工作
# 用转置卷积层，替换全连接层

import torch
import torchvision
from torch import nn
from torch.nn import functional as 

# 用预训练的resnet18 抽取特征
resnet18=torchvision.models.resnet18(pretrain=True)
print(resnet18)