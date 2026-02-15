# ResNet 残差网络
# 加更多层不一定更好，也不一定离最优解更近
# 如果大模型能包含小模型，至少不会变差，加层不会变差
# 输出f(x)=x+g(x),g(x)是残差，还是能得到原来的x
# ResNet块，一个高宽减半，通道翻倍的ResNet块（x走捷径，要过一个1×1卷积，步幅为2，实现高宽减半，通道翻倍）；后接几个高宽相同的ResNet块（可能捷径就跟1×1卷积，步幅为1）

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,datasets

class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1conv:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU(inplace=True)  # inplace=True表示直接在输入张量上做运算，不重新分配一个张量

    def forward(self,X):
        Y=self.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        Y=self.relu(Y)
        return Y

blk=Residual(3,3)
blk=Residual(3,6,use_1conv=True,strides=2)
X=torch.rand(4,3,6,6)
Y=blk(X)
print(Y.shape)

b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding)