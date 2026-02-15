# NiN网络，完全不要全连接层，使用1×1卷积层替代
# NiN块，一个卷积层 接两个1×1卷积层，再接一个激活函数
# 交替使用 NiN块和步幅为2的最大池化层，逐步 减小高宽和增大通道数，最后使用全局平均池化层得到输出，其输入通道数是类别数
# 1×1卷积层对每个像素增加非线性性，NiN使用全局平均池化层替代全连接层

import torch
from torch import nn
from VGG import load_data_fashion_mnist,train_ch6,draw_loss_acc

def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
    )

def init_weights(m):
    if type(m)==nn.Conv2d:
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

net=nn.Sequential(
    nin_block(1,96,kernel_size=11,strides=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(96,256,kernel_size=5,strides=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256,384,kernel_size=3,strides=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),nn.Dropout(0.5),
    nin_block(384,10,kernel_size=3,strides=1,padding=1),
    nn.AdaptiveAvgPool2d((1,1)), # 自适应平均池化层，（i，j）是输出宽高，自适应调整，例如(1,1)表示输出宽高为1×1 
    # nn.AvgPool2d(kernel_size=5)
    nn.Flatten()
)

net.apply(init_weights)

x=torch.randn(size=(1,1,224,224))
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,":",x.shape)

if __name__=="__main__":
    batch_size=256
    train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=224)
    lr,num_epochs=0.01,10
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    train_acc,train_loss,test_acc=train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,'cuda:1')
    draw_loss_acc(train_acc,train_loss,test_acc,name='NiN')