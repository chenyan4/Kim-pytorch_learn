# 多GPU的简洁实现

import torch
from torch import nn
from multi_GPU import load_data_fashion_mnist,draw_acc
from ResNet import evaluate_accuracy
import time


class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1,stride=1)

        if use_1conv:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        Y=self.relu(self.bn1(self.conv1(x)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            x=self.conv3(x)
        Y=Y+x
        return self.relu(Y)

def resnet_18(num_classes,in_channels=1):
    def resnet_block(in_channels,num_channels,num_residuals,first_block=False):
        blk=[]
        for i in range(num_residuals):
            if i==0 and not first_block:
                blk.append(Residual(in_channels,num_channels,use_1conv=True,strides=2))
            else:
                blk.append(Residual(num_channels,num_channels))
        return nn.Sequential(*blk)

    net=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
    resnet_block(64,64,2,True),resnet_block(64,128,2),resnet_block(128,256,2),resnet_block(256,512,2),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,num_classes))

    return net

def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight,0,0.01)

def train(net,num_gpus,batch_size,lr):
    train_iter,test_iter=load_data_fashion_mnist(batch_size)
    devices=[f'cuda:{i}' for i in range(num_gpus)]

    # net.apply(init_weight)
    net.to(devices[0])
    net=nn.DataParallel(net,device_ids=devices) # 调用nn.DataParellel 把net复制到每个GPU上,会自动把x切到不同的 GPU上，相当于重新定义 forward函数
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    num_epochs=10
    test_acc=[]
    start_time=time.time()
    for epoch in range(num_epochs):
        for x,y in train_iter:
            updater.zero_grad()
            x,y=x.to(devices[0]),y.to(devices[0])
            l=loss(net(x),y)
            l.backward()
            updater.step()

        test_acc.append(evaluate_accuracy(net,test_iter,'cuda:0'))
        print(f'epoch:{epoch+1},test_acc:{test_acc[-1]:.4f}') 
    end_time=time.time()
    print(f'GPU数量:{num_gpus},用时:{(end_time-start_time):.4f}')

    return test_acc

if __name__=="__main__":
    net=resnet_18(10)
    test_acc=train(net,2,batch_size=256*2,lr=0.001*2)
    draw_acc(test_acc,name="multi_GPU_simple")