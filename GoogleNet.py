# GoogleNet
# Inception块：全都要，4个路径从不同层面抽取信息，然后在输出通道维合并(concat)
# 分别是1×1卷积（通道压缩到64），1×1卷积+3×3卷积 pad 1（96->128)，1×1卷积+5×5卷积 pad2（16->32)，3×3最大池化 pad 1+1×1卷积(32) (长宽是不变的)
# 跟单个3×3或5×5卷积层比，Inception块所需要更少的参数个数和计算复杂度
# 5段，9个Inception块 

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms,datasets
from torch.utils import data
from VGG import load_data_fashion_mnist,train_ch6,draw_loss_acc

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__() #继承父类nn.module时，必须调用super().__init__()实现初始化
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1=F.relu(self.p1_1(x))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x)))) #nn.ReLU()是类，继承nn.Module,不直接使用，F.relu()是函数
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4=F.relu(self.p4_2(self.p4_1(x)))

        # torch.cat(tensors, dim=0, *, out=None) -> Tensor
        # ensors：要拼接的张量序列，例如 (p1, p2, p3, p4) 或 [p1, p2, p3, p4]。
        # dim：沿哪个维度拼接。除 dim 外，其它维度大小必须相同。

        # torch.stack([a,b,c], dim=0)
        # 作用：把多个张量堆叠出一个新维度，不要求某一维“变长”，而是多出一维。
        # 要求：所有张量的 shape 必须完全一致。
        # a = torch.randn(3, 4)
        # b = torch.randn(3, 4)
        # c = torch.stack((a, b), dim=0)   # (2, 3, 4)，多了一个 dim=0
        # d = torch.stack((a, b), dim=1)   # (3, 2, 4)，在 dim=1 处多了一维

        return torch.cat((p1,p2,p3,p4),dim=1) # 在第一个维度上合并（通道维）

# 5个stage
b1=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

b2=nn.Sequential(
    nn.Conv2d(64,64,kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64,192,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

b3=nn.Sequential(
    Inception(192,64,[96,128],[16,32],32),
    Inception(256,128,[128,192],[32,96],64),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

b4=nn.Sequential(
    Inception(480,192,[96,208],[16,48],64),
    Inception(512,160,[112,224],[24,64],64),
    Inception(512,128,[128,256],[24,64],64),
    Inception(512,112,[144,288],[32,64],64),
    Inception(528,256,[160,320],[32,128],128),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

b5=nn.Sequential(
    Inception(832,256,[160,320],[32,128],128),
    Inception(832,384,[192,384],[48,128],128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

def init_weights(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net=nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10)) 

x=torch.randn(size=(1,1,96,96))
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,":",x.shape)

if __name__=='__main__':
    net.apply(init_weights)
    loss=nn.CrossEntropyLoss()
    lr,num_epochs,batch_size=0.05,10,128
    train_iter,test_iter=load_data_fashion_mnist(batch_size=batch_size,resize=96)
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    train_acc,train_loss,test_acc=train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,device='cuda:0')
    draw_loss_acc(train_acc,train_loss,test_acc,name='GoogleNet')

