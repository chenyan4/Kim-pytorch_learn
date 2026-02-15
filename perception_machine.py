#mlp,多层感知机
#二分类问题，y=wx+b,等价于批量为1的梯度下降，不能你和XOR函数
#激活函数为什么是非线性，如果是线性激活，多个连接层最后输出的结果依然是线性的，这和单层感知机就没有区别了
#激活函数就是处理线性问题的，防止感知机层数扁平或者说失效，因为线性后几层也没区别

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils import data
from softmax_achieve import train_ch3

def load_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4))

batch_size=256
train_iter,test_iter=load_fashion_mnist(batch_size)

#初始化权重w1,b1,w2,b2
num_inputs,num_outputs,num_hiddens=784,10,256
w1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True))
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True)) #和直接定义没区别
# b1_temp=torch.zeros(num_hiddens,requires_grad=True)
w2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[w1,b1,w2,b2]

#定义relu激活函数
def relu(X):
    a=torch.zeros_like(X) #生成和X数据类型、形状一致的全零张量
    return torch.max(a,X) #会对相应位置上元素一一比较取最大



#定义网络
def net(X):
    X=X.reshape(len(X),num_inputs)
    H=relu(torch.matmul(X,w1)+b1)
    return torch.matmul(H,w2)+b2

loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(params=params,lr=0.1)

if __name__=="__main__":
    num_epochs=10
    train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
    # x=torch.tensor([[1,-1,1],
    #                [2,-1,2]])
    # print(relu(x))




