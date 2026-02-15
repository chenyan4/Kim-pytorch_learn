#网络模型层和块
import torch
from torch import nn
from torch.nn import functional as F #激活函数和其他方法

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

#自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x-x.mean()

layer=CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

#可将层作为组件合并到构建更复杂的模型中
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())

#带参数的图层
class MyLinear(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(in_dim,out_dim))
        self.bias=nn.Parameter(torch.randn(out_dim,))

    def forward(self,x):
        Linear=torch.matmul(x,self.weight.data)+self.bias.data #要用.data取数据
        return F.relu(Linear)

dense=MyLinear(5,3)
print(dense.state_dict())
print(dense(torch.randn(2,5)))

#使用自定义层构造模型
net=nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net.state_dict())
print(net(torch.randn(2,64)))

#读取文件，加载和保存张量
x=torch.arange(4)
torch.save(x,'data/x-file')

x2=torch.load('data/x-file')
print(x2)