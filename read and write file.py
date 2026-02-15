#读写文件
import torch
from torch import nn
from torch.nn import functional as F

x=torch.arange(4)
torch.save(x,'data/x-file')

x2=torch.load('data/x-file')

#存储一个list
y=torch.zeros(4)
torch.save([x,y],'data/x-files')
x2,y2=torch.load('data/x-files')
print((x2,y2))

#写入或读取从字符串映射到张量的字典
mydict={'x':x,'y':y}
torch.save(mydict,'data/mydict')
mydict2=torch.load('data/mydict')
print(mydict2) #返回字典

#保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.output=nn.Linear(256,10)

    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))

net=MLP()
x=torch.randn(2,20)
y=net(x)

#net.state_dict() 可以得到字符串到参数的映射，字典
torch.save(net.state_dict(),'data/mlp.params') #把mlp存成一个字典

#还原时，要实例化原始多层感知机模型的备份，直接读取文件中存储的参数
clone=MLP()
clone.load_state_dict(torch.load('data/mlp.params')) #实例化多层感知机
Y_clone=clone(x)
print(Y_clone==y)