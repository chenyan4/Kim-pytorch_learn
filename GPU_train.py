import torch
from torch import nn

a=torch.ones(3,1)
a=a.cuda(0)
b=torch.ones(3,1).cuda(0)
print(a+b)

#如何指定GPU
torch.device('cpu')
torch.cuda.device('cuda') #表示第零个GPU
# torch.cuda.device('cuda:1') 指定第一个GPU

print(torch.cuda.device_count()) #查看GPU个数

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
print(try_gpu())

#查询张量所在设备
# x=torch.tensor([1,2,3])
x=torch.ones(2,3,device=try_gpu()) #创建张量在GPU上
print(x)

#如果要计算x+y，会发生在对应的device上，且必须在同一个GPU上才能计算

#神经网络到GPU
net=nn.Sequential(nn.Linear(3,1))
net=net.to(device=try_gpu())
print(net(x))

print(net[0].weight.data) #取权重的值