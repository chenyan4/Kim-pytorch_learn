#线性代数
import torch

#1.标量由只有一个元素的张量表示
x=torch.tensor([3.0])

#2.向量视为标量值组成列表
x=torch.arange(4)
#只有一个轴的张量。形状只有一个元素
print(x.shape)

#3.指定两个分量m和n来创建一个形状为m,n矩阵
A=torch.arange(20).reshape(5,4)
#转置
print(A.T)

#4.给定相同形状的张量，任何按元素二元运算的结果是相同形状张量
A=torch.arange(20,dtype=torch.float32).reshape(5,4)
B=A.clone()
print(A+B)
#矩阵按元素乘法
print(A*B)

#5.任意张量元素和
A=torch.arange(20*2).reshape(2,5,4)
print(A.sum())
print(A.sum(axis=0)) #按2维度求和，(5,4)
print(A.sum(axis=1)) #按5维度求和，(2,4) 不同行元素相加
print(A.sum(axis=2)) #按4维度求和,(2,5) 不同列元素相加
print(A.sum(axis=[0,1])) #按两个维度求和，可以理解为先按2，再按5

#6.平均值
A=torch.arange(20,dtype=torch.float32).reshape(5,4)
print(A.mean(axis=0)) #跟据5这个维度算均值，就是A.mean(axis=0)/A.shape[0] ,先对5这个维求和再除
print(A.mean(axis=1,keepdim=True)) #对求和维度，让维度等于1而不直接去掉维度，保持轴数不变
#累加求和
print(A.cumsum(axis=0)) #在5这个维度上累加元素

#7.矩阵点积，按位置乘法
x=torch.arange(4,dtype=torch.float32)
y=torch.ones(4)
print(torch.dot(x,y)) #向量点积操作
print(torch.mv(A,x)) #矩阵和向量点积操作
B=torch.zeros(4,3)
print(torch.mm(A,B)) #矩阵和矩阵点积操作

#8.范数（L1、L2）
x=torch.tensor([3.0,-4.0])
print(torch.norm(x)) #张量中每个元素平方相加后求和再开放，L2范数
print(torch.abs(x).sum()) #张量中每个元素取绝对值求和，L1范数
print(torch.norm(torch.ones((4,9)))) #矩阵F范数，即矩阵中每个元素平方求和后再开方