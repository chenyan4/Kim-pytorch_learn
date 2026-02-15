#数据操作
import torch

#1.张量表示一个数值组成的数组，这个数组可能有很多个维度
x=torch.arange(12)
print(x)
print(x.shape) #长度，一个维度
print(x.numel()) #元素总数

#2.改变一个张量的形状而不改变元素数量和元素值，可以用reshape
x=x.reshape(3,4) #三行四列
print(x)

#3.创建全0、全1，zeros(),ones()
x=torch.zeros((2,3,4)) #三维,两个三行四列
y=torch.ones((2,3,4))
print(x,y)

#4.可提供包含数值的python列表来给张量中每个元素赋值
x=torch.tensor([[1,2,3],[2,3,4]]) #二维数组

#5.标准运算（+，-，×,÷）
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
print(x+y,x-y,x*y,x/y)
torch.exp(x) #指数运算

#6.把多个张量连接在一起
x=torch.arange(12,dtype=torch.float32).reshape((3,4)) #原本一维，变为二维
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
a1=torch.cat((x,y),dim=0) #行上合并（堆起来）
a2=torch.cat((x,y),dim=1) #列上合并，沿行拼接
print(a1)
print(a2)

#7.通过逻辑运算符构建二维张量
print(x==y)

#8.对张量中所有元素求和，产生一个元素张量
print(x.sum())

#9.广播机制
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
#x=a+b 会把a的1编程2，把b的1变成3相加

#10.读取张量元素
x[-1] #最后一行
x[1:3] #一到二行
x[1,2] #第一行，第二列
x[0:2,:] #第0-1行，选择所有列

#11.转换为NnmPy张量
a=x.numpy()
b=torch.tensor(a)

#12.大小为1的张量转换为python标量
a=torch.tensor([3.5])
print(a,a.item(),float(a))

def sigmoid(x):
    x_exp=torch.exp(-x)
    ones=torch.ones_like(x)
    return ones/(x_exp+ones)

print(sigmoid(torch.FloatTensor([[1,2,3],[4,5,6]])))


