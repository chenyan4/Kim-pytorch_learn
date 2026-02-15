#自动求导
import torch

#假设函数y=2xTx
x=torch.arange(4.0)

#1.计算x梯度之前，需要地方存储梯度
x.requires_grad_(True)
x.grad #可以访问梯度，默认值是None

#2.定义y
y=2*torch.dot(x,x) #内积，也就是x和x的内积
print(y)

#3.方向传播求梯度
y.backward()
print(x.grad==4*x)
print(x.grad)
#由于pytorch自动累积梯度，要清零
# x.grad.zero_()
y=x.sum()
y.backward()
print(x.grad)

#4.将某些计算移动到记录的计算图之外，固定网络参数时很有用
x.grad.zero_()
y=x*x
u=y.detach() #把y当作一个常数，而不是关于x的函数，即此时u是常数
z=u*x

z.sum().backward()
print(x.grad==u)

#5.构建函数的计算图需要通过python控制流，仍可以计算变量的梯度
def f(a):
    b=a*2
    print(b)
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c

a=torch.randn(size=(),requires_grad=True) #不指定size为标量
print(a)
d=f(a)
d.backward()
print(a.grad==d/a)