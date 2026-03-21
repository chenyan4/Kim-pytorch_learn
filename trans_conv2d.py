# 转置卷积
# 输入为n，核为k，填充为p，步长为s，在转置卷积中意为 在每个元素之间填充 s-1 行或列，再在 外层填充 k-p-1 ,最后得到 输出为 ns-s+k-2p,如果要成倍 放大高宽 k=s+2p
import torch
from torch import nn
def trans_conv(X,K):
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-1+h,X.shape[1]-1+w))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h,j:j+w]+=X[i,j]*K

    return Y

nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,bias=False)

X=torch.tensor([[0.0,1.0],[2.0,3.0]])
K=torch.tensor([[0.0,1.0],[2.0,3.0]])

print(trans_conv(X,K))

X,K=X.reshape(1,1,2,2),K.reshape(1,1,2,2)
tconv=nn.ConvTranspose2d(1,1,kernel_size=2,bias=False)
tconv.weight.data=K
print(tconv(X))

tconv=nn.ConvTranspose2d(1,1,kernel_size=2,padding=1,bias=False)
tconv.weight.data=K
print(tconv(X))

tconv=nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,bias=False)
tconv.weight.data=K
print(tconv(X))

X=torch.randn(size=(1,10,16,16))
conv=nn.Conv2d(10,20,kernel_size=5,padding=2,stride=3)
tconv=nn.ConvTranspose2d(20,10,kernel_size=5,padding=2,stride=3)
print(tconv(conv(X)).shape==X.shape)
