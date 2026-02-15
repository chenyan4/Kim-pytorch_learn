# 池化层：最大池化层、平均池化层
# 参数：填充，步幅、核大小，不融合通道，不改变通道

# 最大池化层（取最强的信号）
#平均池化层（取平均信号强度，比较柔和）

import torch
from torch import nn

def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    y=torch.zeros(size=(X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode=='max':
                y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                y[i,j]=X[i:i+p_h,j:j+p_w].sum()/(p_h*p_w)
                # y[i,j]=X[i:i+p_h,j:j+p_w].mean()

    return y

X=torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(pool2d(X,pool_size=(2,2),mode='max'))
print(pool2d(X,pool_size=(2,2),mode='avg'))

X=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
# 深度学习框架中步幅与池化窗口大小相同
pool2d=nn.MaxPool2d(3)
print(pool2d(X))

pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))

pool2d=nn.MaxPool2d((2,3),padding=(1,1),stride=(2,3))
print(pool2d(X))


X=torch.cat((X,X+1),dim=1) # 在通道维上连接
print(X)

pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))