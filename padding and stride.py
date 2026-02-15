#填充和步幅
#填充ph行和pw列，输出形状变为（nh-kh+ph+1）×（nw-kw+pw+1）

#步幅：可以移动多格
#输出形状：向下取整：[(nh-kh+ph+sh)/sh]×[(nw-kw+pw+sw)/sw]

import torch
from torch import nn

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape) #加上batch_size和channel维度
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:]) #取后面两维

conv2d=nn.Conv2d(1,1,3,padding=1)
X=torch.randn((8,8))
print(comp_conv2d(conv2d,X).shape)

#不对称填充
conv2d=nn.Conv2d(1,1,(5,3),padding=(2,1)) #行各填充两行，列各填充一列
print(comp_conv2d(conv2d,X).shape)

#步幅
conv2d=nn.Conv2d(1,1,3,padding=1,stride=2)
print(comp_conv2d(conv2d,X).shape)

conv2d=nn.Conv2d(1,1,(3,5),padding=(0,1),stride=(3,4))
print(comp_conv2d(conv2d,X).shape)