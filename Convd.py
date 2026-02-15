#卷积
#1.局部性；2.平移不变性
#卷积层将输入和核矩阵进行交叉相关，是特殊全连接，加上偏移后得到输出
#核矩阵和偏移是可学习参数
#核矩阵的大小是超参数

#图像卷积
import torch
from torch import nn

def corr2d(X,K):
    h,w=K.shape
    Y=torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

X=torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
K=torch.tensor([[0.0,1.0],[2.0,3.0]])
print(corr2d(X,K))

#定义卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,X):
        return corr2d(X,self.weight)+self.bias

cov=Conv2D((3,3))
print(cov.bias)

#卷积层简单应用：检测图像中不同颜色的边缘
X=torch.ones((6,8))
X[:,2:6]=0
print(X)

K=torch.tensor([[1.0,-1.0]]) #如果相邻两元素相同，卷积结果为0
Y=corr2d(X,K)
print(corr2d(X,K)) #由白变黑边缘，黑变白边缘,但只可以检测垂直边缘

#学习由X生成Y的卷积核
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X=X.reshape((1,1,6,8)) #加入batch_size维和通道维
Y=Y.reshape((1,1,6,7))

for i in range(10):
    conv2d.zero_grad()
    Y_hat=conv2d(X)
    l=(Y_hat-Y)**2
    l.sum().backward()
    conv2d.weight.data-=3e-2*conv2d.weight.grad
    print(f'epoch:{i+1} , loss:{l.sum():.3f}')

print(conv2d.weight.data) #data就是存tensor格式