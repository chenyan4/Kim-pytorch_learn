#用于分类问题
#回归：单连续数值输出，跟真实值区别做损失，通常单个输出
#分类：通常多个输出，输出i是预测为第i类的置信度

#损失函数
#1.均方损失 梯度是2y_hat均匀增大
#2.绝对值损失函数|y-y_hat|,导数为1或-1，梯度稳定，但0处无梯度
#3.Huber损失，综合前两个损失，在|y-y_hat|>1时用绝对值损失，其余用均方损失，保证0处有梯度并且梯度是渐变过程

#1.图像分类数据集
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

#通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
#并除以255使得所有像素的数值均在0到1之间
trans=transforms.ToTensor()
minist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
minist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)

# print(minist_train[1])

x,y=next(iter(data.DataLoader(minist_train,batch_size=18))) #只是取它们的一份（18）

batch_size=256

train_iter=data.DataLoader(minist_train,batch_size=batch_size,shuffle=True) #四进程读取数据


for x,y in train_iter:
    print(x.shape,'\n',y.shape)
#定义数据集加载函数
def load_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4))
