import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils import data
from softmax_achieve import train_ch3

#pytorch不会隐式调整输入的形状
#定义展平层Flatten()，在线性层前调整形状,Flatten跳过第0维度(batch_size)，从第一维度到最后维度展平
#flatten(start_dim,end_dim) 从start到end维度展平
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

# x=torch.arange(600).reshape(10,20,3)
# x=nn.Flatten(x)
# print(x)

def init_weights(m): #m表示成当前layer
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights) #相当于把函数应用到网络中的所有层，layer作为输入，然后如果是Linear层做，有点像map

#定义损失函数
loss=nn.CrossEntropyLoss()

#优化器
trainer=torch.optim.SGD(net.parameters(),lr=0.1) #在SGD传入所有参数

def load_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4))

batch_size=256
train_iter,test_iter=load_fashion_mnist(batch_size)

if __name__=="__main__":
    num_epochs=10
    train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)



