import torch
from torch import nn
from softmax_achieve import load_fashion_mnist
from softmax_achieve import train_ch3

batch_size=256
train_iter,test_iter=load_fashion_mnist(batch_size)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights) #net其实相当于一个列表了，从里面取layer执行初始化函数，类似map

loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.1)

if __name__=="__main__":
    num_epochs=10
    train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
