import torch
from torch import nn
from dropout import load_data_fashion_mnist,train_ch3

dropout1,dropout2=0.2,0.5
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(dropout1),nn.Linear(256,256)
                  ,nn.ReLU(),nn.Dropout(dropout2),nn.Linear(256,10))

def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weight)

num_epochs,lr,batch_size=10,0.5,256
train_iter,test_iter=load_data_fashion_mnist(batch_size)


loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(params=net.parameters(),lr=lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

