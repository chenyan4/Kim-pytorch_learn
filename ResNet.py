# ResNet 残差网络
# 加更多层不一定更好，也不一定离最优解更近
# 如果大模型能包含小模型，至少不会变差，加层不会变差
# 输出f(x)=x+g(x),g(x)是残差，还是能得到原来的x
# ResNet块，一个高宽减半，通道翻倍的ResNet块（x走捷径，要过一个1×1卷积，步幅为2，实现高宽减半，通道翻倍）；后接几个高宽相同的ResNet块（可能捷径就跟1×1卷积，步幅为1）
# 梯度消失：乘法变加法

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,datasets
from matplotlib import pyplot as plt

class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1conv:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU(inplace=True)  # inplace=True表示直接在输入张量上做运算，不重新分配一个张量

    def forward(self,X):
        Y=self.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        Y=self.relu(Y)
        return Y

blk=Residual(3,3)
blk=Residual(3,6,use_1conv=True,strides=2)
X=torch.rand(4,3,6,6)
Y=blk(X)
print(Y.shape)

b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block :
            blk.append(Residual(input_channels,num_channels,use_1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))

x=torch.rand(size=(1,1,224,224))
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,"output_shape:",x.shape)

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    train_data=datasets.FashionMNIST('./data',train=True,transform=trans,download=False)
    test_data=datasets.FashionMNIST('./data',train=False,transform=trans,download=False)
    return (data.DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True),
    data.DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4,pin_memory=True))

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(dim=1)
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).sum().item()
    return cmp

def evaluate_accuracy(net,data_iter,device):
    if isinstance(net,nn.Module):
        net.eval()
    acc_num,num=0,0
    for x,y in data_iter:
        if isinstance(x,list):
            x=[a.to(device) for a in x]
        else:
            x=x.to(device)
        y=y.to(device)
        y_hat=net(x)
        acc_num+=accuracy(y_hat,y)
        num+=len(x)
    return acc_num/num



def train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,device=None):
    if device is None:
        device=next(iter(net.parameters())).device
    if isinstance(net,nn.Module):
        net.to(device)
    train_acc,train_loss,test_acc=[],[],[]
    for epoch in range(num_epochs):
        acc_sum,l_sum,num=0,0,0
        for x,y in train_iter:
            if isinstance(x,list):
                x=[a.to(device) for a in x]
            else:
                x=x.to(device)
            y=y.to(device)

            updater.zero_grad()
            y_hat=net(x)
            l=loss(y_hat,y)
            l.backward()
            updater.step()

            acc_sum+=accuracy(y_hat,y)
            l_sum+=l.item()
            num+=len(x)
        
        train_acc.append(acc_sum/num)
        train_loss.append(l_sum/num)
        test_acc.append(evaluate_accuracy(net,test_iter,device))

        print(f'epoch:{epoch+1},acc:{train_acc[-1]},loss:{train_loss[-1]},test_acc:{test_acc[-1]}')

    return train_acc,train_loss,test_acc

def draw_loss_acc(train_acc,train_loss,test_acc):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label="train_loss",color="blue",linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_acc,label="train_acc",color="red",linestyle='-',linewidth=2)
    plt.plot(test_acc,label="test_acc",color="green",linestyle='--',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Acc Curve")
    plt.legend("upper right")

    plt.savefig("/workspace/Kim-pytorch_learn/data/output/ResNet.png",dpi=300)
    plt.show()

def init_weights(m):
    if type(m)==nn.Conv2d or type(m) ==nn.Linear:
        nn.init.normal_(m.weight,0,0.01)

net.apply(init_weights)



if __name__=="__main__":
    lr,num_epochs,batch_size=0.0005,10,256
    train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=96)
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    train_acc,train_loss,test_acc=train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,"cuda:0")
    draw_loss_acc(train_acc,train_loss,test_acc)




