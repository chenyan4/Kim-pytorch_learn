# 使用块的网络,VGG块，3×3卷积核，填充为1，n层 m通道，2×2最大池化层，步幅为2
# 主要思想就是 堆块，为什么不 5×5，因为 深但窄效果会好
# 不同次数的重复块得到不同架构，VGG-16，VGG-19
# 将ALexNet中间的不规则卷积操作，用块替代 ，更大更深AlexNet

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,datasets
from matplotlib import pyplot as plt

def vgg_block(num_convs,in_channels,out_channels):
    layers=[] 
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    # *layers *是解包操作符，将列表/元组的元素展开为独立参数
    return nn.Sequential(*layers)

conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512)) #(num_convs,out_channels)

def vgg(conv_arch):
    conv_blks=[]
    in_channels=1 # 看刚开始输入通道
    for num_convs,out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels

    return nn.Sequential(
        *conv_blks,nn.Flatten(),
        nn.Linear(in_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
    )

net=vgg(conv_arch)
x=torch.randn(size=(1,1,224,224))
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,":",x.shape)

def init_weights(m):
    # if type(m)==nn.Linear or type(m)==nn.Conv2d:
    #     nn.init.normal_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=datasets.FashionMNIST(root='./data',train=True,transform=trans,download=False)
    mnist_test=datasets.FashionMNIST(root='./data',train=False,transform=trans,download=False)
    return (data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True),
    data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4,pin_memory=True))

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1)
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).int().sum()
    return cmp.item()

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
        acc_num+=accuracy(net(x),y)
        num+=len(x)
    return acc_num/num

def train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,device=None):
    if device is None:
        device=next(iter(net.parameters())).device
    if isinstance(net,nn.Module):
        net.train()
    net.to(device)
    train_acc,train_loss,test_acc=[],[],[]
    for epoch in range(num_epochs):
        acc_num,loss_num,num=0,0,0
        for x,y in train_iter:
            if isinstance(x,list):
                x=[a.to(device) for a in x]
            else:
                x=x.to(device)
            y=y.to(device)
            y_hat=net(x)
            updater.zero_grad()
            l=loss(y_hat,y)
            l.backward()
            updater.step()
            acc_num+=accuracy(y_hat,y)
            loss_num+=l.item()
            num+=len(x)
        train_acc.append(acc_num/num)
        train_loss.append(loss_num)
        test_acc.append(evaluate_accuracy(net,test_iter,device))
        print(f'epoch:{epoch+1},train_acc:{train_acc[-1]},train_loss:{train_loss[-1]},test_acc:{test_acc[-1]}')
    return train_acc,train_loss,test_acc

def draw_loss_acc(train_acc,train_loss,test_acc,name=''):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label='train_loss',color='blue',linestyle='-',linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')

    plt.subplot(1,2,2)
    plt.plot(train_acc,label='train_acc',color='red',linestyle='-',linewidth=2)
    plt.plot(test_acc,label='test_acc',color='green',linestyle='--',linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve')
    plt.legend(loc='upper right')  

    plt.savefig(f'/data/chenyan/pytoch_learn/data/{name}_loss_acc.png',dpi=300)
    plt.show()  



ratio=4
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]
# small_conv_arch=[(num_convs,out_channels//ratio) for num_convs,out_channels in conv_arch]

small_net=vgg(small_conv_arch)
small_net.apply(init_weights)

if __name__=='__main__':
    batch_size=128
    train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=224)
    num_epochs,lr=10,0.005
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(small_net.parameters(),lr=lr)
    train_acc,train_loss,test_acc=train_ch6(small_net,train_iter,test_iter,loss,num_epochs,lr,updater,'cuda:1')
    draw_loss_acc(train_acc,train_loss,test_acc,name='VGG')