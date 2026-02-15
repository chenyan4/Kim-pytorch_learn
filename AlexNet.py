# AlexNet模型 更深更大的LeNet
# 进入丢弃法、RelU、最大池化层

import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
from matplotlib import pyplot as plt


net=nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1),nn.ReLU(),# 96**54*54
    nn.MaxPool2d(kernel_size=3,stride=2),# 96*26*26
    nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),nn.ReLU(), # 256*26*26
    nn.MaxPool2d(kernel_size=3,stride=2),# 256*12*12
    nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),nn.ReLU(),# 384*12*12
    nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),# 256*5*5
    nn.Flatten(),
    nn.Linear(6400,4096),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)

x=torch.randn(1,1,224,224)
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,':',x.shape)

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[]
    if resize:
        trans=[transforms.Resize(resize)]
    trans.append(transforms.ToTensor())
    trans=transforms.Compose(trans) # compose类似集成 ，Compose([....])
    mnist_train=datasets.FashionMNIST(root='./data',train=True,transform=trans,download=False)
    mnist_test=datasets.FashionMNIST(root='./data',train=False,transform=trans,download=False)
    # 优化：使用多进程加载数据，pin_memory加速GPU传输
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True),
            data.DataLoader(mnist_test,batch_size,shuffle=False,drop_last=True,num_workers=4,pin_memory=True))

batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=224)

def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight,0,0.01)

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(dim=1)
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).sum().item() # item()只能作用于0维数组，单个元素，转换为python数值类型
    return cmp

def evaluate_accuracy(net,data_iter,device):
    if isinstance(net,nn.Module):
        net.eval()
    acc,num=0,0
    for x,y in data_iter:
        if isinstance(x,list):
            x=[a.to(device) for a in x] # 多输入，[image, metadata],图像+文本
        else:
            x=x.to(device)
        y=y.to(device)
        acc+=accuracy(net(x),y)
        num+=len(x)
    return acc/num


def train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,device=None):
    if device is None:
        device=next(iter(net.parameters())).device
    if isinstance(net,nn.Module):
        net.to(device)
    train_loss,train_acc,test_acc=[],[],[]
    for epoch in range(num_epochs):
        l_sum,acc_sum,num=0,0,0
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
            l_sum+=l.item()
            acc_sum+=accuracy(y_hat,y)
            num+=len(x)
        
        test_acc.append(evaluate_accuracy(net,test_iter,device))

        train_loss.append(l_sum/1000)
        train_acc.append(acc_sum/num)

        print(f'epoch:{epoch+1} , train_loss:{train_loss[-1]:.4f} , train_acc:{train_acc[-1]:.4f} , test_acc:{test_acc[-1]:.4f}')
    
    return train_loss,train_acc,test_acc

def draw_loss_acc(train_loss,train_acc,test_acc):
    plt.figure(figsize=(12,4))
    plt.plot(train_loss,color='blue',label='train_loss',linestyle='-',linewidth=2) # linestyle='-'实线，'--'虚线，':'点线，'-.'点划线
    plt.plot(train_acc,color='red',label='train_acc',linestyle='--',linewidth=2)
    plt.plot(test_acc,color='green',label='test_acc',linestyle='-.',linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Loss/Accuracy Curve')
    plt.legend(loc='upper right')
    plt.savefig('/data/chenyan/pytoch_learn/data/loss_acc_curve.png',dpi=300)
    plt.show()


if __name__=='__main__':
    net.apply(init_weights)
    num_epochs,lr=10,0.1
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    train_loss,train_acc,test_acc=train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,'cuda:1')
    draw_loss_acc(train_loss,train_acc,test_acc)