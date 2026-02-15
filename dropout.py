#丢弃法
#一个好的模型需要对输入数据的扰动鲁棒
#丢弃法：在层之间加入噪音,只在训练中使用
#对x加入噪音得到m，我们希望期望E[m]=x
#丢弃法对每个元素进行扰动:m=0(p的概率),x/(1-p)(1-p的概率),也就是随机把你置零或放大

import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms,datasets

def Softmax(X):
    X_exp=torch.exp(X)
    paration=torch.sum(X_exp,dim=1,keepdim=True)
    return X_exp/paration
def dropout_layer(X,dropout):
    assert 0<=dropout<=1 #表示dropout要满足条件
    if dropout==1:
        return torch.zeros(X.shape)
    if dropout==0:
        return X
    mask=(abs(torch.randn(size=X.shape))>dropout).float() #bool转成float,torch.randn随机生成服从均值为0，标准差为1的张量
    return (mask*X)/(1.0-dropout)

x=torch.arange(16,dtype=torch.float32).reshape((4,4))

# torch.randn(*size) —— 标准正态分布
# 分布：N(0, 1)，均值 0，标准差 1。
# 用法：torch.randn(2,3)、torch.randn(size=(2,3))。
# 典型用途：初始化权重、生成噪声、做 dropout 的随机源（如你代码里那样）

# torch.rand(*size) —— [0, 1) 均匀分布
# 分布：在 [0, 1) 上均匀。
# 用途：做 严格概率 的 dropout mask、随机比例、随机采样等。

# torch.randperm(n) —— 随机排列
# 含义：0 到 n-1 的一个随机排列。
# 用途：打乱样本顺序、随机选下标。
# torch.randperm(5)   # 例如 tensor([2, 0, 4, 1, 3])

# torch.randint(low, high, size) —— 离散均匀整数
# 含义：在 [low, high) 内按均匀分布取整数，形状由 size 指定。
# 用途：随机索引、随机类别、随机长度等
# torch.randint(0, 10, (3, 4))   # 形状 (3,4)，值在 [0, 10)
# torch.randint(5, (2, 3))       # 等价于 low=0, high=5

# print(dropout_layer(x,0.5))

#定义两个多层感知机
num_inputs,num_outputs,num_hidden1,num_hidden2=784,10,256,256

dropout1,dropout2=0.2,0.5
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hidden1,num_hidden2,is_training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_training
        self.l1=nn.Linear(num_inputs,num_hidden1)
        self.l2=nn.Linear(num_hidden1,num_hidden2)
        self.l3=nn.Linear(num_hidden2,num_outputs)
        self.relu=nn.ReLU()

    def forward(self,x):
        h1=self.relu(self.l1(x.reshape((-1,self.num_inputs))))
        if self.training:
            h1=dropout_layer(h1,dropout1) #只针对隐藏层操作
        h2=self.relu(self.l2(h1))
        if self.training:
            h2=dropout_layer(h2,dropout2)
        out=self.l3(h2)
        return out

    def is_Train(self):
        self.training=True

    def is_Test(self):
        self.training=False


net=Net(num_inputs,num_outputs,num_hidden1,num_hidden2)

def load_data_fashion_mnist(batch_size):
    trans=transforms.Compose([transforms.ToTensor()])
    train_data=datasets.FashionMNIST('../data',train=True,transform=trans,download=True)
    test_data=datasets.FashionMNIST('../data',train=False,transform=trans,download=True)
    return (data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True),data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False))

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1)
    cmp=y_hat==y.reshape(y_hat.shape)
    cmp=cmp.float()
    return cmp.sum()

def train(net,train_iter,loss,updater):
    # net.is_Train()
    if isinstance(net,nn.Module):
        net.train()
    out_list=[]
    for x,y in train_iter:
        updater.zero_grad()
        output=Softmax(net(x))
        l=loss(output,y)
        l.backward()
        out_list.append((accuracy(output,y),l.sum(),y.numel()))
        updater.step()

    acc,num,los=0,0,0
    for i,j,k in out_list:
        acc+=i
        los+=j
        num+=k

    return (acc/num,los/num)

def evaluate_test(net,test_iter):
    # net.is_Test()
    if isinstance(net,nn.Module):
        net.eval()
    acc_list=[]
    for x,y in test_iter:
        with torch.no_grad():
            output=Softmax(net(x))
            acc_list.append((accuracy(output,y),y.numel()))
    acc,num=0,0
    for i,j in acc_list:
        acc+=i
        num+=j
    return acc/num

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_acc,train_loss=train(net,train_iter,loss,updater)
        test_acc=evaluate_test(net,test_iter)
        print(f'epoch:{epoch+1} , train_acc:{train_acc} , train_loss:{train_loss} , test_acc:{test_acc}')

if __name__=="__main__":
    num_epochs,lr,batch_size=10,0.5,256
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(params=net.parameters(),lr=lr)
    train_iter,test_iter=load_data_fashion_mnist(batch_size)
    train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)






