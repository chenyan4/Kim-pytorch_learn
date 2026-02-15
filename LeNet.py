# LeNet：卷积层学习图片空间信息
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.reshape(shape=(-1,1,28,28))

net=torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1,6,kernel_size=5,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(), # 保留batch_size维度
    nn.Linear(16*5*5,120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10)
)


x=torch.randn(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,':',x.shape)

print("开始加载数据集...")
batch_size=256
trans=transforms.Compose([transforms.ToTensor()])
print("正在加载训练集...")
mnist_train=torchvision.datasets.FashionMNIST(root='./data',train=True,transform=trans,download=True)
print(f"训练集加载完成，样本数: {len(mnist_train)}")
print("正在加载测试集...")
mnist_test=torchvision.datasets.FashionMNIST(root='./data',train=False,transform=trans,download=True)
print(f"测试集加载完成，样本数: {len(mnist_test)}")

print("创建数据加载器...")
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter=data.DataLoader(mnist_test,batch_size,shuffle=False) # drop_last表示是否丢弃不完整batch
print("数据加载器创建完成")

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(dim=1)  # PyTorch标准API使用dim而不是axis
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).sum().item()  # 转换为Python数值类型，便于累加和计算
    return cmp

def evaluate_accuracy(net,data_iter,device=None):
    if isinstance(net,torch.nn.Module):
        net.eval()
        if not device:
            # net.parameters()返回迭代器，iter()获取第一个迭代器，第一个convd，next()返回第一个迭代器的第一个元素
            device=next(iter(net.parameters())).device
        acc=0
        num=0
        for x,y in data_iter:
            if isinstance(x,list):
                x=[a.to(device) for a in x] #多输入问题
            else:
                x=x.to(device)
            y=y.to(device)
            batch_acc=accuracy(net(x),y)  # 只计算一次，避免重复计算
            acc+=batch_acc
            num+=y.numel()
        return acc/num

def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight)
        
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    net.apply(init_weights)
    print("train on",device)
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        acc=0
        num=0
        l_num=0
        for x,y in train_iter:
            x,y=x.to(device),y.to(device)
            y_hat=net(x)
            optimizer.zero_grad()
            l=loss(y_hat,y)
            l_num+=l.item()  # CrossEntropyLoss返回标量，使用.item()获取Python数值
            l.backward()
            optimizer.step()
            acc+=accuracy(y_hat,y)
            num+=len(y)
        
        test_acc=evaluate_accuracy(net,test_iter)
        print(f'epoch:{epoch+1},train_acc:{(acc/num):.4f},train_loss:{(l_num/num):.4f},test_acc:{test_acc:.4f}')
            
lr,num_epochs,device=0.9,10,'cuda:0'  # 降低学习率，0.9太大导致训练不稳定
print(f"准备开始训练，lr={lr}, epochs={num_epochs}, device={device}")
train_ch6(net,train_iter,test_iter,num_epochs,lr,device)
print("训练完成")