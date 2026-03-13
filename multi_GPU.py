# 多GPU：数据并行、模型并行、通道并行（数据+模型并行）
# 数据并行：将小批量分成n块，每个GPU拿到完整参数计算一块的梯度
# 模型并行：将模型分成n块，每个GPU拿到一块模型计算它的前向和方向结果（适合模型大到单GPU放不下）

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scale=0.01
w1=torch.randn(size=(20,1,3,3))*scale
b1=torch.zeros(20)
w2=torch.randn(size=(50,20,5,5))*scale
b2=torch.zeros(50)
w3=torch.randn(size=(800,128))*scale
b3=torch.zeros(128)
w4=torch.randn(size=(128,10))*scale
b4=torch.zeros(10)
params=[w1,b1,w2,b2,w3,b3,w4,b4]

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)

    train_data=datasets.FashionMNIST('./data',train=True,transform=trans,download=False)
    test_data=datasets.FashionMNIST('./data',train=False,transform=trans,download=False)

    return (DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4),
    DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4))


def lenet(X,params):
    h1_conv=F.conv2d(input=X,weight=params[0],bias=params[1])
    h1_activation=F.relu(h1_conv)
    h1=F.avg_pool2d(input=h1_activation,kernel_size=(2,2),stride=2)
    h2_conv=F.conv2d(input=h1,weight=params[2],bias=params[3])
    h2_activation=F.relu(h2_conv)
    h2=F.avg_pool2d(input=h2_activation,kernel_size=2,stride=2)
    h2=h2.reshape(h2.shape[0],-1)
    h3_linear=torch.matmul(h2,params[4])+params[5]
    h3_activation=F.relu(h3_linear)
    Y_hat=torch.matmul(h3_activation,params[6])+params[7]
    return Y_hat

loss=nn.CrossEntropyLoss(reduction='none') # reduction='none'表示不在内部做平均或求和，返回样本各自损失，[batch_size]

# 将参数放到GPU上的函数


def get_params(params,device):
    new_params=[p.clone().to(device) for p in params] #将模型参数分别放到不同GPU上,clone保证复制一份
    for param in new_params:
        param.requires_grad_(True)
    return new_params

new_params=get_params(params,'cuda:0')
print("b1 weight:",new_params[1])
print("b1 grad:",new_params[1].grad)

def allreduce(data):
    # 把所有GPU的结果在第一个 GPU上做相加,因为计算要在同一个 GPU 上
    for i in range(1,len(data)):
        data[0]+=data[i].to(data[0].device) #写成data[0][:],如果别处引用data[0],则会一并更新
    # 再把结果返回 其他 GPU中
    for i in range(1,len(data)):
        data[i]=data[0].to(data[i].device)

g0=torch.tensor([1.0,2.0],device="cuda:0")
g1=torch.tensor([3.0,4.0],device="cuda:1")

data=[g0,g1]
allreduce(data)
print(data[0],data[1])

data=torch.arange(20).reshape(4,5)
devices=[torch.device('cuda:0'),torch.device('cuda:1')]
# 可以用nn.parallel.scatter(data,devices) 进行均匀分割
def parallel_scatter(data,devices):
    num_device=len(devices)
    num_data=len(data)
    output=[]
    split=(num_data+num_device-1)//num_device
    for i in range(num_device):
        max_line=(i+1)*split
        if max_line>num_data:
            output.append(data[(i*split):num_data].to(devices[i]))
        else:
            output.append(data[(i*split):((i+1)*split)].to(devices[i]))
    return tuple(output)

def split_batch(X,y,devices):
    assert X.shape[0]==y.shape[0]
    return (parallel_scatter(X,devices),parallel_scatter(y,devices))

def SGD(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*(param.grad/batch_size)
            param.grad.zero_()

output=parallel_scatter(data,devices)
print(output)



def train_batch(X,y,device_params,devices,lr):
    X_splits,y_splits=split_batch(X,y,devices)
    ls=[loss(lenet(X_split,device_param),y_split).sum() for X_split,y_split,device_param in zip(X_splits,y_splits,device_params)]

    # 计算每个小批量上的梯度
    for l in ls:
        l.backward()
    
    with torch.no_grad():
        # 对params中的梯度逐一做 求和和广播
        for i in range(len(device_params[0])):
            # 拿去每个device_params 的w1、b1、w2等的梯度
            allreduce([device_params[c][i].grad for c in range(len(device_params))])
    
    # 对每个 device_params进行梯度更新：
    for param in device_params:
        SGD(param,lr,X.shape[0])

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1)
    y=y.reshape(y_hat.shape)
    cmp=(y==y_hat).sum().item()
    return cmp

def evaluate_accuracy(params,test_iter):
    device=params[0].device
    acc_num,num=0,0
    with torch.no_grad():
        for x,y in test_iter:
            x=x.to(device)
            y=y.to(device)
            y_hat=lenet(x,params)
            acc_num+=accuracy(y_hat,y)
            num+=len(x)
        return acc_num/num

def train(num_gpus,batch_size,lr):
    train_iter,test_iter=load_data_fashion_mnist(batch_size)
    devices=[f'cuda:{i}' for i in range(num_gpus)]
    device_params=[get_params(params,device) for device in devices]
    num_epochs=10
    test_acc=[]
    for epoch in range(num_epochs):
        for x,y in train_iter:
            train_batch(x,y,device_params,devices,lr)
            torch.cuda.synchronize()
        test_acc.append(evaluate_accuracy(device_params[0],test_iter))
        print(f'epoch:{epoch+1},test_acc:{test_acc[-1]},')
    
    return test_acc

def draw_acc(test_acc,name):
    plt.figure(figsize=(12,4))
    plt.plot(test_acc,label="test_acc",color="blue",linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Acc Curve")
    plt.legend(loc="upper right")
    
    plt.savefig(f"/data/chenyan/pytorch_learn/data/output/{name}.png",dpi=300)
    plt.show()

if __name__=="__main__":
    # 保证每个GPU拿到相同batch_size,同时可以增大lr
    test_acc=train(2,256*2,lr=0.2*2)
    draw_acc(test_acc)



        


