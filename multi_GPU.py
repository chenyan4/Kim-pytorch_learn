# 多GPU：数据并行、模型并行、通道并行（数据+模型并行）
# 数据并行：将小批量分成n块，每个GPU拿到完整参数计算一块的梯度
# 模型并行：将模型分成n块，每个GPU拿到一块模型计算它的前向和方向结果（适合模型大到单GPU放不下）

import torch
from torch import nn
from torch.nn import functional as F

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

loass=nn.CrossEntropyLoss(reduction='none') # reduction='none'表示不在内部做平均或求和，返回样本各自损失，[batch_size]

# 将参数放到GPU上的函数
def get_params(params,device):
    new_params=[p.clone().to(device) for p in params] #将模型参数分别放到不同GPU上,clone保证复制一份
    for param in new_params:
        param.requires_grad
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

output=parallel_scatter(data,devices)
print(output)

        


