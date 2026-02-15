import numpy
import torch
from torch.utils import data

def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w))) #均值为0，方差为1，行数为样本数，列数为len(w)
    y=torch.matmul(x,w)+b #一维张量默认为行向量，但相乘时依然可以
    y+=torch.normal(0,0.1,y.shape) #随机噪音
    return x,y.reshape((num_examples,1)) #y.reshape((-1,1))也行，变成列向量，因为一维张量默认为行向量

#准备人工数据集
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

#构造一个pytorch数据迭代器
def load_arrays(data_arrays,batch_size,is_train=True):
    #print(data_arrays)
    dataset=data.TensorDataset(*data_arrays)
    print(dataset.tensors==data_arrays) #dataset只是一个地址，dataset.tensors存的是数据
    return data.DataLoader(dataset,batch_size,shuffle=is_train) #从dataset选batch_size份组成一组

batch_size=10
data_iter=load_arrays((features,labels),batch_size)
next(iter(data_iter)) #相当于把data_iter中的内容显示转化，通过next函数转为python的iter格式
# for x,y in data_iter: 输出形式为batch_size大小的数据+标签组合，每一个组合被放在一个列表
#     print(x,'\n',y)

#定义模型
from torch import nn
net=nn.Sequential(nn.Linear(2,1)) #输入维度是2，输出维度是1,Sequential相当于list of layers

#初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0) #可以用数组下标访问

#定义误差损失,均方误差
loss=nn.MSELoss()

#实例化优化器（SGD）
trainer=torch.optim.SGD(net.parameters(),lr=0.03) #第一个是指定所有参数

#训练模块
num_epoch=3
for epoch in range(num_epoch):
    for x,y in data_iter:
        y_hat=net(x)
        l=loss(y_hat,y)
        trainer.zero_grad() #梯度清零要不然会累加
        l.backward() #反向求导
        trainer.step() #参数更新
    with torch.no_grad(): #不求梯度
        l=loss(net(features),labels)
        print(f'epoch:{epoch+1},loss:{l:f}')

