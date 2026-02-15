# 模型选择
# 训练误差：模型在训练数据集上的误差
# 泛化误差：模型在新数据上的误差
# 在过去的考试表现很好（训练误差）不代表未来考试一定会好（泛化误差）
# 验证数据集：一个用来评估模型好坏的数据集（测试数据集不能拿来调超参，泛化不好）
# 测试数据集

# 如果没有那么多数据集（常态）
# K-折交叉验证，将数据集分割为k块，每次使用第i块作为验证数据集，其余的作为训练数据集

#过拟合与欠拟合
#模型容量：拟合各种函数的能力，低容量的模型难以拟合训练数据；高容量的模型可以记住所有的训练数据
#首先模型先足够大，其次再考虑如何降拟合性
#vc维：评判模型好坏的理论依据

#数据复杂度：样本个数，每个样本的元素个数，时间和空间结构（视频），多样性
#模型容量需要匹配数据复杂度，否则会过拟合或欠拟合

#欠拟合
import math
import numpy as np
import torch
from torch import nn
from torch.utils import data

max_degree=20 #比如处理数据的最大维度
n_train,n_test=100,100
true_w=np.zeros(max_degree)
true_w[0:4]=np.array([5,1.2,-3.4,5.6]) #相当于一个多项式，y=5+1.2x-(3.4x^2/2!)+(5.6x^3/3!)

features=np.random.normal(size=(n_train+n_test,1))
np.random.shuffle(features)
poly_features=np.power(features,np.arange(max_degree)) #广播机制
# print(poly_features)
for i in range(1,max_degree):
    poly_features[:,i]/=math.gamma(i)
# print(poly_features)
labels=np.matmul(poly_features,true_w)
labels+=np.random.normal(scale=0.1,size=labels.shape) #和实际值产生些偏差，噪音

true_w,features,poly_features,labels=[torch.tensor(x,dtype=torch.float32) for x in [true_w,features,poly_features,labels]]
print(features[:2])
print(poly_features[:2,:])

#评估网络
def evaluate_loss(net,data_iter,loss):
    if isinstance(net,nn.Module):
        net.eval()
    metric=[]
    for x,y in data_iter:
        output=net(x)
        y=y.reshape(output.shape)
        l=loss(output,y)
        metric.append((l.sum(),y.numel()))
    l_sum=0
    n_sum=0
    for i,j in metric:
        l_sum+=i
        n_sum+=j
    return l_sum/n_sum

def load_arrays(data_arrays,batch_size,is_train=True):
    #print(data_arrays)
    dataset=data.TensorDataset(*data_arrays)
    print(dataset.tensors==data_arrays) #dataset只是一个地址，dataset.tensors存的是数据
    return data.DataLoader(dataset,batch_size,shuffle=is_train) #从dataset选batch_size份组成一组

def train(train_features,test_features,train_labels,test_labels,num_epochs=400):
    loss=nn.MSELoss()
    input_shape=train_features.shape[-1]
    net=nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size=min(10,len(train_labels))

    train_iter=load_arrays((train_features,train_labels),batch_size,True)
    test_iter=load_arrays((test_features,test_labels),batch_size,False)

    trainer=torch.optim.SGD(net.parameters(),lr=0.01)
    metric=[]

    for epoch in range(num_epochs):
        if isinstance(net, nn.Module):
            net.train()
        out_list=[]
        for x,y in train_iter:
            trainer.zero_grad()
            output=net(x)
            l=loss(output,y.reshape(output.shape))
            l.backward()
            trainer.step()
            out_list.append(l.sum()/len(y))

        l_sum=0
        for e in out_list:
            l_sum+=e

        print(f'训练轮数:{epoch+1} , 损失值:{l_sum/len(out_list)}')

        print(f'测试集上损失:{evaluate_loss(net,test_iter,loss)}')

    print('weight:',net[0].weight.data.numpy())

#没有问题
# train(poly_features[:100,:4],poly_features[100:,:4],labels[:100],labels[100:],400)

#欠拟合，数据不全
# train(poly_features[:100,:2],poly_features[100:,:2],labels[:100],labels[100:])

#过拟合，数据全给你，权重最大维,维数过大了
train(poly_features[:100,:],poly_features[100:,:],labels[:100],labels[100:])






