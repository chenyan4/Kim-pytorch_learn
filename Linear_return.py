#线性回归，房价预测

#1.假设影响房价因素是卧室个数、卫生间个数和居住面积，x1、x2、x3
#向量版本：y=<w,x>+b,线性模型可以看作单层神经网络

import random
import torch

#1.构建简易数据集
def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w))) #均值为0，方差为1，行数为样本数，列数为len(w)
    y=torch.matmul(x,w)+b #一维张量默认为行向量，但相乘时依然可以
    y+=torch.normal(0,0.1,y.shape) #随机噪音
    return x,y.reshape((num_examples,1)) #y.reshape((-1,1))也行，变成列向量，因为一维张量默认为行向量

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000) #样本和标签，训练集

#2.定义批量大小
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    #样本随机读取
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices] #每次循环都返回样本的一个batch,相当于我得到样本标号集合，通过feature[集合]可以取相应下标的张量

# batch_size=5
# for x,y in data_iter(batch_size,features,labels):
#     print(x,'\n',y)

#3.定义模型初始化参数
w=torch.normal(0,0.01,(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#4.定义模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b

#5.定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y)**2/2

#6.定义优化算法
def SGD(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            print(param.grad)
            param-=lr*(param.grad/batch_size) #反向求导其实就是
            param.grad.zero_() #梯度清零

lr=0.05
num_epochs=10

for epoch in range(num_epochs):
    for x,y in data_iter(3,features,labels):
        output=linreg(x,w,b) #也是一个3维大小列向量
        loss=squared_loss(y,output) #此时的loss其实还是一个batch_size大小的列向量
        loss.sum().backward() #要把3个样本的损失相加再反向求梯度
        SGD([w,b],lr,3)
    with torch.no_grad():
        output=linreg(features,w,b)
        test_l=squared_loss(labels,output)
        print(f'epoch:{epoch+1},loss:{float(test_l.sum())}')
print(f'w的误差:{abs(true_w.reshape(w.shape)-w)}')
print(f'b的误差:{abs(true_b-b)}')