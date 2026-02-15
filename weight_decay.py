#权重衰减，限制权重

#使用均方范数作为硬性限制，通常不限制b，||w||^2<Θ,更小意味着更强的正则项

#使用均方范数作为柔性限制,min(f(w,b))+(λ/2*||w||^2),超参数m控制正则项重要程度
#L2正则项使得模型参数不会过大，从而控制模型的负责都，正则项权重是控制模型复杂度的参数数，即上面的λ
#其实在优化参数时，Wt+1=(1-lr*λ)*Wt-导(loss(w,b)/w),相当于在减去梯度前，先缩小一次W

import torch
from torch.utils import data
from torch import nn

n_train,n_test,num_inputs,batch_size=20,100,200,5
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.5 #y=0.05+0.01X1+...+0.01Xn

def synthetic_data(w,b,num_example):
    x=torch.normal(0,1,(num_example,num_inputs))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y

def load_array(data_array,batch_size,is_Train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_Train)

train_data,train_label=synthetic_data(true_w,true_b,n_train)
test_data,test_label=synthetic_data(true_w,true_b,n_test)
train_iter=load_array((train_data,train_label),batch_size,True)
test_iter=load_array((test_data,test_label),batch_size,False)

def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return (w,b)

def l2_penalty(w):
    return torch.sum(pow(w,2))/2 #相当于λ是1

def net(X,w,b):
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    l=pow(y_hat-y,2)/2
    return l

def SGD(params,lr,batch_size): #更新参数不要计算梯度
    with torch.no_grad():
        for param in params:
            param-=lr*(param.grad/batch_size)
            param.grad.zero_()

def evaluate_loss(net,test_d):
    if isinstance(net,nn.Module):
        net.eval()
    metric=[]
    for x,y in test_d:
        l=squared_loss(net(x),y)
        metric.append((l.sum(),y.numel()))
    l_sum=0
    y_sum=0
    for i,j in metric:
        l_sum+=i
        y_sum+=j
    return l_sum/y_sum



# def train(lambd):
#     w,b=init_params()
#     num_epochs,lr=100,0.003
#     metric=[]
#     for epoch in range(num_epochs):
#         for x,y in train_iter:
#             with torch.enable_grad():
#                 l=squared_loss(net(x,w,b),y)+l2_penalty(w)*lambd
#                 l.sum().backward()
#                 SGD([w,b],lr,len(x))
#
#         if (epoch+1)%5==0:
#             print(f'epoch:{epoch+1} , test_loss:{evaluate_loss(net,test_iter,w,b)} , {evaluate_loss(net,train_iter,w,b)}')
#     print('w的L2范数是:',torch.norm(w).item())

#简洁实现
def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters(): #要循环遍历取里面的权重值
        # print(param)
        param.data.normal_() #正态分布
        # print(param)
    loss=nn.MSELoss()
    num_epochs,lr=100,0.003
    # print(net[0].weight) 可以取出神经网络第0层的权重
    # print(net[0].bias) 取出第零层偏置
    #可以自己定义要更新的参数，放在一个列表，一个字典是一个参数
    trainer=torch.optim.SGD([{"params":net[0].weight,'weight_decay':wd},{"params":net[0].bias}],lr=lr) #也就是net.parameter()其实是一个列表，里面是各层参数的一个字典,对于参数基本都会提供一个weight_decay选项
    for epoch in range(num_epochs):
        for x,y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l=loss(net(x),y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            print(
                f'epoch:{epoch + 1} , test_loss:{evaluate_loss(net, test_iter)} , {evaluate_loss(net, train_iter)}')
    print('w的L2范数是:', torch.norm(net[0].weight).item())
if __name__=="__main__":
    # train(0)
    train_concise(3)




