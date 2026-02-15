#加州房价预测
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#数据集准备
train_path=os.path.join('califor_price','train.csv')
test_path=os.path.join('califor_price','test.csv')

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)
# print(train_data.shape)
# print(test_data.shape)

train_features=train_data.iloc[:,4:]
test_features=test_data.iloc[:,3:]
train_labels=train_data.iloc[:,2]
# print(train_features)

numeric_features=train_features.dtypes[train_features.dtypes!='object'].index
# print(numeric_features)

# def to_zhengtai(data_frame,index):
#     for e in index:
#         data_frame[e]=(data_frame[e]-data_frame[e].mean())/(data_frame[e].std())
#     return data_frame

train_features[numeric_features]=train_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
test_features[numeric_features]=train_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
train_features[numeric_features]=train_features[numeric_features].fillna(0)
test_features[numeric_features]=test_features[numeric_features].fillna(0)
train_select=train_features['State']
test_select=test_features['State']
train_features=train_features[numeric_features]
test_features=test_features[numeric_features]
train_features=pd.concat((train_features,train_select),axis=1)
test_features=pd.concat((test_features,test_select),axis=1)
train_features=pd.get_dummies(train_features,dummy_na=True)
test_features=pd.get_dummies(test_features,dummy_na=True)


# print(train_features.shape)
# print(test_features.shape)
# print(numeric_features)

#准备数据集
train_features=torch.tensor(train_features.values.astype(float),dtype=torch.float32)
train_labels=torch.tensor(train_labels.values.astype(float),dtype=torch.float32)
test_features=torch.tensor(test_features.values.astype(float),dtype=torch.float32)

def load_array(data_array,batch_size):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=True)

loss=nn.MSELoss()

def log_rmse(net,data_iter):
    rmse_l=0
    with torch.no_grad():
        for x,y in data_iter:
            output=net(x)
            output=torch.clamp(output,1,float('inf'))
            rmse_l+=torch.sqrt(loss(torch.log(output),torch.log(y.reshape(output.shape))))
    return rmse_l/len(data_iter)




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(
            nn.Linear(21,64),
            nn.ReLU(),
            nn.Linear(64,1),

        )

    def forward(self,x):
        return self.f(x)

def get_net():
    net=Net()
    return net

def train(net,num_epochs,train_features,train_labels,test_features,test_labels,learning_rate,weight_decay,batch_size):
    if isinstance(net,nn.Module):
        net.train()
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    train_iter=load_array((train_features,train_labels),batch_size)
    if test_features is not None:
        test_iter=load_array((test_features,test_labels),batch_size)

    train_ls=[]
    test_ls=[]
    for epoch in range(num_epochs):
        if epoch % 10 == 0:  # 每10个epoch打印一次进度
            print(f'Epoch {epoch}/{num_epochs}')
        for x,y in train_iter:
            optimizer.zero_grad()
            output=net(x)
            l=loss(output,y.reshape(output.shape))
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_iter))
        if test_features is not None:
            test_ls.append(log_rmse(net,test_iter))

    return train_ls,test_ls

def k_fold_data(k,i,x,y):
    assert k>1
    x_train,y_train=None,None
    fold_size=x.shape[0]//k
    for j in range(k):
        start,end=j*fold_size,(j+1)*fold_size
        x_temp,y_temp=x[start:end,:],y[start:end]
        if j==i:
            x_valid,y_valid=x_temp,y_temp
        elif x_train is None:
            x_train,y_train=x_temp,y_temp
        else:
            x_train=torch.cat((x_train,x_temp),dim=0)
            y_train=torch.cat((y_train,y_temp),dim=0)
    return x_train,y_train,x_valid,y_valid

def k_fold(k,num_epochs,train_data,train_label,learning_rate,weight_decay,batch_size):
    train_l_sum=0
    test_l_sum=0
    net=get_net()
    for i in range(k):
        print(f'Starting fold {i+1}/{k}...')
        train_f,train_l,test_f,test_l=k_fold_data(k,i,train_data,train_label)
        train_ls,test_ls=train(net,num_epochs,train_f,train_l,test_f,test_l,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        test_l_sum+=test_ls[-1]
        if i==0:
            plt.figure()
            plt.plot(train_ls,label='train_loss')
            plt.plot(test_ls,label='test_loss')
            plt.legend(loc='upper right')
            plt.xlabel('Epoch:')
            plt.ylabel('loss:')
            plt.savefig('loss_curve.png')  # 保存图片而不是显示，避免阻塞
            print('Loss curve saved to loss_curve.png')
        print(f'fold:{i} , train_loss:{train_ls[-1]} , test_loss:{test_ls[-1]}')
    return train_l_sum/k,test_l_sum/k

def train_and_predict(num_epochs,train_data,train_labels,test_data,learning_rate,weight_decay,batch_size):
    net=get_net()
    train_ls,_=train(net,num_epochs,train_data,train_labels,None,None,learning_rate,weight_decay,batch_size)
    print(f'train_loss:{train_ls[-1]}')
    pred=net(test_data).detach().numpy()
    ans=dict()
    ans['Id']=range(47439,len(pred)+47439)
    ans['Sold Price']=pred.reshape(-1).astype(int)
    ans=pd.DataFrame(ans)
    ans.to_csv('data/sub_ans.csv',index=False)
    # print(pred)


if __name__=="__main__":
    k,num_epochs,lr,weight_decay,batch_size=10,100,0.1,0,256
    # train_sum_l,test_sum_l=k_fold(k,num_epochs,train_features,train_labels,lr,weight_decay,batch_size)
    # print(f'train_loss:{train_sum_l} , test_loss:{test_sum_l}')
    train_and_predict(num_epochs,train_features,train_labels,test_features,lr,weight_decay,batch_size)





