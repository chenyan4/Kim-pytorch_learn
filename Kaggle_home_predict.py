import numpy as np
import pandas as pd
import torch
from torch import nn
import os
from torch.utils import data
import matplotlib.pyplot as plt

train_path=os.path.join('data','train.csv')
test_path=os.path.join('data','test_toup.csv')
test_out_path=os.path.join('data','samples submission.csv')
train_data=pd.read_csv(filepath_or_buffer=train_path)
test_data=pd.read_csv(filepath_or_buffer=test_path)
test_out=pd.read_csv(filepath_or_buffer=test_out_path)
# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]]) #打印前四行，前四列和后三列

#第一列是id，去掉，同时train多一列预测房价
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
# print(all_features.iloc[0:4,-1])

numeric_features=all_features.dtypes[all_features.dtypes!='object'].index # dtype会打印所有列的列名和类型，而index则会取现有的所有列的名，[]内可以添加筛选
# x=all_features.iloc[:3] #x其实本质是一个字典，键值对
# for e in x: #for 取的是键
#     print(x[e])
# print(x[numeric_features].dtypes) #dtypes是打印每一列的元素名和类型
# x=x[numeric_features].apply(lambda x:print(x.mean())) # x取的是每一列的所有值,存在一个dataframe中，对一列所有元素求均值
# x=x[numeric_features].apply(lambda x:print(x[1])) #取每一列下标为1的元素
# x=x[numeric_features].apply(lambda x:print(x)) #打印每一列的所有元素
# x=x[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
# print(x)
all_features[numeric_features]=all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std())) #只是将数值类型的取均值除方差
all_features[numeric_features]=all_features[numeric_features].fillna(0) #填充缺失值为0，missing_data
all_features=pd.get_dummies(all_features,dummy_na=True) #get_dummies()处理离散值，data为处理数据，prefix为转换后列明前缀，columns指定需要实现类别转换的列名，dummy_na增加一列表示空缺值
# print(all_features.values.astype(float)) 转成numpy数组

train_features=torch.tensor(all_features.values.astype(float),dtype=torch.float32)
test_out=test_out.iloc[:,1:]
test_valid=torch.tensor(test_out.values.astype(float),dtype=torch.float32)
train_labels=pd.concat((train_data.iloc[:,-1],test_out))
train_labels=torch.tensor(train_labels.values.astype(float),dtype=torch.float32)

loss=nn.MSELoss()
in_features=train_features.shape[1]

def get_net():
    net=nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net,features,labels):
    #因为要用log，要保证值大于1，这时数是正的
    clipped_preds=torch.clamp(net(features),1,float('inf')) #clamp(data,min,max) 将张量数据压缩到[min,max]之间，超过的值就取min或max，看是超过上界还是下界
    # print(clipped_preds)
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels.reshape(clipped_preds.shape)))) #先取log后，再做均方损失后取平方根，降低损失值表示
    return rmse.item()

# y_hat=torch.tensor([[-5,1000,100000],
#                     [1,6,10]],dtype=torch.float32)
# y=torch.tensor([10,9],dtype=torch.float32)
#
# l=log_rmse(get_net(),y_hat,y)
# print(l)

def load_array(data_array,batch_size):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,shuffle=True,batch_size=batch_size)

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    train_iter=load_array((train_features,train_labels),batch_size)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for x,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(x),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls


#K折交叉验证，数据处理
def get_k_fold_data(k,i,X,y): #i表示我要第几折
    assert k>1 #至少要分成1份以上
    fold_size=X.shape[0]//k #每一折大小就是样本数除k
    # print(f'fold_size:{fold_size}')
    X_train,y_train=None,None
    for j in range(k):
        start,end=j*fold_size,(j+1)*fold_size
        x_part,y_part=X[start:end,:],y[start:end]
        if j==i:
            x_valid,y_valid=x_part,y_part #第i折作为验证集
        elif X_train is None:
            X_train,y_train=x_part,y_part
        else:
            X_train=torch.cat((X_train,x_part),dim=0) #按列拼接，就是接在每行下面
            y_train=torch.cat((y_train,y_part),dim=0)
    return X_train,y_train,x_valid,y_valid

#k折交叉验证-训练
def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,test_l_sum=0,0
    for i in range(k): #做k次
        train_data,train_labels,test_data,test_labels=get_k_fold_data(k,i,X_train,y_train) #拿到第i折
        net=get_net()
        train_ls,test_ls=train(net,train_data,train_labels,test_data,test_labels,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        test_l_sum+=test_ls[-1]
        if i==0:
            # print(f'epoch:{j+1} , train_l:{train_ls[j]} , test_l:{test_ls[j]}')
            plt.figure()
            plt.plot(train_ls, label='train_loss')
            plt.plot(test_ls, label='test_loss')
            plt.legend(loc='upper right')
            plt.ylabel("Loss:")
            plt.xlabel("Epochs:")
            plt.show()
        print(f'fold{i+1} , train_loss:{train_ls[-1]} , test_loss:{test_ls[-1]}')
    return train_l_sum/k,test_l_sum/k

k,num_epochs,lr,weight_decay,batch_size=5,100,5,0,64
# train_l,test_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
# print(f'{k}-折验证:平均训练log rmse:{float(train_l):f} , 平均验证log rmse:{float(test_l):f}')

test_features=all_features.iloc[1321:,:]
# print(test_features.shape)
# test_features[numeric_features]=test_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
# test_features[numeric_features]=test_features[numeric_features].fillna(0)
# test_features=pd.get_dummies(test_features,dummy_na=True)
print(test_features.shape)
test_features=torch.tensor(test_features.values.astype(float),dtype=torch.float32)

def train_and_pred(train_features,test_features,train_labels,num_epochs,lr,weight_decay,batch_size):
    net=get_net()
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    print(f'train_loss:{train_ls[-1]:f}')
    preds=net(test_features).detach().numpy()
    ans=dict()
    ans['id']=range(1,len(preds)+1)
    ans['SalePrice']=preds.reshape(-1).astype(int)
    # print(ans['SalePrice'])
    # print(ans)
    ans=pd.DataFrame(ans)
    ans.to_csv('data/submission.csv',index=False)

train_and_pred(train_features,test_features,train_labels,num_epochs,lr,weight_decay,batch_size)


