# 序列模型：是有时序结构，变量是不独立的
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils import data

T=1000
time=torch.arange(start=1,end=T+1,step=1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(mean=0,std=0.2,size=(T,))

# plt.figure(figsize=(12,6))
# plt.plot(time.tolist(),x.tolist(),color='b',linestyle='-',linewidth=2) # plt.plot(x,y) x，y轴数值
# plt.xlabel('Time')
# plt.ylabel('X')
# plt.grid(True) # 显示网格
# plt.xlim(1,1000) # 限制 x轴范围
# # plt.ylin(0,5)
# # plt.xticks(x) 强制 X轴显示你的内容
# plt.savefig("/data/chenyan/pytorch_learn/data/images/time.png",dpi=300)
# plt.close()

# 引用 马尔科夫,τ 是4表示跟前4个元素相关，也意味着要有四个才能预测
tau=4
features=torch.zeros(size=(T-tau,tau))
# 对于
for i in range(tau):
    features[:,i]=x[i:T-tau+i]

labels=x[tau:].reshape((-1,1))

def load_array(data_array,batch_size):
    train_data=data.TensorDataset(*data_array) # torch.utils.data.TensorDataset 接受若干个同长度的张量，按下标对齐打包成样本
    return data.DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True)

batch_size,n_train=16,600
train_iter=load_array((features[:n_train],labels[:n_train]),batch_size)

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform(m.weight)

def get_net():
    net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss=nn.MSELoss()

def train(net,train_iter,loss,num_epochs,updater):
    if isinstance(net,nn.Module):
        net.train()
    for epoch in range(num_epochs):
        num,l_num=0,0
        for x,y in train_iter:
            updater.zero_grad()
            y_hat=net(x)
            l=loss(y_hat,y)
            l.backward()
            updater.step()

            l_num+=l.item()
            num+=len(x)
        print(f'epoch:{epoch+1},train_loss:{l_num/num}')

if __name__=="__main__":
    net=get_net()
    lr,num_epochs=0.01,5

    updater=torch.optim.Adam(net.parameters(),lr=lr)
    train(net,train_iter,loss,num_epochs,updater)

    # 预测得到的 下一个预测值，作为输入再预测下一个（提升难度）
    multistep_preds=torch.zeros(T)

    # 因为 603是训练的一部分，作为预测，没必要在算603，所以从 604开始
    multistep_preds[:n_train+tau]=x[:n_train+tau]
    for i in range(n_train+tau,T):
        multistep_preds[i]=net(multistep_preds[i-tau:i].unsqueeze(0)).reshape(-1)
    

    # output_preds=net(features).reshape(-1)
    # output_preds=output_preds.detach().tolist()

    # x=x.tolist()
    # output_preds=x[:tau]+output_preds
    # time=time.tolist()
    # multistep_preds=multistep_preds.tolist()

    max_steps=64
    # 一次 把（1，4，16，64）步长都算了，取的是最小样本数，按最小样本去算相同样本下，不同 预测步长的差异
    features=torch.zeros(size=(T-tau-max_steps+1,tau+max_steps))
    for i in range(tau):
        features[:,i]=x[i:i+T-tau-max_steps+1]

    for i in range(tau,tau+max_steps):
        features[:,i]=net(features[:,i-tau:i]).reshape(-1)

    steps=(1,4,16,64)
    plt.figure(figsize=(12,6))
    x=x.tolist()
    plt.plot(time,x,label='data',color='b',linestyle='-',linewidth=2) # plt.plot(x,y) x，y轴数值
    colors=['r','g','c','m']
    for i,step in enumerate(steps):
        prob=step+tau-1
        step_time=time[step+tau-1:T-max_steps+step]
        step_preds=features[:,prob]

        plt.plot(step_time.tolist(),step_preds.tolist(),label=f'{step}-step pred',color=colors[i],linestyle='-',linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid(True) # 显示网格
    plt.xlim(1,1000) # 限制 x轴范围
    plt.legend(loc="lower left")

    plt.savefig("/data/chenyan/pytorch_learn/data/images/multi_step.png",dpi=300)
    plt.close()




    

    # plt.figure(figsize=(12,6))
    # plt.plot(time,x,label='data',color='b',linestyle='-',linewidth=2) # plt.plot(x,y) x，y轴数值
    # plt.plot(time,output_preds,label='1-step preds',color='r',linestyle='-',linewidth=2)
    # plt.plot(time,multistep_preds,label='multistep preds',color='g',linestyle='--',linewidth=2)

    # plt.xlabel('Time')
    # plt.ylabel('X')
    # plt.grid(True) # 显示网格
    # plt.xlim(1,1000) # 限制 x轴范围
    # plt.legend(loc="lower left")
    # # plt.ylin(0,5)
    # # plt.xticks(x) 强制 X轴显示你的内容
    # plt.savefig("/data/chenyan/pytorch_learn/data/images/time.png",dpi=300)
    # plt.close()





