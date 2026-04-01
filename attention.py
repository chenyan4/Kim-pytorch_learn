# 注意力机制
# 束搜索：seq2seq 用贪心搜索预测序列（预测概率最大的那个词）;保存最好的 K个候选，
# 注意力机制，通过 query(随意线索)和 key(不随意线索)来有偏向性的选择输入
# f(x)=求和 a(X,Xi)yi , a(X,Xi)一般称为 注意力权重，其实 就是衡量 query和 所有key之间的距离

import torch
from torch import nn
import matplotlib.pyplot as plt

n_train=50
# torch.sort 返回 值+索引，torch.argsort 只返回 索引
# torch.rand 均匀分布在[0,1]之间，randn是均值为0，方差为1的分布
x_train,index=torch.sort(torch.rand(n_train)*5) # 类似于torch.max,返回的是排列好的值,默认升序排列，和值对应的下标，但是 维度和之前一样

def f(x):
    return 2*torch.sin(x)+x**0.8

y_train=f(x_train)+torch.normal(mean=0.0,std=0.5,size=(n_train,)) # torch.noraml(meam,std,size)
x_test=torch.arange(0,5,0.1)
y_truth=f(x_test)
print(x_test)

def plot_kernel_reg(y_hat):
    plt.figure(figsize=(6,3))
    plt.plot(x_test.detach().tolist(),y_truth.detach().tolist(),label='True',color='b',linestyle='-',linewidth=2)
    plt.plot(x_test.detach().tolist(),y_hat.detach().tolist(),label='pred',color='r',linestyle='--',linewidth=2)
    plt.plot(x_train,y_train,marker='o',alpha=0.5,linestyle='none') # 画圆点 plt.plot(x,y,color,alpha,marker='o,,*,^,s,+') alpha是透明度，marker是圆的样式
    plt.xlim(0,5)
    plt.ylim(-1,5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.savefig('/data/chenyan/pytorch_learn/data/images/no_dig_attention.png',dpi=300)
    plt.close()

# y_hat=torch.repeat_interleave(y_truth.mean(),n_train)
# plot_kernel_reg(y_hat)

# 非参数注意力汇聚
X_repeat=x_test.repeat_interleave(n_train).reshape(-1,n_train)
attention_weight=nn.functional.softmax(-(X_repeat-x_train)**2,dim=1) #这里把 x_train当成 key，相当于 x_test和 x_train所有点算距离差距，差距越小，exp后值越大，softmax 概率越大
y_hat=torch.matmul(attention_weight,y_train)
plot_kernel_reg(y_hat)

X=torch.ones(size=(2,1,4))
Y=torch.ones(size=(2,4,6))

print(torch.bmm(X,Y).shape) # torch.bmm 带批量的矩阵惩罚，表示 第0个x 和 第0个y乘，第1个x 和 第1个y乘

weights=torch.ones(size=(2,10))*0.1
values=torch.arange(20.0).reshape(2,10)
print(torch.bmm(weights.unsqueeze(1),values.unsqueeze(-1)))

class NWKernelRegression(nn.Module):
    def __init__(self):
        super(NWKernelRegression,self).__init__()
        self.w=nn.Parameter(torch.rand(size=(1,),requires_grad=True)) # 会控制窗口大小，就是在 w较大时，差距较大的那些数值经过 exp后概率值会更低，只有距离真的很近时概率值 高，比较敏感和尖锐；w 较小时，会平滑一些
    
    def forward(self,queries,keys,values):
        queries=queries.repeat_interleave(keys.shape[1]).reshape(-1,keys.shape[1])
        self.attention_weights=nn.functional.softmax(-((queries-keys)*self.w)**2/2,dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)) # [50,1,50]*[50,50,1]  行向量×列向量

x_tile=x_train.repeat((n_train,1))
y_tile=y_train.repeat((n_train,1))

# 模型在训练的时候，做attention时，只能看别人的，不能看自己的（这里做 模仿）
# torch.eye(n) 生成 n×n的单位矩阵,1- 表示将对角线上（自己）去掉
keys=x_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape(n_train,-1) # 通过 bool索引出来的是一维张量，[50×49，]
values=y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape(n_train,-1) # 变成[50,49]

net=NWKernelRegression()
loss=nn.MSELoss(reduction='none')
updater=torch.optim.SGD(net.parameters(),lr=0.5)

train_loss=[]
for epoch in range(5):
    updater.zero_grad()
    y_hat=net(x_test,keys,values).reshape(-1)
    l=loss(y_hat,y_train)
    l.sum().backward()
    updater.step()

    train_loss.append(l.sum().item())

plt.figure(figsize=(6,3))
plt.plot(train_loss,label='train_loss',color='b',linestyle='-',linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("/data/chenyan/pytorch_learn/data/images/attention_loss.png",dpi=300)

y_hat=net(x_test,x_tile,y_tile).reshape(-1)
plot_kernel_reg(y_hat)





