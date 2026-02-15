# 批量归一化
# 损失出现在最后，后面的层训练较快，底层训练较慢，归一化使得所有层训练速度更一致
# 顶部训练快，底部训练慢，就导致顶部可能趋近最优后，底部还没达到，导致底部一变，顶部都要跟着变
# 为了避免底部变化时，顶部也跟着重新训练
# 思想：固定小批量里面的均值和方差，固定分布
# 数学公式：y=((x-E(x))/(Var(x)+ϵ))*γ+β，其中E(x)是均值，Var(x)是方差，ϵ是防止分母为0的常数，γ和β是可学习参数（均值0，方差1不一定最好，还是要拉伸）
# 作用在 全连接层和卷积层输出上，激活函数前；全连接层和卷积层输入上
# 对全连接层，作用于特征维；对于卷积层，作用于通道维
# γ（随机缩放）和β（随机偏移），在每个小批量加入随机噪音来控制模型复杂度，没必要和丢弃法混合使用
# 可以加速收敛速度，但一般不改变模型精度，可以用较大学习率
 
# 测试模式时，用移动平均的均值和方差，也就是训练得到的 γ和β（moving_mean和moving_var,全局移动平均），测试时没有小批量,eps防止除0,momentum是移动平均的动量，类似lr

import torch
from torch import nn
from torchvision import transforms,datasets
from torch.utils import data
from VGG import load_data_fashion_mnist,train_ch6,draw_loss_acc

def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    # 如果不需要计算梯度，测试模式
    if not torch.is_grad_enabled():
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4) #要么为2 全连接层，要么为4 卷积层
        if len(X.shape)==2:
            mean=X.mean(dim=0,keepdim=True)
            var=((X-mean)**2).mean(dim=0,keepdim=True)
        else:
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        X_hat=(X-mean)/torch.sqrt(var+eps)
        # moving_mean和moving_var是为了测试模式准备的，相当于每一个小批次进来，都计算一次mean和var，然后移动平均
        moving_mean=momentum*moving_mean+(1-momentum)*mean # momentum比如0.9，历史mean占90%，新mean占10%
        moving_var=momentum*moving_var+(1-momentum)*var
    Y=gamma*X_hat+beta # 缩放和偏移
    # .data返回的是和原张量共享存储的Tensor，.data只得到一个不参与autograd的Tensor视图，仍然是Tensor
    return Y,moving_mean.data,moving_var.data

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super(BatchNorm,self).__init__()
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        
        self.gamma = nn.Parameter(torch.ones(shape)) # 这样定义的参数会自动加入模型参数列表，能用优化器优化
        self.beta = nn.Parameter(torch.zeros(shape))
        # 用 register_buffer 注册：会随 model.to(device) 迁移、会进入 state_dict 保存/加载
        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.ones(shape)

    def forward(self,X):
        # 将 self.moving_mean 和 self.moving_var 放到相同device上,要手动；self.gamma 和 self.beta 随定义的net放到相同设备
        if self.moving_mean.device!=X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y

# 调包的话，卷积用nn.BatchNorm2d(num_features),全连接用nn.BatchNorm1d(num_features)
net=nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5),BatchNorm(6,num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),BatchNorm(16,num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4,120),BatchNorm(120,num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120,84),BatchNorm(84,num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84,10)
)

if __name__=="__main__":
    lr,num_epochs,batch_size=0.05,10,256
    train_iter,test_iter=load_data_fashion_mnist(batch_size)
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(net.parameters(),lr)
    train_acc,train_loss,test_acc=train_ch6(net,train_iter,test_iter,loss,num_epochs,lr,updater,device='cuda:0')
    draw_loss_acc(train_acc,train_loss,test_acc,name="batch_normlization_LeNet")
    print(net[1].gamma.reshape((-1,)),net[1].beta.reshape((-1,)))




