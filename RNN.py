# 循环神经网络
# 预测损失，就是 n次交叉熵平均损失
# 困惑度就是用exp(loss) 在 平均交叉熵上多一个 exp指数，也可以表示 平均每次可能选项
# 梯度剪裁：预防梯度 爆炸，把一层的梯度 拉成一个向量 g，g=min(1,θ/||g||)g,也就是 梯度长度超过 θ,那么拖影回长度；||g||是L2范数;这么做可以让向量长度变成 θ，但是 向量方向不变，压缩长度

import math
import torch
from torch import nn
from torch.nn import functional as F
from text_predo import load_data_time_machine
import matplotlib.pyplot as plt

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

print(F.one_hot(torch.tensor([0,2]),vocab.__len__())) #F.one_hot(下标索引，长度) 第一个是类别索引，第二个是长度

X=torch.arange(10).reshape((2,5))
print(F.one_hot(X.T,vocab.__len__())) # 做 转置后是连续的，因为到第一个维度

def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01

    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(size=(num_hiddens,),device=device)
    W_hq=normal((num_hiddens,num_outputs)) # 从 隐层到预测
    b_q=torch.zeros(size=(num_outputs,),device=device)

    params=[W_xh,W_hh,b_h,W_hq,b_q]

    for param in params:
        param.requires_grad_(True)
    return params

# 初始化隐藏状态(batch_size,num_hiddens,device):
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros(size=(batch_size,num_hiddens),device=device),) # 放在一个 元组里，LSTM 有两个

def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.matmul(H,W_hh)+torch.matmul(X,W_xh)+b_h)
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)

# init_state和forward_fn 都是初始化函数
class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(self.vocab_size,self.num_hiddens,device)
        self.init_state,self.forward_fn=init_state,forward_fn
        self.device=device

    def __call__(self,X,state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32).to(self.device)
        return self.forward_fn(X,state,self.params)
    
    def begin_state(self,device,batch_size):
        return self.init_state(batch_size,self.num_hiddens,device)

# prefix 生成句子，num_preds 是要生成多少个词
def predict_ch8(prefix,num_preds,net,vocab,device):
    with torch.no_grad():
        state=net.begin_state(device,batch_size=1)
        outputs=[vocab.__getitem__(prefix[0])]
        def get_input():
            return torch.tensor(outputs[-1],device=device).reshape((1,1)) # 留下batch_size和 时间步长
        # 这里不 care输出，只是关注得到最后一个字符的 state
        for y in prefix[1:-1]:
            _,state=net.__call__(get_input(),state) 
            outputs.append(vocab.__getitem__(y))
        outputs.append(vocab.__getitem__(prefix[-1]))
        for _ in range(num_preds):
            y,state=net.__call__(get_input(),state)
            outputs.append(y.argmax(dim=1).reshape(-1).item())

        return ''.join([vocab.to_tokens(idx) for idx in outputs])

# 梯度剪裁：相当于将所有层的梯度 拉成一条向量后，取平方再加和，再取根号，就得到长度;theta 就是阈值，或者说缩放到theta
def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.params # 共享的数值
    norm=torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm.item()>theta:
        for param in params:
            param.grad=param.grad*(theta/norm)

# use_random_iter 表示下一个批量是否要接着使用上一个批量的state,如果是，就是顺序的data
def train_epoch_ch8(net,train_iter,loss,updater,lr,device,use_random_iter):
    state=None
    for X,Y in train_iter:
        acc_num,l_num,num,batch=0,0,0,0
        if state is None or use_random_iter:
            state=net.begin_state(device,batch_size=X.shape[0])
        else:
            if isinstance(net,nn.Module) and not isinstance(state,list):
                state.detach_() # detach()返回一个新张量视图，detach_()直接原地修改 x本身，把state 从旧的计算图中截断，避免反向传播跨越 无限长时间
            else:
                for s in state:
                    s.detach_()
        y=Y.T.reshape(-1)
        X,y=X.to(device),y.to(device)
        y_hat,state=net.__call__(X,state)
        l=loss(y_hat,y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(net.params,batch_size=1,lr=lr) # 因为在里面已经取mean(),也就没有必要再除以 batch_size
        acc_num+=Accuracy(y_hat,y)
        l_num+=l.item()
        num+=X.shape[0]*X.shape[1]
        batch+=1
    return math.exp(l_num/batch),acc_num/num

def SGD(params,batch_size,lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # 要原地修改 模型参数
            param.grad.zero_()

def train_ch8(net,train_iter,vocab,lr,num_epochs,device,use_random_iter=False):
    loss=nn.CrossEntropyLoss(reduction='none')
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        updater=SGD
    train_acc,train_loss=[],[]
    perfix='time traveller'
    for epoch in range(num_epochs):
        l_num,acc_num=train_epoch_ch8(net,train_iter,loss,updater,lr,device,use_random_iter)
        train_loss.append(l_num)
        train_acc.append(acc_num)

        if (epoch+1)%10==0:
            predict=predict_ch8(perfix,30,net,vocab,device)
            print(f'epoch:{epoch+1},困惑度:{train_loss[-1]},train_acc:{train_acc[-1]},predict:{predict}')

    return train_loss,train_acc
        

        
def Accuracy(y_hat,y):
    y_hat=y_hat.argmax(dim=1)
    y=y.reshape(y_hat.shape).long()
    cmp=(y_hat==y).sum().item()
    return cmp

def draw_loss_acc(train_loss,train_acc):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label="train loss",color='b',linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend(loc="upper right")

    plt.subplot(1,2,2)
    plt.plot(train_acc,label="train acc",color='r',linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Acc Curve")
    plt.legend(loc="upper right")

    plt.savefig("/data/chenyan/pytorch_learn/data/images/RNN.png",dpi=300)


num_hiddens=512
net=RNNModelScratch(vocab.__len__(),num_hiddens,'cuda:0',get_params,init_rnn_state,rnn)
# state=net.begin_state(X.shape[0])
# Y,new_state=net.__call__(X,state)
# print(Y.shape,new_state[0].shape)
# print(predict_ch8('time traveller',10,net,vocab,'cuda:0'))
if __name__=="__main__":
    num_epochs,lr=500,1
    train_loss,train_acc=train_ch8(net,train_iter,vocab,lr,num_epochs,'cuda:0',use_random_iter=True)
    draw_loss_acc(train_loss,train_acc)

