# GRU 门控单元
# 能关注的机制（更新门）、能遗忘的机制（重置门），Rt和Zt分别是重置门、更新门
# 输入是 当前时刻Xt和上一时刻 隐变量Ht-1，计算方式和RNN一样，长度和隐藏层一致，使用sigmoid 作为激活函数，RNN是使用tanh
# 生成候选 隐藏状态，Rt按 元素乘以 Ht-1，Rt 经过sigmoid 在0-1之间，越趋近于0 表示遗忘，表示这个时刻开始，前面的信息就 不要了
# 真正隐藏状态是 Ht=Zt*Ht-1+（1-Zt）*Ht'(候选隐藏状态)，Zt等于1 表示状态不变，Zt等于0，就表示用新状态，Rt如果等于 1就回到RNN了
# 遗忘门表示要用 过去的多少信息，更新门表示 要用当前的多少信息

import torch 
from torch import nn
import re
import collections
from text_predo import load_data_time_machine
from RNN import draw_loss_acc
from RNN_simple import RNNModel
from torch.nn import functional as F
import math

def read_time_machine(base_url):
    with open(base_url,'r') as f:
        lines=f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

def tokenizer(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines]
    elif token=='char':
        return [list(line) for line in lines]

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens=[]
        if reserved_tokens is None:
            reserved_tokens=[]
        counter=count_tokens(tokens)
        self.token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        self.unk,uni_tokens=0,['<unk>']+reserved_tokens
        uni_tokens+=[token for token,freq in self.token_freqs if freq>=min_freq and token not in uni_tokens]

        self.idx_to_token,self.token_to_idx=[],{}
        for token in uni_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token]=len(self.idx_to_token)-1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(tokens):
        if not isinstance(token,(list,tuple)):
            return self.token_to_idx[tokens]
        
        return [self.__getitem__(token) for token in tokens]
    
    def idx_token(self,indices):
        if not isinstance(indices,(tuple,list)):
            return self.idx_to_token[indices]
        return [self.idx_to_token.get(idx,self.unk) for idx in indices]


        
def count_tokens(tokens):
    if len(tokens)==0:
        return collections.Counter()
    count_tokens=[]
    if isinstance(tokens[0],list):
        for line in tokens:
            for token in line:
                count_tokens.append(token)
        return collections.Counter(count_tokens)
    else:
        return collections.Counter(tokens)

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(shape,device=device)*0.01

    def three():
        return (normal((vocab_size,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(size=(num_hiddens,),device=device))

    W_xz,W_hz,b_z=three() # 重置门
    W_xr,W_hr,b_r=three() # 隐藏门

    W_xh,W_hh,b_h=three()
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros((num_outputs,),device=device)

    params=[W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]

    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

def gru(inputs,state,params):
    W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        Z=torch.sigmoid(torch.matmul(X,W_xz)+torch.matmul(H,W_hz)+b_z)
        R=torch.sigmoid(torch.matmul(X,W_xr)+torch.matmul(H,W_hr)+b_r)
        H_tilda=torch.tanh(torch.matmul(X,W_xh)+torch.matmul(H*R,W_hh)+b_h)
        H=Z*H+(1-Z)*H_tilda
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    
    return torch.cat(outputs,dim=0),(H,)

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(self.vocab_size,self.num_hiddens,device)
        self.init_state,self.forward_fn=init_state,forward_fn
        self.device=device
    
    def __call__(self,X,state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32).to(self.device)
        Y,state=self.forward_fn(X,state,self.params)
        return Y,state

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)
    
def predict_ch8(perfix,num_preds,net,vocab,device):
    with torch.no_grad():
        state=net.begin_state(1,device)
        outputs=[vocab.__getitem__(perfix[0])]
        def get_input():
            return torch.tensor(outputs[-1],device=device).reshape((1,1))
        for y in perfix[1:-1]:
            _,state=net.__call__(get_input(),state)
            outputs.append(vocab.__getitem__(y))
        outputs.append(vocab.__getitem__(perfix[-1]))

        for _ in range(num_preds):
            y,state=net.__call__(get_input(),state)
            outputs.append(y.argmax(dim=1).reshape(-1).item())
        return ''.join([vocab.to_tokens(idx) for idx in outputs])

def grad_clipping(net,theta=1):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters()]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm>theta:
        for param in params:
            param.grad=param.grad*(theta/norm)
    
def SGD(params,batch_size,lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def Accuracy(y_hat,y):
    y_hat=y_hat.argmax(dim=-1)
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).sum().item()
    return cmp

def train_epoch_ch8(net,train_iter,loss,updater,lr,device,use_random_iter):
    state=None
    acc_num,l_num,num,batch=0,0,0,0
    for X,Y in train_iter:
        if state is None or use_random_iter:
            state=net.begin_state(X.shape[0],device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,list):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y=Y.T.reshape(-1)
        X,y=X.to(device),y.to(device)
        Y_hat,state=net.__call__(X,state)
        l=loss(Y_hat,y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net)
            updater.step()
        else:
            l.backward()
            grad_clipping(net)
            updater(net.params,1,lr)
        acc_num+=Accuracy(Y_hat,y)
        l_num+=l.item()
        num+=X.shape[0]*X.shape[1]
        batch+=1

    return math.exp(l_num/batch),acc_num/num

def train_ch8(net,train_iter,vocab,num_epochs,lr,device,use_random_iter=False):
    loss=nn.CrossEntropyLoss(reduction='none')
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        updater=SGD
    train_loss,train_acc=[],[]
    for epoch in range(num_epochs):
        l_num,acc_num=train_epoch_ch8(net,train_iter,loss,updater,lr,device,use_random_iter)
        train_loss.append(l_num)
        train_acc.append(acc_num)

        if (epoch+1)%10==0:
            print(f"epoch:{epoch+1},困惑度:{train_loss[-1]},train_acc:{train_acc[-1]},predict:{predict_ch8('time traveller',50,net,vocab,device)}")

    return train_loss,train_acc

vocab_size,num_hiddens,device=vocab.__len__(),256,'cuda:0'
num_epochs,lr=500,1
# model=RNNModelScratch(vocab_size,num_hiddens,device,get_params,init_gru_state,gru)
gru_layer=nn.GRU(vocab_size,num_hiddens)
model=RNNModel(gru_layer,vocab_size)
model=model.to(device)
train_loss,train_acc=train_ch8(model,train_iter,vocab,num_epochs,lr,device)
draw_loss_acc(train_loss,train_acc,'GRU')


