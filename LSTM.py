# LSTM:长短期记忆网络
# 忘记门（Ft）：将值朝0减少；输入门（It)：决定是不是忽略掉输入数据；输出门（Ot）：决定是不是使用隐状态，计算 和RNN一样，激活函数是sigmoid
# 候选记忆单元 Ct'(区别于 H),计算和 RNN一样，但激活函数是tanh
# 计算 记忆单元 Ct=Ft*Ct-1+It*Ct' # 由于 可能在（-2，2）之间
# 隐藏状态   Ht=Ot*tanh(Ct) tanh重新放回 （-1，1）之间
# It=simgoid(...)
# Ft=sigmoid(...)
# Ot=sigmoid(...)
# Ct'=tanh(...) 
# Ct=Ft*Ct-1+It*Ct' # 可以理解为 C是一个辅助的记忆单元，忘掉过去 C，或者忘一个保留一个，或者都不忘，比较灵活
# Ht=Ot*tanh(Ct) 也可以重置

import torch
from torch import nn
from text_predo import load_data_time_machine
from RNN import RNNModelScratch,train_ch8,draw_loss_acc
from RNN_simple import RNNModel

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

def get_lstm_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(shape,device=device)*0.01
    
    def three():
        return (normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros((num_hiddens,),device=device))

    W_xi,W_hi,b_i=three()
    W_xf,W_hf,b_f=three()
    W_xo,W_ho,b_o=three()
    W_xc,W_hc,b_c=three()
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros((num_outputs,),device=device)

    params=[W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]
    for param in params:
        param.requires_grad_(True) # 多有一个 _ 表示对自身进行修改，不重新生成一份数据
    return params

def init_lstm_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),torch.zeros((batch_size,num_hiddens),device=device))

def lstm(inputs,state,params):
    W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q=params
    (H,C)=state
    outputs=[]
    for X in inputs:
        I=torch.sigmoid(torch.matmul(X,W_xi)+torch.matmul(H,W_hi)+b_i)
        F=torch.sigmoid(torch.matmul(X,W_xf)+torch.matmul(H,W_hf)+b_f)
        O=torch.sigmoid(torch.matmul(X,W_xo)+torch.matmul(H,W_ho)+b_o)
        C_tilda=torch.tanh(torch.matmul(X,W_xc)+torch.matmul(H,W_hc)+b_c)
        C=F*C+I*C_tilda
        H=O*torch.tanh(C)
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,C)

if __name__=="__main__":
    vocab_size,num_hiddens,device=vocab.__len__(),256,'cuda:0'
    num_epochs,lr=500,1
    # model=RNNModelScratch(vocab_size,num_hiddens,device,get_lstm_params,init_lstm_state,lstm)
    lstm_layer=nn.LSTM(vocab_size,num_hiddens)
    model=RNNModel(lstm_layer,vocab_size)
    model=model.to(device)
    train_loss,train_acc=train_ch8(model,train_iter,vocab,lr,num_epochs,device)
    draw_loss_acc(train_loss,train_acc,'LSTM')