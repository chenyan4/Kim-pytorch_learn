# 双向循环神经网络
# 利用 正返向特征 cat起来
# 只在顶层做拼接，中间层不拼接，是两个独立的 循环神经网络
# 不能用来做预测，可以用来 做翻译、填空之类的

import torch
from torch import nn
from text_predo import load_data_time_machine
from RNN_simple import RNNModel,train_ch8,draw_loss_acc

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

vocab_size,num_hiddens,num_layers=vocab.__len__(),256,2
device='cuda:0'
lstm_layer=nn.LSTM(vocab_size,num_hiddens,num_layers,bidirectional=True)
model=RNNModel(lstm_layer,vocab_size)
model=model.to(device)
num_epochs,lr=500,1

train_loss,train_acc=train_ch8(model,train_iter,vocab,lr,num_epochs,device)
draw_loss_acc(train_loss,train_acc,'bid_RNN')