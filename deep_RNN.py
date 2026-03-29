# 深度循环神经网络
# 用更多隐藏层，得到更多非线性性，也就是在 隐藏层到输出层去加深 就行了

import torch
from torch import nn
from text_predo import load_data_time_machine
from RNN_simple import RNNModel,train_ch8,draw_loss_acc

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

vocab_size,num_hiddens,num_layers=vocab.__len__(),256,2
device='cuda:0'
lstm_layer=nn.LSTM(vocab_size,num_hiddens,num_layers) # num_layers 表示 深度要多少层
model=RNNModel(lstm_layer,vocab_size)
model=model.to(device)

num_epochs,lr=500,2
train_loss,train_acc=train_ch8(model,train_iter,vocab,lr,num_epochs,device)
draw_loss_acc(train_loss,train_acc,'RNN_deep')