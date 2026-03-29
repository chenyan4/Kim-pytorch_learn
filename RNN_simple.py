import torch
from torch import nn
from torch.nn import functional as F
from text_predo import load_data_time_machine
from RNN import train_ch8,draw_loss_acc

batch_size,num_steps=32,35
train_iter,vocab=load_data_time_machine(batch_size,num_steps)

num_hiddens=256
rnn_layer=nn.RNN(vocab.__len__(),num_hiddens)

state=torch.zeros(size=(1,batch_size,num_hiddens))

X=torch.randn(size=(num_steps,batch_size,vocab.__len__()))
Y,state_new=rnn_layer(X,state)
print(Y.shape)


class RNNModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size):
        super(RNNModel,self).__init__()
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size
        # 如果不是双向
        if not self.rnn.bidirectional:
            self.num_directions=1
            self.linear=nn.Linear(self.num_hiddens,self.vocab_size)
        else:
            self.num_directions=2
            self.linear=nn.Linear(self.num_hiddens*2,self.vocab_size)

    def forward(self,inputs,state):
        X=F.one_hot(inputs.T.long(),self.vocab_size)
        X=X.to(torch.float32)
        Y,state=self.rnn(X,state)
        Y=Y.reshape(-1,Y.shape[-1])
        output=self.linear(Y)
        return output,state

    def begin_state(self,device,batch_size=1):
        if not isinstance(self.rnn,nn.LSTM):
            state=torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device)
            return state
        else:
            return (torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device),
            torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device))
        
def predict_ch8(perfix,num_steps,net,vocab,device):
    state=net.begin_state(device,batch_size=1)
    output=[vocab.__getitem__(perfix[0])]

    def get_inputs():
        return torch.tensor(output[-1],device=device).reshape((1,1))

    for y in perfix[1:-1]:
        _,state=net(get_inputs(),state)
        output.append(vocab.__getitem__(y))
    output.append(vocab.__getitem__(perfix[-1]))

    for _ in range(num_steps):
        y,state=net(get_inputs(),state)
        output.append(y.argmax(dim=1).reshape(-1).item())
    
    return ''.join([vocab.to_tokens(idx) for idx in output])




if __name__=="__main__":
    device='cuda:0'
    net=RNNModel(rnn_layer,vocab.__len__())
    net=net.to(device)
    print(predict_ch8('time traveller',10,net,vocab,'cuda:0'))

    num_epochs,lr=500,1
    train_loss,train_acc=train_ch8(net,train_iter,vocab,lr,num_epochs,device)
    draw_loss_acc(train_loss,train_acc)
