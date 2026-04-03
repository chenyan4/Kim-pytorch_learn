# 序列到序列学习：机器翻译
# 编码器是一个 RNN（可以双向），拿到最后一个state，丢进解码器
# 解码器是一个 RNN输出，得到 state作为初始状态进行解码，只能单向（因为预测），有开始符<bos>,结束符<eos>
# 指标 BLEU，看 n-gram的概率（一般n到4），取log后取平均，再取 e，对词序和句子连续性的惩罚；对长度的惩罚，exp(min(0,1-label/pred)) #如果 句子长，不惩罚；句子越短，惩罚越厉害；两个 数值相乘就是BLEU

import collections
import math
import torch
from torch import nn
from transfer_data_predo import load_data_nmt
import matplotlib.pyplot as plt

class Seq2SeqEncoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super(Seq2SeqEncoder,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size) 
        # 词嵌入，内部维护一个 可学习的矩阵（vocab_size,embed_size),用vocab_size就可以索引一个 稠密向量；一个输出（batch_size,num_steps) 经过embedding后 (batch_size,num_steps,embed_size)
        # embedding 初始化是通过正态分布进行初始化的，保持独立，初始化 是随机采样，向量一样概率 ≈0（每个元素都是 随机采样的）

        # 正态分布初始化
        # nn.init.normal_(embedding.weight, mean=0, std=0.1)
        # 均匀分布初始化
        # nn.init.uniform_(embedding.weight, a=-0.1, b=0.1)

        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self,X,X_valid_len):
        X=self.embedding(X)
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        return output,state

class Seq2SeqDecoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super(Seq2SeqDecoder,self).__init__()
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs):
        return enc_outputs[1] # 返回的是(output,state) 取state
    
    def forward(self,X,state,pred_state=None):
        X=self.embedding(X).permute(1,0,2)
        context=state[-1].repeat(X.shape[0],1,1)
        X_and_contenx=torch.cat((X,context),dim=-1)
        if pred_state is None:
            output,state=self.rnn(X_and_contenx)
        else:
            output,state=self.rnn(X_and_contenx,pred_state)
        # output=self.dense(output.reshape(-1,output.shape[-1]))
        # output=out.reshape(-1,X.shape[1],output.shape[-1]).permute(1,0,2)
        output=self.dense(output).permute(1,0,2) # 输入 [num_steps,batch_size,num_hiddens] * [num_hiddens,vocab_size] 最后应得到 [num_steps,batch_size,vocab_size]
        return output,state

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,X,dec_input,X_valid_len):
        enc_outputs=self.encoder(X,X_valid_len)
        state=self.decoder.init_state(enc_outputs)
        output,state=self.decoder.forward(dec_input,state)
        return output,state

# 零值化 屏蔽不相关项
def sequence_mask(X,valid_len,value=0):
    batch_size=X.shape[0]
    for i in range(batch_size):
        X[i,int(valid_len[i].item()):]=value
    return X 

# 带mask的交叉熵损失函数，继承 nn.CrossEntropyLoss
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # 不需要初始化，重构一个 forward功能就行，__init__直接使用父类的参数
    def forward(self,pred,label,valid_len):
        # 掩码
        weights=torch.ones_like(label)
        weights=sequence_mask(weights,valid_len)
        self.reduction='none'
        num_classes,batch_size=pred.shape[-1],pred.shape[0]
        # 调用父类的函数，要用super().funtion()
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.reshape(-1,num_classes),label.reshape(-1)).reshape(batch_size,-1) # nn.CrossEntorpyLoss() 对于pred，要把预测值放到第二个维度
        weighted_loss=(unweighted_loss*weights).mean(dim=1)
        return weighted_loss

def grad_clipping(net,theta=1):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters()]
    else:
        params=net.params
    norm=torch.sqrt(sum((torch.sum(p.grad**2) for p in params)))
    if norm>theta:
        for param in params:
            param.grad*=(theta/norm)

def train_seq2seq(net,data_iter,lr,num_epochs,tgt_vocab,device):
    def xavier_init_weights(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names: # 因为 GRU中是很多参数，名字是weight_ih_l0,bias_ih_l0,所以要先找名字；m.flat_weights_names 拿到所有参数名列表；m._parameters[name]拿出参数
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    updater=torch.optim.Adam(net.parameters(),lr=lr)
    loss=MaskedSoftmaxCELoss()

    net.train()
    train_loss=[]
    for epoch in range(num_epochs):
        l_num,num=0,0
        for batch in data_iter:
            updater.zero_grad()
            X,X_valid_len,Y,Y_valid_len=[x.to(device) for x in batch]
            # 开始符
            bos=torch.tensor([tgt_vocab.__getitem__('<bos>')]*Y.shape[0],device=device).reshape(-1,1)
            dec_input=torch.cat([bos,Y[:,:-1]],dim=1) # 保持 词序列长度相同，丢到最后一个
            Y_hat,_=net(X,dec_input,X_valid_len)
            l=loss(Y_hat,Y,Y_valid_len)
            l.sum().backward()
            grad_clipping(net,1)
            num_tokens=Y_valid_len.sum()
            updater.step()

            l_num+=l.sum().item()
            num+=len(X)
        train_loss.append(l_num/num)
        if (epoch+1)%10==0:
            print(f'epoch:{epoch+1},train_loss:{train_loss[-1]}')
    return train_loss

def draw_loss(train_loss,save_name):
    plt.figure(figsize=(12,4))
    plt.plot(train_loss,label="train_loss",color='b',linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title('Loss Curve')
    plt.legend(loc='upper right')

    plt.savefig(f"/workspace/Kim-pytorch_learn/data/images/{save_name}.png",dpi=300)

def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    else:
        return line+[padding_token]*(num_steps-len(line))

def begin_state(num_layers,batch_size,num_hiddens,device):
    return torch.zeros(size=(num_layers,batch_size,num_hiddens),device=device)

def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device,save_attention_weights=False):
    net.eval()
    src_tokens=src_vocab.__getitem__(src_sentence.split())+src_vocab.__getitem__(['<eos>'])
    enc_valid_len=torch.tensor([len(src_tokens)],device=device)
    src_tokens=truncate_pad(src_tokens,num_steps,src_vocab.__getitem__('<pad>'))
    enc_X=torch.tensor(src_tokens,device=device).unsqueeze(0)
    enc_outputs=net.encoder(enc_X,enc_valid_len)
    dec_state=net.decoder.init_state(enc_outputs)
    dec_X=torch.tensor([tgt_vocab.__getitem__('<bos>')],device=device).unsqueeze(0)
    output_seq,attention_weight_seq=[],[]
    state=begin_state(num_layers,1,num_hiddens,device)
    for _ in range(num_steps):
        Y,state=net.decoder(dec_X,dec_state,state)
        dec_X=Y.argmax(dim=2)
        pred=dec_X.squeeze(0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weight)
        if pred==tgt_vocab.__getitem__('<eos>'):
            break
        output_seq.append(pred)
    return ' '.join([tgt_vocab.to_tokens(idx) for idx in output_seq]),attention_weight_seq

# k 是n-gram
def blue(pred_seq,label_seq,k):
    pred_tokens,label_tokens=pred_seq.split(),label_seq.split()
    len_pred,len_label=len(pred_tokens),len(label_tokens)
    # 计算 词长度的影响
    score=math.exp(min(0,1-(len_label/len_pred)))
    for n in range(1,k+1):
        num_matches,label_subs=0,collections.defaultdict(int)
        for i in range(len_label-n+1):
            label_subs[''.join(label_tokens[i:i+n])]+=1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred_tokens[i:i+n])] and label_subs[''.join(pred_tokens[i:i+n])]>0:
                num_matches+=1
                label_subs[''.join(pred_tokens[i:i+n])]-=1
        score*=math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score




# encoder=Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
# encoder.eval()
# X=torch.zeros(size=(4,7),dtype=torch.long)
# output,state=encoder(X)

# decoder=Seq2SeqDecoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
# decoder.eval()
# state=decoder.init_state(encoder(X))
# output,state=decoder(X,state)
# print(output.shape)
# print(state.shape)

# X=torch.tensor([[1,2,3],[4,5,6]])
# valid_len=torch.tensor([1,2])
# print(sequence_mask(X,valid_len))

# loss=MaskedSoftmaxCELoss()
# print(loss(torch.ones(size=(3,4,10)),torch.ones(size=(3,4),dtype=torch.long),torch.tensor([4,2,0])))

if __name__=="__main__":
    embed_size,num_hiddens,num_layers,dropout=32,32,2,0.1
    batch_size,num_steps=64,10
    lr,num_epochs,device=0.005,300,'cuda:0'

    train_iter,src_vocab,tgt_vocab=load_data_nmt(batch_size,num_steps)
    encoder=Seq2SeqEncoder(src_vocab.__len__(),embed_size,num_hiddens,num_layers,dropout)
    decoder=Seq2SeqDecoder(tgt_vocab.__len__(),embed_size,num_hiddens,num_layers,dropout)

    net=EncoderDecoder(encoder,decoder)
    train_loss=train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
    draw_loss(train_loss,'seq2seq')

    engs=['go .','i lost .','he\'s calm .','i\'m home .']
    fras=['va !','j\'ai perdu .','il est calme .','je suis chez moi .']
    for eng,fra in zip(engs,fras):
        translation,attention_weight_seq=predict_seq2seq(net,eng,src_vocab,tgt_vocab,num_steps,device)
        print(f'{eng}=>{translation},blue:{blue(translation,fra,k=2)}')