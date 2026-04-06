# tansformer
# 基于 编码器-解码器架构，纯基于注意力（self_attention),

# 多头注意力
# 层归一化，对于 每一个样本(batch)去做归一化；为什么 不在样本之间做归一化了呢（即对每个特征），因为序列长度是 会变化的，相对来说不固定，每次预测后 序列长度会变长，不稳定
import torch
from torch import nn
import math
import pandas as pd
import matplotlib.pyplot as plt
from transfer_data_predo import load_data_nmt
import collections

def sequence_mask(X,valid_lens,value):
    # 全张量实现，避免 Python 循环与 .item()（否则会大量触发 CPU/GPU 同步，训练极慢）
    maxlen=X.size(-1)
    lens=valid_lens.to(device=X.device,dtype=X.dtype).view(-1,1) # [batch_size,1]
    col=torch.arange(maxlen,device=X.device,dtype=X.dtype).view(1,-1) # [1,maxlen]
    mask=col>=lens
    return X.masked_fill(mask,value)

def masked_softmax(X,valid_lens=None):
    if valid_lens==None:
        return nn.functional.softmax(X,dim=-1)
    shape=X.shape
    if valid_lens.dim()==1:
        valid_lens=valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens=valid_lens.reshape(-1)
    X=sequence_mask(X.reshape(-1,shape[2]),valid_lens,value=-1e6)
    return nn.functional.softmax(X.reshape(shape),dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,queries,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.permute(0,2,1))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

# 多头注意力机制
def transpose_qkv(X,num_heads):
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1) # 也就是把每个 序列长度切成 num_heads份 并行处理
    X=X.permute(0,2,1,3) # [batch_size,num_heads,num_query,num_hiddens]
    # 为什么这么做 ，可以理解为 每个头都有所有 token的不同但固定长度的信息（比如 头1 有token1-token10 前32维;头2 有token1-token10中间32维）头1 学一种模式；头2 学一种模式
    return X.reshape(-1,X.shape[2],X.shape[3]) # [batch_size*num_heads,num_query,num_hiddens] 

def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)


# num_hiddens指 将key、value、query映射到 num_hiddens ; num_heads指多少个头
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k=nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v=nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o=nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys),self.num_heads)
        values=transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens=valid_lens.repeat_interleave(self.num_heads,dim=0)

        output=self.attention(queries,keys,values,valid_lens)

        output_concat=transpose_output(output,self.num_heads)
        return self.W_o(output_concat)

# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_outputs):
        super(PositionWiseFFN,self).__init__()
        self.dense1=nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X))) # pytorch 中，当nn.Linear输入不是二维时，将前面维度都看做样本维，最后维度 看做 feature维

# 残差连接和归一化层
class AddNorm(nn.Module):
    def __init__(self,normalized_shape,dropout):
        super(AddNorm,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(X+self.dropout(Y))

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias=False):
        super(EncoderBlock,self).__init__()
        self.attention=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lens):
        Y=self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y))

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_lens=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.p=torch.zeros(size=(1,max_lens,num_hiddens))
        X=torch.arange(0,max_lens,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,2*torch.arange(0,num_hiddens//2,dtype=torch.float32)/num_hiddens)
        self.p[:,:,0::2]=torch.sin(X)
        self.p[:,:,1::2]=torch.cos(X)

    def forward(self,X):
        X=X+self.p[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)

class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias=False):
        super(TransformerEncoder,self).__init__()
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,self.num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}",EncoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout))

    def forward(self,X,valid_lens):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens)) # self.embedding 满足均值为0，方差为1，num_hiddens越大，数值越小；而Position是 -1~1之间，所以乘一个系数
        self.attention_weights=[None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return X

# i 表示第 i个块
class DecoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,i,use_bias=False):
        super(DecoderBlock,self).__init__()
        self.i=i
        self.attention1=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.attention2=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm2=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm3=AddNorm(norm_shape,dropout)

    def forward(self,X,state):
        enc_outputs,enc_valid_lens=state[0],state[1] # state存了三个东西，一个是encoder 输出，一个是 encoder的valid_lens,一个是 decoder 状态，是一个列表[None]*解码器层数，当第0个词进来时，会经过所有解码器，这时候所有层数解码器 被附上第一个状态
        # 在第一个词进入 解码器时成立,在训练时不会使用
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat([state[2][self.i],X],dim=1) # [batch_size,num_steps,num_hiddens]
        state[2][self.i]=key_values

        if self.training:
            batch_size,num_steps,_=X.shape
            # 因为 每一个序列的 词只能看到自己和自己之前的词，在训练的时候，第一个词只有自己和前面的词的 注意力分数是有用的
            dec_valid_lens=torch.arange(1,num_steps+1,device=X.device).repeat(batch_size,1) # 是按照 query级别的mask
        else:
            dec_valid_lens=None
        
        X2=self.attention1(X,key_values,key_values,dec_valid_lens)
        Y=self.addnorm1(X,X2)
        Y2=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z=self.addnorm2(Y,Y2)
        return self.addnorm3(Z,self.ffn(Z)),state

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout):
        super(TransformerDecoder,self).__init__()
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,self.num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block{i}',DecoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,i))    

        self.dense=nn.Linear(num_hiddens,vocab_size)
    
    def init_state(self,enc_outputs,enc_valid_lens):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]
    
    def forward(self,X,state):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights=[[None]*self.num_layers for _ in range(2)] # 一层解码器有两个 多头注意力层
        for i,blk in enumerate(self.blks):
            X,state=blk(X,state)
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i]=blk.attention2.attention.attention_weights
        return self.dense(X),state

    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,X,dec_X,enc_valid_lens):
        enc_outputs=self.encoder(X,enc_valid_lens)
        state=self.decoder.init_state(enc_outputs,enc_valid_lens)
        Y,state=self.decoder(dec_X,state)
        return Y,state


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_lens):
        weights=torch.ones_like(label)
        weights=sequence_mask(weights,valid_lens,value=0)
        self.reduction='none'
        batch_size,num_classes=pred.shape[0],pred.shape[2]
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.reshape(-1,num_classes),label.reshape(-1)).reshape(batch_size,-1)
        weighted_loss=(unweighted_loss*weights).mean(dim=1)
        return weighted_loss

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters()]
    else:
        params=net.params
    norm=torch.sqrt(sum((torch.sum(p.grad**2) for p in params)))
    if norm>theta:
        for param in params:
            param.grad*=theta/norm

def train_seq2seq(net,train_iter,lr,num_epochs,vocab,device):
    def xavier_init_weights(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) in (nn.GRU,nn.LSTM):
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    updater=torch.optim.Adam(net.parameters(),lr=lr)
    loss=MaskedSoftmaxCELoss()

    train_loss=[]
    for epoch in range(num_epochs):
        l_num,num=0,0
        for batch in train_iter:
            X,X_valid_len,tgt,Y_valid_len=[x.to(device) for x in batch]
            begin_bos=torch.tensor([vocab['<bos>']]*len(tgt),device=device,dtype=torch.long).reshape(-1,1)
            dec_in=torch.cat((begin_bos,tgt[:,:-1]),dim=1)
            updater.zero_grad()
            Y_hat,state=net(X,dec_in,X_valid_len)
            l=loss(Y_hat,tgt,Y_valid_len)
            ls=l.sum()
            ls.backward()
            updater.step()

            l_num+=ls.detach().item()
            num+=len(X)
        train_loss.append(l_num/num)

        if (epoch+1)%10==0:
            print(f'epoch:{epoch+1},train_loss:{train_loss[-1]}')
    return train_loss

def draw_loss(train_loss,save_name):
    plt.figure(figsize=(6,3))
    plt.plot(train_loss,label="train_loss",color='b',linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(f"/workspace/Kim-pytorch_learn/data/images/{save_name}.png",dpi=300)

def truncate_pad(line,num_steps,value):
    if len(line)>num_steps:
        return line[:num_steps]
    else:
        return line+[value]*(num_steps-len(line))

# 
def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device,save_attention_weights=False):
    net.eval()
    src_tokens=src_vocab[src_sentence.split()+['<eos>']]
    enc_valid_lens=torch.tensor([min(len(src_tokens),num_steps)],device=device)
    src_tokens=torch.tensor(
        truncate_pad(src_tokens,num_steps,value=src_vocab['<pad>']),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    enc_outputs=net.encoder(src_tokens,enc_valid_lens)
    state=net.decoder.init_state(enc_outputs,enc_valid_lens)

    dec_X=torch.tensor([[tgt_vocab.__getitem__('<bos>')]],dtype=torch.long,device=device)
    output_seq,attention_weight_seq=[],[]
    for _ in range(num_steps):
        Y,state=net.decoder(dec_X,state)
        dec_X=Y.argmax(dim=-1)
        pred=int(dec_X.squeeze(-1).item())
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights())
        if pred==tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)),attention_weight_seq

def blue(pred,label,k):
    pred,label=pred.split(),label.split()
    len_pred,len_label=len(pred),len(label)
    score=math.exp(min(0,1-(len_label/len_pred)))
    for n in range(1,k+1):
        num_matches=0
        label_subs=collections.defaultdict(int)
        for i in range(len_label-n+1):
            label_subs[''.join(label[i:i+n])]+=1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred[i:i+n])] and label_subs[''.join(pred[i:i+n])]>0:
                num_matches+=1
                label_subs[''.join(pred[i:i+n])]-=1
        score=score*math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score


if __name__=="__main__":
    num_hiddens,num_layers,dropout,batch_size,num_steps=32,2,0.1,64,10
    lr,num_epochs,device=0.005,200,'cuda:0'
    ffn_num_input,ffn_num_hiddens,num_heads=32,64,4
    key_size,query_size,value_size=32,32,32
    norm_shape=[32]

    train_iter,src_vocab,tgt_vocab=load_data_nmt(batch_size=batch_size,num_steps=num_steps)
    encoder=TransformerEncoder(src_vocab.__len__(),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)
    decoder=TransformerDecoder(tgt_vocab.__len__(),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)

    net=EncoderDecoder(encoder,decoder)
    train_loss=train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
    draw_loss(train_loss,"transformer")

    engs=['go .','i lost .','he\'s calm .','i\'m home .']
    fras=['va !','j\'ai perdu .','il est calme .','je suis chez moi .']
    for eng,fra in zip(engs,fras):
        translation,attention_weight_seq=predict_seq2seq(net,eng,src_vocab,tgt_vocab,num_steps,device)
        print(f'{eng}=>{translation},blue:{blue(translation,fra,k=2)}')

# num_hiddens,num_heads=100,5
# attention=MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
# attention.eval()
# batch_size,num_queries,num_kvparis,valid_lens=2,4,6,torch.tensor([3,2])
# X=torch.ones(size=(batch_size,num_queries,num_hiddens))
# Y=torch.ones(size=(batch_size,num_kvparis,num_hiddens))
# print(attention(X,Y,Y,valid_lens).shape)       

# ln=nn.LayerNorm(2) # 层归一化，样本内的值做归一化，layerNorm(num_hiddens) 要归一化的维度大小，比如[batch_size,queries,num_hiddens] 就在最后一个维度上 (token级别)做归一化，可学习gamma(num_hiddens,),beta(num_hiddens,)，所有token 共用一套gamma和beta
# bn=nn.BatchNorm1d(2) # 批量归一化，样本之间的特征做归一化

# X=torch.tensor([[1,2],[2,3]],dtype=torch.float32)

# add_norm=AddNorm([3,4],0.5)
# X=torch.ones(size=(2,3,4))
# print(add_norm(X,torch.ones(size=(2,3,4))).shape)

# X=torch.ones(size=(2,100,24))
# valid_lens=torch.tensor([3,2])
# encoder_blk=EncoderBlock(24,24,24,24,[100,24],24,48,8,0.5)
# encoder_blk.eval()
# print(encoder_blk(X,valid_lens).shape)



