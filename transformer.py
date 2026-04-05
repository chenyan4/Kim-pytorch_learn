# tansformer
# 基于 编码器-解码器架构，纯基于注意力（self_attention),

# 多头注意力
# 层归一化，对于 每一个样本(batch)去做归一化；为什么 不在样本之间做归一化了呢（即对每个特征），因为序列长度是 会变化的，相对来说不固定，每次预测后 序列长度会变长，不稳定
import torch
from torch import nn
import math
import pandas as pd

def sequence_mask(X,valid_lens,value):
    for i in range(len(valid_lens)):
        X[i,int(valid_lens[i].item()):]=value
    return X

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
        return self.ln(self.dropout(Y)+X)

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

num_hiddens,num_heads=100,5
attention=MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
attention.eval()
batch_size,num_queries,num_kvparis,valid_lens=2,4,6,torch.tensor([3,2])
X=torch.ones(size=(batch_size,num_queries,num_hiddens))
Y=torch.ones(size=(batch_size,num_kvparis,num_hiddens))
print(attention(X,Y,Y,valid_lens).shape)       

ln=nn.LayerNorm(2) # 层归一化，样本内的值做归一化，layerNorm(num_hiddens) 要归一化的维度大小，比如[batch_size,queries,num_hiddens] 就在最后一个维度上 (token级别)做归一化，可学习gamma(num_hiddens,),beta(num_hiddens,)，所有token 共用一套gamma和beta
bn=nn.BatchNorm1d(2) # 批量归一化，样本之间的特征做归一化

X=torch.tensor([[1,2],[2,3]],dtype=torch.float32)

add_norm=AddNorm([3,4],0.5)
X=torch.ones(size=(2,3,4))
print(add_norm(X,torch.ones(size=(2,3,4))).shape)

X=torch.ones(size=(2,100,24))
valid_lens=torch.tensor([3,2])
encoder_blk=EncoderBlock(24,24,24,24,[100,24],24,48,8,0.5)
encoder_blk.eval()
print(encoder_blk(X,valid_lens).shape)


