# 基于微调做 NLP模型
# 预训练模型抽取了足够的特征
# 只有编码器的Transformer
# 两个版本：base blocks=12 hidden_size=768 heads=12 ; Large blocks=24 hidden_size=1024 heads=12
# 输入：把多个句子拼起来，每个样本是一个句子对，加入额外的分段嵌入，位置编码可学习

# 预训练任务1：带掩码的语言模型
# Transformer的编码器是双向的，标准语言模型要求单向
# 带掩码的语言模型每次随机（15%）将一些词元换成 <mask> (有点像 完形填空)
# 在微调任务中不出现 <mask>,80%下将选中词元换成 <mask>,10% 换成下一个随机词元，10% 保持原有的词元

# 预训练任务2：下一句子预测
# 预测一个句子对中两个句子是不是相邻

import torch
from torch import nn
import math

# 拼接两个句子，返回 句子和分割编码
def get_tokens_and_segments(tokens_a,tokens_b=None):
    tokens=['<cls>']+tokens_a+['<sep>']
    # 分割编码
    segments=[0]*(len(tokens_a)+2)
    if tokens_b is not None:
        tokens+=tokens_b+['<sep>']
        segments+=[1]*(len(tokens_b)+1)
    return tokens,segments

def sequence_mask(X,valid_lens,value):
    max_lens=X.shape[-1]
    row=valid_lens.reshape(-1,1)
    col=torch.arange(max_lens,device=X.device,dtype=X.dtype).reshape(1,-1)
    mask=col>=row
    return X.masked_fill(mask,value)

def masked_softmax(X,valid_lens=None):
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    shape=X.shape
    if valid_lens.dim()==1:
        valid_lens=valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens=valid_lens.reshape(-1)
    X=sequence_mask(X.reshape(-1,shape[2]),valid_lens,value=-1e6)
    return nn.functional.softmax(X.reshape(shape),dim=-1)


def DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,queries,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.permute(0,2,1))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

def transpose_qkv(X,num_heads):
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X=X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.reshape[3])

def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

def MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.W_k=nn.Linear(key_size,num_hiddens)
        self.W_q=nn.Linear(query_size,num_hiddens)
        self.W_v=nn.Linear(value_size,num_hiddens)
        self.W_o=nn.Linear(num_hiddens,num_hiddens)

    def forward(self,queries,keys,values,valid_lens=None):
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys),self.num_heads)
        values=transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens=valid_lens.repeat_interleave(self.num_heads,dim=0)
        output=self.attention(queries,keys,values,valid_lens)
        output_concat=transpose_output(X,self.num_heads)
        return self.W_o(output_concat)

def AddNorm(nn.Module):
    def __init__(self,normlized_shape,dropout):
        super(AddNorm,self).__init__()
        self.ln=nn.LayerNorm(normalized_shape)
        self.dropout=nn.Dropout(dropout)

    def forward(self,X,Y):
        return self.ln(X+self.dropout(Y))

def PositionWiseFFN(nn.Module):
    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_outputs):
        super(PositionWiseFFN,self).__init__()
        self.dense1=nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu=nn.ReLU
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias=False):
        self.attention=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lens):
        Y=self.addnorm1(self.attention(X,X,X,valid_lens))
        return self.addnorm2(self.ffn(Y))

class BERTEncoder(nn.Module):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,max_len=1000,key_size=768,query_size=768,value_size=768,use_bias=False):
        super(BERTEncoder,self).__init__()
        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        # 片段分类嵌入
        self.segment=nn.Embedding(2,num_hiddens)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block{i}',EncoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias))
        # 定义位置编码（可学习）
        self.pos_embedding=nn.Paramters(torch.randn(size=(1,max_lens,num_hiddens)))
    
    def forward(self,tokens,segments,valid_lens):
        X=self.token_embedding(tokens)+self.segment(segments)
        X=X+self.pos_embedding[:,:tokens.shape[1],:]
        for blk in self.blks:
            X=blk(X,valid_lens)
        return X

# torch.randint(start,end,size) # 生成[start,end)间随机整数

# 掩码语言模型,计算两个Loss，一个是token间
class MaskLM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_inputs=768):
        super(MaskLM,self).__init__()
        self.mlp=nn.Sequential(nn.Linear(num_inputs,num_hiddens),
        nn.ReLU(),
        nn.LayerNorm(num_hiddens),
        nn.Linear(num_hiddens,vocab_size))

    def forward(self,X,pred_positions):
        num_pred_positions=pred_positions.shape[1] # [batch_size,num_pred_positions]
        pred_positions=pred_positions.reshape(-1)
        batch_size=X.shape[0]
        batch_idx=torch.arange(0,batch_size)
        batch_idx=batch_id.repeat_interleave(num_pred_positions) # [batch_size × num_pred_positions,]
        masked_X=X[batch_idx,pred_positions] # [batch_size × num_pred_positions,num_hiddens]
        masked_X=masked_X.reshape(batch_size,num_pred_positions,-1)
        mlm_Y_hat=self.mlp(masked_X)
        return mlm_Y_hat

# 下一个句子预测,输入[batch_size,seq_len,num_hiddens],在第一个维度上 torch.flatten,得到 [batch_size,seq_len × num_hiddens]
class NextSentencePred(nn.Module):
    def __init__(self,num_inputs):
        super(NextSentencePred,self).__init__():
        self.output=nn.Linear(num_inputs,2)

    def forward(self,X):
        return self.output(X)

class BERTModel(nn.Module):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,max_len=1000,key_size=768,query_size=768,value_size=768,hid_in_features=768,mlm_in_features=768,nsp_in_features=768):
        super(BERTModel,self).__init__()
        self.encoder=BERTEncoder(vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,max_len,key_size,query_size,value_size)
        self.mlm=MaskLM(vocab_size,num_hiddens,mlm_in_features)
        
        self.hidden=nn.Sequential(nn.Linear(hid_in_features,num_hiddens),nn.Tanh())
        self.nsp=NextSentencePred(nsp_in_features)

    def forward(self,tokens,segments,valid_lens=None,pred_positions=None):
        encoded_X=self.encoder(tokens,segments,valid_lens)
        if pred_positions is not None:
            mlm_Y_hat=self.mlm(encoded_X,pred_positions)
        else:
            mlm_Y_hat=None
        nsp_Y_hat=self.nsp(self.hidden(encoded_X[:,0,:])) # 取 <cls>这个来做预测
        return encoded_X,mlm_Y_hat,nsp_Y_hat


