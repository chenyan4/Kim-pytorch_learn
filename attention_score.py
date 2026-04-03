# 注意力分数
# 加性注意力：a(k,q)=vT @ tanh(Wk @ k,Wq @ q) 其中Wk[h,k],Wq[h,q],k [k,],q [q,];得到 [h,1] 和 VT(1,h) 相乘得到数值 1
# a(k,q)计算的是 一个key和一个query的 分数（未softmax）；实际上是一层 mlp，这里允许 k和q 长度不同

# 如果长度相同,假设 q和k 长度都为d，除以sqrt(d),是因为这么乘之后，是为了保证 方差仍为 1，因为遍历 i在所有 keys  E(取和qki)=0,var(取和qki)=d
# a(q,ki)/sqrt(d)

# 向量化版本
# Q[n,d],K[m,d],V[m,v]
# 注意力分数：a(Q,K)=QKT/sqrt(d) # [n,m] 注意力权重是 注意力分数的softmax
# 注意力池化:f=softmax(a(Q,K))V # [n,v]

import math
import torch
from torch import nn


def sequence_mask(X,valid_len,value):
    num_queries=len(valid_len)
    for i in range(num_queries):
        X[i,valid_len[i].item():]=-1e6
    return X

# 有些元素是 padding，在计算softmax时并不想要它，就是把那些 padding对应值设置成一个很小的负数（exp 接近0）
# X[batch_size,queries,num_keys],valid_len有两种：[batch_size,] 一个batch下所有query共用一个 valid ; [batch_size,num_queries] 更精细，一个batch 每个query有自己的 valid 
def masked_softmax(X,valid_len):
    if valid_len is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=torch.repeat_interleave(valid_len,X.shape[1]) # 复制 num_quries次
        else:
            valid_len=valid_len.reshape(-1)
        X=sequence_mask(X.reshape(-1,shape[-1]),valid_len,value=-1e6)
        return nn.functional.softmax(X.reshape(shape),dim=-1)

# 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout):
        super(AdditiveAttention,self).__init__()
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout)
    
    # queries[batch_size,num_quries,query_size],keys[batch_size,num_keys,key_size],values[batch_size,num_keys,value_size]
    def forward(self,queries,keys,values,valid_lens):
        queries,keys=self.W_q(queries),self.W_k(keys)
        features=queries.unsqueeze(2)+keys.unsqueeze(1) # [batch_size,num_quries,num_keys,num_hiddens]  
        features=torch.tanh(features)
        scores=self.w_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.permute(0,2,1))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)


print(masked_softmax(torch.rand(2,2,4),torch.tensor([[1,3],[2,4]])))

# 加性注意力示例：query 最后一维可与 key 不同（经 W_q/W_k 映射到同一 num_hiddens）
queries_add, keys_add = torch.normal(0, 1, size=(2, 1, 20)), torch.ones(size=(2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
# attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0)
# attention.eval()
# print(attention(queries_add, keys_add, values, valid_lens))

# 缩放点积注意力：queries 与 keys 的最后一维必须相同（均为 d）
queries = torch.normal(0, 1, size=(2, 1, 2))
keys = torch.ones(size=(2, 10, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))
