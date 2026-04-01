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

# 有些元素是 padding，在计算softmax时并不想要它，就是把那些 padding对应值设置成一个很小的负数（exp 接近0）
# X[batch_size,queries,num_keys],valid_len有两种：[batch_size,] 一个batch下所有query共用一个 valid ; [batch_size,num_queries] 更精细，一个batch 每个query有自己的 valid 

def sequence_mask(X,valid_len,value):
    num_queries=len(valid_len)
    for i in range(num_queries):
        X[i,valid_len[i].item():]=-1e6
    return X

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

print(masked_softmax(torch.rand(2,2,4),torch.tensor([[1,3],[2,4]])))