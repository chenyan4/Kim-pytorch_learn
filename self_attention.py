# 自注意力和位置编码
# 对一个给定序列 X1-Xn, 其中 X是长为d的序列
# 自注意力为 Yi = f(Xi,(X1,X1),....,(Xn,Xn))) 即 X1-Xn，（X1,X1）表示 即为key 又是value，把 Xi当做query，求它和其他 X1-Xn的注意力 就得到Yi

# 位置编码：将位置信息注入到输入里
# 跟CNN/RNN不同，自注意力并没有记录位置信息（序列随机打乱 也是可以的，向量内容是不变的）
# 假设长度为 n的序列 X[n,d],那么使用位置编码矩阵 P[n,d] 来输出 X+P作为自编码输入
# P(i,2j)=sin(i/10000^(2j/d)) , P(i,2j+1)=cos(i/10000^(2j/d)) 偶数列是 sin,奇数列是 cos，每一行的 i是不同的，但同列的 分母是一样的

# 多头注意力
import math
import torch
from torch import nn

# 位置编码 ，因为 sin和cos是成对出现的，所以 num_hiddens 必需是偶数
class PositionalEncoding(nn.Module):
    # num_hiddens 是序列长度，max_len就是有多少个序列
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.p=torch.zeros(size=(1,max_len,num_hiddens)) # [batch_size,max_lens,seq_size]

        # for i in range(max_len):
        #     for j in range(num_hiddens//2):
        #         self.p[:,i,2*j]=math.sin(i/math.pow(10000,2*j/num_hiddens))
        #         self.p[:,i,2*j+1]=math.cos(i/math.pow(10000,2*j/num_hiddens))

        X=torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,2*torch.arange(0,num_hiddens//2,dtype=torch.float32)/num_hiddens) # [max_len,num_hiddens//2]
    
        self.p[:,:,0::2]=torch.sin(X)
        self.p[:,:,1::2]=torch.cos(X)

    def forward(self,X):
        X=X+self.p[:,:X.shape[1],:].to(X.device) # 去取到 X一共有多少个序列
        return self.dropout(X)

encoding_dim,num_steps=32,60
pos_encoding=PositionalEncoding(encoding_dim,0)
X=pos_encoding(torch.zeros(size=(1,num_steps,encoding_dim)))
print(X[:,:5,:])

    
