#输入输出通道
#彩色图像有RGB三个通道，转为灰度会丢失信息

#多个输入通道：每个通道都有一个卷积核，结果是所有通道卷积结果的和
# 输入X：ci×nh×nw
# 核W：ci×kh×kw
# 输出Y:mh×mw 输出是一个单通道，对多通道卷积结果相加了

#多输出通道
# 输入X：ci×nh×nw
# 核W:co×ci×kh×kw
# 输出Y：co×mh×mw
# 一一拿出核W，多个单通道concat起来就是输出了

#多个输入和输出通道：
# 每个输出通道可以识别特定模式
# 输入通道核识别并组合输入中的模式，组合多个模式判断，哪个重要哪个不重要，层数越多，就越组合，先分后合

#1×1卷积层：不会识别空间模式，只是融合通道，融合不同通道信息
# 相当于nhnw×ci，权重为co×ci的全连接层

import torch

def corr2d(X,K):
    h,w=K.shape
    y=torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return y

# 实现多输入通道运算
def corr2d_multi_in(X,K):
    # 因为X和K都是三维张量，所以可以zip打包，每一个是x和k二维张量的组合元组
    return sum(corr2d(x,k) for x,k in zip(X,K)) # 就是矩阵上相对应位置相加
#     h,w=K.shape[1:]
#     y=torch.zeros(size=(X.shape[1]-h+1,X.shape[2]-w+1))
#     for i in range(X.shape[0]):
#         y+=corr2d(X[i],K[i])
#     return y

X=torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],[[1.0,2.0,3.0],
[4.0,5.0,6.0],[7.0,8.0,9.0]]])
K=torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])


print(corr2d_multi_in(X,K))

# # 1. 准备多个形状相同的张量
# a = torch.tensor([[1, 2], [3, 4]])  # 形状 (2, 2)
# b = torch.tensor([[5, 6], [7, 8]])  # 形状 (2, 2)
# c = torch.tensor([[9, 10], [11, 12]])  # 形状 (2, 2)

# # 2. 基础用法：dim=0（默认），在第0维堆叠
# stack_dim0 = torch.stack([a, b, c], dim=0)
# print("dim=0 堆叠结果：")
# print(stack_dim0)
# print("形状：", stack_dim0.shape)  # 输出 torch.Size([3, 2, 2])

def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K])
    # c_out=len(K)
    # h,w=K.shape[2:]
    # y=torch.zeros(size=(c_out,X.shape[1]-h+1,X.shape[2]-w+1))
    # for i in range(c_out):
    #     y[i]=corr2d_multi_in(X,K[i])
    # return y

# K=torch.ones(size=(3,2,2,2))
K=torch.stack([K,K+1,K+2],dim=0)
print(corr2d_multi_in_out(X,K))

        
    
