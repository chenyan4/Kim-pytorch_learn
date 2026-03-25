# 序列模型：是有时序结构，变量是不独立的
import torch
from torch import nn
import matplotlib.pyplot as plt

T=1000
time=torch.arange(start=1,end=T+1,step=1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(mean=0,std=0.2,size=(T,))

plt.figure(figsize=(12,6))
plt.plot(time.tolist(),x.tolist(),color='b',linestyle='-',linewidth=2) # plt.plot(x,y) x，y轴数值
plt.xlabel('Time')
plt.ylabel('X')
plt.grid(True) # 显示网格
plt.xlim(1,1000) # 限制 x轴范围
# plt.ylin(0,5)
# plt.xticks(x) 强制 X轴显示你的内容
plt.savefig("/data/chenyan/pytorch_learn/data/images/time.png",dpi=300)
plt.close()
