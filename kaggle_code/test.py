import torch
from torch import nn
import torchvision

finetune_net=nn.Sequential()
finetune_net.features=torchvision.models.resnet34(pretrained=True) #相当于 把resnet做成一部分，features
finetune_net.output=nn.Sequential(nn.Linear(1000,256),nn.ReLU()) # 作为一部分添加到 sequential中

# print(finetune_net)
# finetune_net=torchvision.models.resnet34(pretrained=True) # 直接就是resnet的本体了
# finetune_net.output=nn.Sequential(nn.Linear(1000,256)) # 添加到resnet中
print(finetune_net)