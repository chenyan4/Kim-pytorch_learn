import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

alexnet_net=torchvision.models.alexnet(pretrained=True)

def softmax(x):
    x=x-torch.max(x,dim=1,keepdim=True)[0] # 返回的是 最大元素和 下标，元组
    x_exp=torch.exp(x)
    x_exp_sum=torch.sum(x_exp,dim=1,keepdim=True)
    return x_exp/x_exp_sum

def evaluate_accuracy(net,image,device,transform=None):
    if isinstance(net,nn.Module):
        net.eval()
    net.to(device)
    image=transform(image)
    # AlexNet 需要 (N, C, H, W)。transform 后是 (C, H, W)，所以要补 batch 维度
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image=image.to(device)
    output=net(image)
    output=softmax(output)

    return output.argmax(dim=1).reshape(-1).item()

trans=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

image_path="/data/chenyan/pytorch_learn/data/images/9060.jpg_wh860.jpg"
image=Image.open(image_path)

output=evaluate_accuracy(alexnet_net,image,device='cuda:0',transform=trans)
print(output)