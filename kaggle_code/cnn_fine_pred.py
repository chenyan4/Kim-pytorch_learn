import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

alexnet=torchvision.models.alexnet(pretrained=True)
vgg16=torchvision.models.vgg16(pretrained=True)
googlenet=torchvision.models.googlenet(pretrained=True)
resnet18=torchvision.models.resnet18(pretrained=True)

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

image_path="/workspace/Kim-pytorch_learn/data/images/9060.jpg_wh860.jpg"
image=Image.open(image_path)

# print(f'AlexNet:{alexnet}\n')
# print(f'VGG16:{vgg16}\n')
# print(f'GoogleNet:{googlenet}\n')
# print(f'ResNet18:{resnet18}')

output=evaluate_accuracy(alexnet,image,device='cuda:0',transform=trans)

print('AlexNet:',evaluate_accuracy(alexnet,image,device='cuda:0',transform=trans))
print('VGG16:',evaluate_accuracy(vgg16,image,device='cuda:0',transform=trans))
print('GoogleNet:',evaluate_accuracy(googlenet,image,device='cuda:0',transform=trans))
print('ResNet18:',evaluate_accuracy(resnet18,image,device='cuda:0',transform=trans))