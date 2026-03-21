# 全连接卷积神经网络（FCN）
# 用深度学习做语义分割的 开创性工作
# 用转置卷积层，替换全连接层

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
from torch.nn import functional as F
from lang_segmentation import load_data_voc
from ResNet import draw_loss_acc
import numpy as np

# 用预训练的resnet18 抽取特征
resnet18=torchvision.models.resnet18(pretrained=True)
# print(list(resnet18.children())[:-2])
net=nn.Sequential(*list(resnet18.children())[:-2]) # children用来提取 子模块
# print(net)

num_classes=21 # 有 21个类别
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1)) # net.add_module(名，层)


def pil2tensor(image):
    image = image.convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return image_tensor.float() / 255.0

def tensor2pil(data):
    data=data.permute(1,2,0)
    data=Image.fromarray(data.cpu().numpy().astype(np.uint8))
    return data

# k=2p+s
net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))

def loss(inputs,targets):
    batch_size,num_classes=inputs.shape[0],inputs.shape[1]
    inputs=inputs.permute(0,2,3,1).reshape(-1,num_classes)
    l=F.cross_entropy(inputs,targets.reshape(-1),reduction='none').reshape(batch_size,-1)
    return l.mean(dim=1)

def cls_eval(y_hat,y):
    num_classes=y_hat.shape[1]
    y_hat=y_hat.permute(0,2,3,1)
    y_hat=y_hat.argmax(dim=-1)
    cmp=(y_hat==y).sum().item()
    return cmp

def evluate_accuracy(net,test_iter,device):
    if isinstance(net,nn.Module):
        net.eval()
    net.to(device)
    acc_num,num=0,0
    for x,y in test_iter:
        x,y=x.to(device),y.to(device)
        y_hat=net(x)
        acc_num+=cls_eval(y_hat,y)
        num+=len(x)
    
    return acc_num/(num*320*480)



def train_ch13(net,train_iter,test_iter,num_epochs,updater,device):
    net.to(device)
    train_acc,train_loss,test_acc=[],[],[]
    for epoch in range(num_epochs):
        if isinstance(net,nn.Module):
            net.train()
        acc_num,l_num,num=0,0.0,0
        for x,y in train_iter:
            x,y=x.to(device),y.to(device)
            updater.zero_grad()
            y_hat = net(x)
            l = loss(y_hat, y)
            l.mean().backward()
            updater.step()

            acc_num+=cls_eval(y_hat,y)
            l_num+=l.mean().item()
            num+=len(x)
        train_acc.append(acc_num/(num*320*480))
        train_loss.append(l_num/num)
        test_acc.append(evluate_accuracy(net, test_iter, device))

        print(f'epoch:{epoch+1},train_acc:{train_acc[-1]},train_loss:{train_loss[-1]},test_acc:{test_acc[-1]}')

    torch.save(net.state_dict(),"/data/chenyan/pytorch_learn/data/net_weights/FCN.params")

    return train_acc,train_loss,test_acc

def predict(img, net, device):
    net.eval()
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.409], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = trans(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(x).permute(0, 2, 3, 1).argmax(dim=-1)
    return pred.reshape(pred.shape[1], pred.shape[2]).cpu()



VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# 对应类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def label2image(pred):
    H,W=pred.shape
    pred=pred.long()
    image=torch.zeros(size=(3,H,W))
    voc_colormap=torch.tensor(VOC_COLORMAP,dtype=torch.long)
    for i in range(H):
        for j in range(W):
            color_idx=pred[i,j].item()
            image[:,i,j]=voc_colormap[color_idx,:]
    image=tensor2pil(image)
    image.save('/data/chenyan/pytorch_learn/data/images/FCN_pred.png')
    

if __name__== "__main__":
    batch_size,crop_size=32,(320,480)
    train_iter,test_iter=load_data_voc(batch_size,crop_size)
    num_epochs,lr,wd,device=10,0.001,1e-3,'cuda:0'
    updater = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=wd)

    train_acc,train_loss,test_acc=train_ch13(net,train_iter,test_iter,num_epochs,updater,device)
    draw_loss_acc(train_acc,train_loss,test_acc,'FCN')

    img = Image.open(
        "/data/chenyan/pytorch_learn/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
    ).convert("RGB")
    # torchvision.transforms.functional.crop(img, top, left, height, width)，不是 PIL 的 (左,上,右,下)
    top, left, height, width = 0, 0, 320, 480
    img = transforms.functional.crop(img, top, left, height, width)
    img.save("/data/chenyan/pytorch_learn/data/images/crop.png")
    pred = predict(img, net, device)
    label2image(pred)


