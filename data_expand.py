# 数据增广
# 翻转：左右翻转、上下翻转（不总是可行，要看数据情况）
# 切割：随机高宽比（3/4,4/3)、随机大小(8%,100%)、随机位置
# 颜色：改变色调（红、黄、绿）、饱和度、亮度([0.5-1.5])

import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
from ResNet import accuracy,evaluate_accuracy,draw_loss_acc
from multi_GPU_simple import resnet_18

image=Image.open("/data/chenyan/pytorch_learn/data/images/15.jpg")
image.show()


def draw_grid(images,num_rows=2,num_cols=4,scale=1.5):
    assert len(images)>0

    w,h=images[0].size
    print(w,h)
    w_scale=int(w*scale)
    h_scale=int(h*scale)

    grid_w=w_scale*num_cols
    grid_h=h_scale*num_rows

    num_image=num_rows*num_cols

    bg=Image.new("RGB",(grid_w,grid_h),color=(0,0,0))

    for idx,img in enumerate(images):
        row=idx//num_cols
        col=idx%num_cols
        x=col*w_scale
        y=row*h_scale

        bg.paste(img,(x,y))

    bg.save("/data/chenyan/pytorch_learn/data/images/expand.png")



def apply(image,function,num_rows=2,num_cols=4,scale=1.5):
    Y=[function(image) for i in range(num_rows*num_cols)]
    draw_grid(Y,num_rows,num_cols,scale)

# transforms.RandomHorizontalFlip() 左右翻转
trans=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip()]) # transforms.RandomHorizontalFlip() 左右翻转
trans=transforms.Compose([transforms.Resize((224,224)),transforms.RandomVerticalFlip()]) # transforms.RandomVerticalFlip()上下翻转
trans=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))]) #transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2)) 第一个参数图片要多大，scale随机裁剪的面积占原图面积的比例在 [0.1, 1.0] 之间，ratio随机裁剪块的高宽比在 [0.5, 2.0] 之间
trans=transforms.Compose([transforms.Resize((224,224)),transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)]) #brightness=0.5只启用了亮度随机变化，其他三项都为 0 表示不变。具体是从区间 ([1-0.5, 1+0.5] ;contrast 对比度;saturation 饱和度,颜色淡浓;hue 色相随机变化，色调偏移 （都是在一个区间 增大或减小）

trans=trans=transforms.Compose([transforms.Resize((224,224)),
transforms.RandomHorizontalFlip(),
transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))])

apply(image,trans)

train_data=torchvision.datasets.CIFAR10(root='./data',train=True,download=True)
img=train_data[0][0]
img.save("/data/chenyan/pytorch_learn/data/images/cifar10.png")

normalize=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_augs=transforms.Compose([transforms.RandomResizedCrop(32,scale=(0.6,1.0),ratio=(1.0,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
test_augs=transforms.Compose([transforms.Resize(40),transforms.CenterCrop(32),transforms.ToTensor(),normalize])

def load_cifar10(train,augs,batch_size):
    dataset=datasets.CIFAR10(root='./data',train=train,transform=augs,download=False)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=train,drop_last=True,num_workers=4)

    return dataloader

def train_batch_ch13(net,x,y,loss,updater,devices):
    if isinstance(x,list):
        x=[a.to(devices[0]) for a in x]
    else:
        x=x.to(devices[0])
    y=y.to(devices[0])
    net.train()
    updater.zero_grad()
    y_hat=net(x)
    l=loss(y_hat,y)
    l.backward()
    updater.step()
    train_loss=l.sum().item()
    train_acc=accuracy(y_hat,y)

    return train_loss,train_acc

def train_ch13(net,train_iter,test_iter,loss,num_epochs,updater,devices=['cuda:0','cuda:1']):
    net.to(devices[0])
    net=nn.DataParallel(net,device_ids=devices)
    train_acc,train_loss,test_acc=[],[],[]
    for epoch in range(num_epochs):
        if isinstance(net,nn.Module): # 从测试eval模式换回train模式
            net.train()
        acc_num,l_num,num=0,0,0
        for x,y in train_iter:
            # t_loss,t_acc=train_batch_ch13(net,x,y,loss,updater,devices)
            if isinstance(x,list):
                x=[a.to(devices[0]) for a in x]
            else:
                x=x.to(devices[0])
            y=y.to(devices[0])
            updater.zero_grad()
            y_hat=net(x)
            l=loss(y_hat,y)
            l.sum().backward()
            updater.step()
            acc_num+=accuracy(y_hat,y)
            l_num+=l.sum().item()
            # acc_num+=t_acc
            # l_num+=t_loss
            num+=len(x)
        
        train_acc.append(acc_num/num)
        train_loss.append(l_num/num)
        test_acc.append(evaluate_accuracy(net,test_iter,'cuda:0'))

        print(f'epoch:{epoch+1},train_loss:{train_loss[-1]},train_acc:{train_acc[-1]},test_acc:{test_acc[-1]}')
    
    return train_acc,train_loss,test_acc
            
def init_weights(m):
    if type(m) ==nn.Linear or type(m) ==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

batch_size,net=256,resnet_18(18,3)
net.apply(init_weights)

def train_with_data_augs(train_augs,test_augs,net,lr,batch_size):
    train_iter=load_cifar10(True,train_augs,batch_size=batch_size)
    test_iter=load_cifar10(False,test_augs,batch_size=batch_size)

    loss=nn.CrossEntropyLoss(reduction='none')
    updater=torch.optim.Adam(net.parameters(),lr=lr)

    train_acc,train_loss,test_acc=train_ch13(net,train_iter,test_iter,loss,10,updater)
    draw_loss_acc(train_acc,train_loss,test_acc,"data_expand")

if __name__=="__main__":
    batch_size,net=256,resnet_18(18,3)
    net.apply(init_weights)
    train_with_data_augs(train_augs,test_augs,net,lr=0.005,batch_size=256)




