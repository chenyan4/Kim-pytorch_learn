import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

def pil2tensor(image):
    image=image.convert("RGB")
    image_np=np.array(image).astype(np.float32)/255.0
    image_tensor=torch.tensor(image_np)
    image_tensor=image_tensor.permute(2,0,1) # 原本维度是 H×W×C，替换维度为 C×H×W
    return image_tensor

# 加载训练集 labels
train_csv=pd.read_csv("/data/chenyan/pytorch_learn/data/kaggle_data/train.csv")
test_csv=pd.read_csv("/data/chenyan/pytorch_learn/data/kaggle_data/test.csv")
train_features=train_csv.iloc[:,1]

# 177种类
train_features=pd.get_dummies(train_features,dummy_na=True)
label_index=train_features.columns.values
# print(train_features.columns.values[0])

train_labels=torch.tensor(train_features.values.astype(float),dtype=torch.float32)
train_labels=train_labels.argmax(axis=1)
train_labels=train_labels.reshape(-1)

# 加载图片数据
# image_dir="/data/chenyan/pytorch_learn/data/kaggle_data"

# train_image,test_image=[],[]
# for fname in train_csv['image']:
#     image_url=os.path.join(image_dir,fname)
#     image=Image.open(image_url)
#     image_tensor=pil2tensor(image)
#     train_image.append(image_tensor)


# for fname in test_csv['image']:
#     image_url=os.path.join(image_dir,fname)
#     image=Image.open(image_url)
#     image_tensor=pil2tensor(image)
#     test_image.append(image_tensor)

# train_data=torch.stack(train_image,dim=0)
# train_data=torch.stack(test_image,dim=0)

class LeafDataset(Dataset):
    def __init__(self,csv_data,image_dir,labels=None,train=False,transform=None):
        self.csv_data=csv_data
        self.image_dir=image_dir
        self.labels=labels
        self.train=train
        self.transform=transform

    # 获取整个数据集长度
    def __len__(self):
        return len(self.csv_data)
    # Dataloader 会传idx
    def __getitem__(self,idx):
        image_name=self.csv_data["image"][idx]
        image_path=os.path.join(self.image_dir,image_name)
        image=Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image=self.transform(image)
        else:
            image=pil2tensor(image)
        
        if self.train:
            return image,self.labels[idx].item()
        else:
            return image

image_dir="/data/chenyan/pytorch_learn/data/kaggle_data"
trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

train_dataset=LeafDataset(train_csv,image_dir,train_labels,train=True,transform=trans)
test_datasets=LeafDataset(test_csv,image_dir,train=False,transform=trans)

train_iter=DataLoader(train_dataset,batch_size=256,shuffle=True,num_workers=4)
test_iter=DataLoader(test_datasets,batch_size=256,shuffle=False,num_workers=4)

# 模型定义
# 1.ResNet
class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)

        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

        if use_1conv:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))

        if self.conv3:
            x=self.conv3(x)
        y=y+x
        return self.relu(y)

b1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(in_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(in_channels,num_channels,use_1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2=nn.Sequential(*resnet_block(64,64,4,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,4))
b4=nn.Sequential(*resnet_block(128,256,4))
b5=nn.Sequential(*resnet_block(256,512,4))

resnet_net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,177))

def init_weights(m):
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight,0,0.01)

def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1)
    y=y.reshape(y_hat.shape)
    cmp=(y_hat==y).sum().item()
    return cmp

def test_write_csv(net,test_iter,test_csv,device):
    if isinstance(net,nn.Module):
        net.eval()
    label=[]
    for x in test_iter:
        if isinstance(x,list):
            x=[a.to(deivce) for a in x]
        else:
            x=x.to(device)
        y_hat=net(x)
        y_hat=y_hat.argmax(axis=1).reshape(-1)
        for index in y_hat:
            label.append(label_index[index])
    
    test_csv['label']=label
    test_csv.to_csv("/data/chenyan/pytorch_learn/data/output/submission.csv",index=False)




def train_ch6(net,train_iter,test_iter,num_epochs,loss,lr,updater,device=None):
    if device is None:
        device=next(iter(net.parameters())).device
    if isinstance(net,nn.Module):
        net.to(device)
    train_acc,train_loss=[],[]
    for epoch in range(num_epochs):
        acc_num,l_num,num=0,0,0
        for x,y in train_iter:
            if isinstance(x,list):
                x=[a.to(device) for a in x]
            else:
                x=x.to(device)
            y=y.to(device)
            updater.zero_grad()
            y_hat=net(x)
            l=loss(y_hat,y)
            l.backward()
            updater.step()

            acc_num+=accuracy(y_hat,y)
            l_num+=l.item()
            num+=len(x)

        train_acc.append(acc_num/num)
        train_loss.append(l_num/num)

        print(f'epoch:{epoch+1},train_acc:{train_acc[-1]},train_loss:{train_loss[-1]}')

    torch.save(net.state_dict(),"/data/chenyan/pytorch_learn/data/net_weights/leaf_ResNet16.params")
    test_write_csv(net,test_iter,test_csv,device=device)
    return train_acc,train_loss

def draw_loss_acc(train_acc,train_loss):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label="train_loss",color="blue",linestyle="-",linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_acc,label="train_acc",color="red",linestyle="-",linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend("upper right")

    plt.savefig("/data/chenyan/pytorch_learn/data/output/Resnet_leaf.png",dpi=300)
    plt.show()

if __name__=="__main__":
    lr,num_epochs=0.02,20
    loss=nn.CrossEntropyLoss()
    updater=torch.optim.SGD(resnet_net.parameters(),lr=lr)
    train_acc,train_loss=train_ch6(resnet_net,train_iter,test_iter,num_epochs,loss,lr,updater,device="cuda:0")
    draw_loss_acc(train_acc,train_loss)




    
    






