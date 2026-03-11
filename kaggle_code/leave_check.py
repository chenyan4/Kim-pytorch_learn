import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch
import numpy as np
import pandas as pd
from PIL import Image
import os

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
trans=transforms.Compose([transforms.ToTensor()])

train_dataset=LeafDataset(train_csv,image_dir,train_labels,train=True,transform=trans)
test_datasets=LeafDataset(test_csv,image_dir,train=False,transform=trans)

train_iter=DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=4)
test_iter=DataLoader(test_datasets,batch_size=128,shuffle=False,num_workers=4)

# 模型定义
# 1.ResNet
class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1)
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
        y1=self.relu(self.bn1(self.conv1(x)))
        y2=self.bn2(self.conv2(y1))

        if self.conv3:
            x=self.conv3(x)
        y=y+x
        return self.relu(x)

b1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7))

        


    
    






