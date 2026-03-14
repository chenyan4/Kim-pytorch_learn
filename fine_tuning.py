# 微调：复用 之前模型的特征提取部分，修改最后输出层为自己的
# 重点训练输出层，微调 特征提取
# 源数据集 远大于 自己数据集训出来的模型
# 底层是 提取部分，到 更高层 越接近你识别的种类
# 固定 一些层的参数

import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from PIL import Image
from ResNet import accuracy,evaluate_accuracy,draw_loss_acc


hotdog_path="/data/chenyan/pytorch_learn/data/hotdog"
img=os.path.join(hotdog_path,'train/hotdog/1.png')
img=Image.open(img)
print(img.size)

# datasets.ImageFolder(path,transform) 会根据你路径文件夹下的不同 子文件夹划分类别，例如：第一个文件夹 类别0，第二个文件夹 类别1
def load_hot_dog_data(batch_size,image_size=224):
    normalize=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    # transforms.RandomResizedCrop(image_size,scale=(0.8,1)) 随机裁剪+缩放
    train_augs=transforms.Compose([transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),normalize])

    test_augs=transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),normalize])

    train_data=datasets.ImageFolder(os.path.join(hotdog_path,'train'),transform=train_augs)
    test_data=datasets.ImageFolder(os.path.join(hotdog_path,'test'),transform=test_augs)

    train_iter=DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
    test_iter=DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4)

    return train_iter,test_iter

# 第二种 继承Dataset类，本质就是把 图片路径和类别 打包成一个元组 放到一个列表
# class hot_dog(Dataset):
#     def __init__(self,path,is_train,transform):
#         split='train' if is_train else 'test'
#         data_path=os.path.join(path,split)

#         self.data=[]
#         self.classes=["hotdog","not-hotdog"]
#         self.classes_idx={"hotdog":0,"not-hotdog":1}
#         self.transform=transform

#         for cls in self.classes:
#             cls_dir=os.path.json(data_path,cls)
#             # os.listdir(dir) 列出 目录下的 文件名和子目录名
#             for file_name in os.listdir(cls_dir):
#                 self.data.append((os.path.join(cls_dir,file_name),self.classes_idx[cls]))

#     def __len__(self):
#         return len(self.data)

#     def _getitem__(self,idx):
#         image,label=self.data[idx]
#         image=self.transform(Image.open(image))
#         return image,label
        
def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

finetune_net=torchvision.models.resnet18(pretrained=True)
# finetune_net=torchvision.models.resnet18()
finetune_net.fc=nn.Linear(finetune_net.fc.in_features,2)
init_weights(finetune_net.fc)

def train_ch13(net,train_iter,test_iter,num_epochs,loss,updater,devices):
    if isinstance(net,nn.Module):
        net.to(devices[0])
    net=nn.DataParallel(net,device_ids=devices)
    train_acc,train_loss,test_acc=[],[],[]
    for epoch in range(num_epochs):
        if isinstance(net,nn.Module):
            net.train()
        acc_num,l_num,num=0,0,0
        for x,y in train_iter:
            if isinstance(x,list):
                x=[a.to(devices[0] for a in x)]
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
            num+=len(x)
        
        train_acc.append(acc_num/num)
        train_loss.append(l_num/num)
        test_acc.append(evaluate_accuracy(net,test_iter,'cuda:0'))

        print(f'epoch:{epoch+1},train_loss:{train_loss[-1]},train_acc:{train_acc[-1]},test_acc:{test_acc[-1]}')

    return train_acc,train_loss,test_acc

def train_fine_tunning(net,lr,batch_size=128,num_epochs=10,param_group=True):
    train_iter,test_iter=load_hot_dog_data(batch_size=batch_size)
    devices=['cuda:0','cuda:1']
    loss=nn.CrossEntropyLoss(reduction='none') # 默认 reduction='mean'，损失要除以batch_size;当none时，就不用除以batch_size,固然梯度是，mean的几倍，下降步幅会更大

    # 区分训练
    if param_group:
        param_class=[param for name,param in net.named_parameters() if name not in ['fc.weight','fc.bias']] # net.named_parameters()取出层名和参数，但最后只保留参数
        # 其实 parameters()返回的都是[w1,b1,w2,b2]之类的数组，函数内部应该是有 字典检索功能
        updater=torch.optim.SGD([{'params':param_class},{'params':net.fc.parameters(),'lr':lr*10}],lr=lr,weight_decay=0.001) # 最后层用大一些的学习率，其它层用小一些学习率
    else:
        updater=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=0.001) # weight_decay就是 L2正则， loss=...+wd/2*(|w|)^2，求导就是 wd*param*lr+lr*param.grad; momentum就是动量，累加历史梯度，v=momentum*v+(1-momentum)*grad,param=param-lr*v

    train_acc,train_loss,test_acc=train_ch13(net,train_iter,test_iter,num_epochs,loss,updater,devices)
    draw_loss_acc(train_acc,train_loss,test_acc,"hotdog")
    
if __name__=="__main__":
    train_fine_tunning(finetune_net,lr=5e-5)



