# 目标检测

# 1.边缘框实现
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import pandas as pd
import torchvision
import numpy as np
from torch.utils.data import Dataset,DataLoader

image=Image.open('/data/chenyan/pytorch_learn/data/images/catdog.jpg')
print(image.size)

def pil2tensor(image):
    image = image.convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0  # (H, W, C)
    image_tensor = torch.tensor(image_array,dtype=torch.float32)
    return image_tensor.permute(2, 0, 1)  # (C, H, W)，符合 PyTorch 约定

def box_corner_to_center(boxes):
    # 从（左上，右下）转换到（中间，宽度，高度）
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3] # boxes[:,0:1] 才会保留列的维度，boxes[:,0]只会保留一个维度
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    print(cx)
    boxes = torch.stack([cx, cy, w, h], dim=-1)  # 现在维度是（N），dim=-1 指(N,)这个位置
    return boxes

def bbox_to_rect(bbox,color):
    return patches.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)

def draw_boxes(is_train,num_rows=2,num_cols=5,color="white"):
    if is_train:
        data_url='bananas_train'
    else:
        data_url='bananas_val'

    csv_data = pd.read_csv(os.path.join(base_url, data_url, 'label.csv'))
    n = num_rows * num_cols
    images = csv_data['img_name'].iloc[:n]
    boxes = csv_data.iloc[:n, 2:]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8)) # fig是整块画布（figure），axes是子图数组
    axes = axes.flatten()  # 2x5 时 axes 是二维，flatten 成 10 个单独子图,展开方便循环
    for idx, ax in enumerate(axes):
        img_name = images.iloc[idx]
        img_path = os.path.join(base_url, data_url, f'images/{img_name}')
        img = Image.open(img_path)
        box = list(boxes.iloc[idx].values)  # boxes 已是第 2 列起，一行 4 个数
        ax.imshow(img)
        ax.add_patch(bbox_to_rect(box, color))
        ax.axis('off')

    plt.tight_layout() # 自动调整子图间 间距，避免重叠
    plt.savefig('/data/chenyan/pytorch_learn/data/images/od.png') 




base_url='/data/chenyan/pytorch_learn/data/banana-detection'
def read_data_bananas(is_train=True):
    if is_train:
        data_url='bananas_train'
    else:
        data_url='bananas_val'
    csv_fname=os.path.join(base_url,data_url,'label.csv')
    csv_data=pd.read_csv(csv_fname)
    csv_data=csv_data.set_index("img_name") # 将 img_name做成行索引
    images,targets=[],[]
    # csv.iterrows(), 把行索引和这一行的值取出
    for img_name,target in csv_data.iterrows():
        image_temp=Image.open(os.path.join(base_url,data_url,f'images/{img_name}'))
        images.append(pil2tensor(image_temp))
        # images.append(torchvision.io.read_image(os.path.join(base_url,data_url,img_name))) # torchvision.io.read_image(path) 把图片读到内存中，并直接得到pytorch张量
        targets.append(target.values)

    return images,torch.tensor(targets).unsqueeze(1)/256 # 在第一个维度上 补充一个物体数量维度,除以 256是为了把坐标归一化

class BananaDataset(Dataset):
    def __init__(self, is_train, transform=None):
        self.features,self.labels=read_data_bananas(is_train)
        # if is_train:
        #     self.data_url = 'bananas_train'
        # else:
        #     self.data_url = 'bananas_val'
        # self.csv = pd.read_csv(os.path.join(base_url, self.data_url, 'label.csv'))
        # self.images = self.csv['img_name']        # 用 self.csv，不是 csv
        # self.labels = self.csv.iloc[:, 1:]        # 去掉 img_name 列，保留 label,xmin,ymin,xmax,ymax

    def __len__(self):
        return len(self.features)
        # return len(self.images)

    def __getitem__(self, idx):
        return (self.features[idx],self.labels[idx])
        # img_name = self.images.iloc[idx]          # 单张图文件名，如 '0.png'，只有一列，不用values也可以
        # image_path = os.path.join(base_url, self.data_url, 'images', img_name)
        # image = Image.open(image_path)
        # image = transform(image)
        # label = self.labels.iloc[idx].values.astype(np.float32)  # 一行：label,xmin,ymin,xmax,ymax，values转成numpy
        # return image, torch.tensor(label)

def load_data_bananas(batch_size):
    train_data=BananaDataset(is_train=True)
    val_data=BananaDataset(is_train=False)

    train_iter=DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
    val_iter=DataLoader(val_data,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4)

    return train_iter,val_iter

# 示例：左上(0,0)、右下(100,80) 的框
if __name__ == "__main__":
    # 形状 (N, 4)，每行是一个框的 (x1, y1, x2, y2)
    corner_boxes = torch.tensor([[0.0, 0.0, 100.0, 80.0],   # 框1: 宽100, 高80
                                 [10.0, 20.0, 50.0, 60.0]])  # 框2: 宽40, 高40
    center_boxes = box_corner_to_center(corner_boxes)
    print("角点格式 (x1,y1,x2,y2):\n", corner_boxes)
    print("中心格式 (cx,cy,w,h):\n", center_boxes)
    # 框1: 中心(50, 40), 宽100, 高80
    # 框2: 中心(30, 40), 宽40, 高40

    dog_bbox,cat_bbox=[100, 100, 500.0, 500.0],[520.0, 200.0, 800.0, 520.0]
    fig=plt.imshow(image)
    fig.axes.add_patch(bbox_to_rect(dog_bbox,'red'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox,'blue'))
    plt.savefig('/data/chenyan/pytorch_learn/data/images/dog_cat.png')


    image=Image.open('/data/chenyan/pytorch_learn/data/banana-detection/bananas_train/images/0.png')
    print(image.size)

    # batch_size=32
    # train_iter,test_iter=load_data_bananas(batch_size)
    # for x,y in train_iter:
    #     print(x.shape,y.shape)
    #     break

    draw_boxes(is_train=True)

