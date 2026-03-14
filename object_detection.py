# 目标检测

# 1.边缘框实现
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import pandas as pd

image=Image.open('/data/chenyan/pytorch_learn/data/images/catdog.jpg')
print(image.size)

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

bash_url='/data/chenyan/pytorch_learn/data/banana-detection'
def read_data_bananas(is_train=True):
    if is_train:
        data_url='bananas_train'
    else:
        data_url='bananas_val'
    csv_fname=os.path.join(bash_url,data_url,'label.csv')
    csv_data=pd.read_csv(csv_fname)


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
