# 数据增广
# 翻转：左右翻转、上下翻转（不总是可行，要看数据情况）
# 切割：随机高宽比（3/4,4/3)、随机大小(8%,100%)、随机位置
# 颜色：改变色调（红、黄、绿）、饱和度、亮度([0.5-1.5])

import torch
import torchvision
from torchvision import transforms
from torch import nn
from PIL import Image

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


apply(image,trans)


