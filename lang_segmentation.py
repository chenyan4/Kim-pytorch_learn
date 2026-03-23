# 语义分割：将图片中的 每个像素分类到对应类别

# 语义分割数据集
import os 
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

base_url="/data/chenyan/pytorch_learn/data/VOCdevkit/VOC2012"

def pil2tensor(image):
    image=image.convert("RGB")
    image_array=np.array(image,dtype=np.uint8)
    image_tensor=torch.from_numpy(image_array).permute(2,0,1)
    return image_tensor.float()/255.0

def read_voc_images(voc_dir,is_train=True):
    if is_train:
        txt_fname=os.path.join(voc_dir,"ImageSets/Segmentation","train.txt")
    else:
        txt_fname=os.path.join(voc_dir,"ImageSets/Segmentation","val.txt")
    with open(txt_fname,'r') as f:
        images=[x for x in f.read().split('\n') if x] # f.read()读成一个大字符串('dog/ncat/nbird') ，如果文本末有换行，注意要去掉
    features,labels=[],[]
    mode=torchvision.io.image.ImageReadMode.RGB
    for image in images:
        # feature=torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{image}.jpg'),mode) # 读取的是(C,H,W),0-255
        # label=torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{image}.png'),mode)

        feature=Image.open(os.path.join(voc_dir,'JPEGImages',f'{image}.jpg'))
        label=Image.open(os.path.join(voc_dir,'SegmentationClass',f'{image}.png'))

        feature=pil2tensor(feature)
        label=label.convert('RGB')
        label=torch.from_numpy(np.array(label,dtype=np.uint8)).permute(2,0,1)

        features.append(feature)
        labels.append(label)
    return features,labels

def show_images(imgs,num_rows=2,num_cols=5,save_name=None):
    fig,axes=plt.subplots(num_rows,num_cols,figsize=(12,8))
    for idx,ax in enumerate(axes.flatten()):
        ax.imshow(imgs[idx])
        ax.axis("off") # 关闭坐标轴
    plt.tight_layout()
    plt.savefig(f'/data/chenyan/pytorch_learn/data/images/{save_name}.png',dpi=300)
    plt.close()

# 定义 每个RGB颜色所属的类别
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

# RGB 对应 类别
def voc_colormap2label():
    # R、G、B三个通道，每个取值有256总可能，组合就有 256**3，16777216种 
    colormap2label=torch.zeros(256**3,dtype=torch.long)
    for i,colormap in enumerate(VOC_COLORMAP):
        # 最大上限是 255*256*256 + 255*256 + 255 = 16777215 没超上限，类似于 10进制，这里是 256进制
        colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]]=i
    return colormap2label

# 将 voc标签的RGB值到类别索引,colormap (C,H,W)
def voc_label_indices(colormap,colormap2label):
    # 先转成 （H，W，C）
    colormap=colormap.permute(1,2,0).numpy().astype('int32')
    idx=(colormap[:,:,0]*256*256+colormap[:,:,1]*256+colormap[:,:,2]).reshape(-1) # （H，W），转变成一维 (H×W,)
    return colormap2label[idx].reshape((colormap.shape[0],colormap.shape[1])) # 返回 每个像素点 对应的种类，(H，W)

# 做 数据增强
def voc_rand_crop(feature,label,height,width):
    rect=torchvision.transforms.RandomCrop.get_params(feature,(height,width)) # 要 裁剪的高宽，
    feature=torchvision.transforms.functional.crop(feature,*rect)
    label=torchvision.transforms.functional.crop(label,*rect)
    return feature,label

class VOCSegDataSets(Dataset):
    def __init__(self,is_train,crop_size,voc_dir):
        self.transform=transforms.Normalize(mean=[0.485,0.456,0.409],std=[0.229,0.224,0.225])
        features,labels=read_voc_images(base_url,is_train)
        self.crop_size=crop_size
        self.features=[self.transform(image) for image in self.filter(features)]
        self.labels=self.filter(labels) # 保证还在 0-255
        self.colormap2label=voc_colormap2label()

    def filter(self,images):
        return [image for image in images if (image.shape[1]>=self.crop_size[0] and image.shape[2]>=self.crop_size[1])]

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        feature,label=voc_rand_crop(self.features[idx],self.labels[idx],self.crop_size[0],self.crop_size[1])
        return (feature,voc_label_indices(label,self.colormap2label))

def load_data_voc(batch_size,crop_size):
    train_data,test_data=VOCSegDataSets(True,crop_size,base_url),VOCSegDataSets(False,crop_size,base_url)
    train_iter=DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
    test_iter=DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4,pin_memory=True)

    return train_iter,test_iter


if __name__=="__main__":
    pass  # 下面均为注释时必须有语句，否则触发 IndentationError
    # n=5
    # train_features,train_labels=read_voc_images(base_url,is_train=True)
    # imgs=train_features[0:n]+train_labels[0:n]
    # show_images(imgs,save_name="seg_feat_label")
    # h,w=train_labels[0].shape[1],train_labels[0].shape[2]
    # print(h,w)
    # y=voc_label_indices(train_labels[0],voc_colormap2label())
    # print(y.reshape((h,w))[105:115,130:140])

    # imgs_features,imgs_labels=[],[]
    # for _ in range(5):
    #     result=voc_rand_crop(train_features[0],train_labels[0],100,200)
    #     imgs_features.append(result[0])
    #     imgs_labels.append(result[1])
    # show_images(imgs_features+imgs_labels,save_name="fea_label_expand")

    # crop_size=(320,480)
    # voc_train=VOCSegDataSets(True,crop_size,base_url)
    # voc_test=VOCSegDataSets(False,crop_size,base_url)

    # batch_size=64
    # train_iter=DataLoader(voc_train,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=8,pin_memory=True)

    # for x,y in train_iter:
    #     print(x.shape,y.shape)
    #     break


        
    
    
    