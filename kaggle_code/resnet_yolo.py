import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import json
from matplotlib import patches

device='cuda:0'

with open("/data/chenyan/pytorch_learn/data/cowboys/train.json","r",encoding='utf-8') as f:
    data=json.load(f)
    images=data['images']
    categories=data['categories']
    annotations=data['annotations']

def boxes_xywh_to_center(bboxes):
    x,y,w,h=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]

    cx=x+w/2
    cy=y+h/2

    return torch.stack([c,cx,cy,w,h],dim=1s)

def boxes_xywh_to_xyxy(bboxes):
    x,y,w,h=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
    x1=x
    y1=y
    x2=x+w
    y2=y+h
    return torch.stack([x1,y1,x2,y2],dim=1)

def boxes_corner_to_center(bboxes):
    x1,y1,x2,y2=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]

    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1

    return torch.stack([cx,cy,w,h],dim=1) 

def boxes_center_to_corner(bboxes):
    cx,cy,w,h=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]

    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2

    return torch.stack([x1,y1,x2,y2],dim=1)

def image_pillow(images,size=640,color=(0,0,0)):
    new_images,ratios=[],[]
    for image in images:
        w,h=image.size
        max_line=max(w,h)

        ratio=size/max_line
        target_w,target_h=int(w*ratio),int(h*ratio)

        image=image.resize((target_w,target_h),Image.Resampling.BILINEAR)
        new_image=Image.new("RGB",(size,size),color)
        new_image.paste(image,(0,0))

        new_images.append(new_image)
        ratios.append(ratio)
    
    return new_images,ratios

def bbox_pillow(annotations,ratios,size=640):
    for i in range(len(ratios)):
        annotations[i,:,1:]=(annotations[i,:,1:]*ratios[i])/size
    return annotations

def read_cowboy_data(images,annotations,categories,max_anchors=2):
    categories_idx,images_idx,annotations_idx={},[],[]
    for idx,cae in enumerate(categories):
        categories_idx[idx]=cae['id']
    print(categories_idx)

    for idx,image in enumerate(images):
        img=Image.open(os.path.join("/data/chenyan/pytorch_learn/data/cowboys/images",image['file_name']))
        images_idx.append(img)
        id=image['id']
        ana_idx=[]
        for ana_value in annotations:
            if ana_value['image_id']==id and len(ana_idx)<max_anchors:
                ana=[ana_value['category_id']]+ana_value['bbox']
                ana_idx.append(ana)
        if len(ana_idx)<max_anchors:
            ana_idx=ana_idx+[[-1]*5]*(max_anchors-len(ana_idx))
        annotations_idx.append(ana_idx)

    images_idx,ratios=image_pillow(images)
    annotations_idx=bbox_pillow(torch.tensor(annotations_idx,device=device),ratios)
    
    return images_idx,torch.tensor(annotations_idx),categories_idx

class CowBoyDataset(Dataset):
    def __init__(self,images,annotations,transform):
        self.images=images
        self.annotations=annotations
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image=transform(self.images[idx])
        return image,annotations[idx]

def load_cowboy_data(images,annotations,batch_size):
    trans=transforms.Compose([transforms.ToTensor(),torch.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    train_data=CowBoyDataset(images,annotations,trans)
    train_iter=DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)

    return train_iter
    

def multibox_prior(data,sizes,ratios):
    in_height,in_width=data.shape[-2:]
    num_size,num_ratio=len(sizes),len(ratios)
    num_anchors=num_size+numratio-1

    sizes=torch.tensor(sizes,device=device)
    ratios=torch.tensor(ratios,device=device)


    step_h=1/in_height
    step_w=1/in_width

    offset_h,offset_w=0.5,0.5

    center_h=(torch.arrange(range(in_height))+offset_h)*step_h
    center_w=(torch.arrange(range(in_width))+offset_w)*step_w

    shift_y,shift_x=torch.meshgrid(center_h,center_w,index='ij')
    shift_y,shift_x=shift_y.reshape(-1),shift_x.reshape(-1)

    w=torch.cat(sizes[0]*torch.sqrt(ratios[1:]),size[1:]*torch.sqrt(ratios[0]))*(in_height/in_width)
    h=torch.cat(sizes[0]/torch.sqrt(ratios[1:]),size[1:]/torch.sqrt(ratios))

    wh_grid=torch.stack([-w,-h,w,h],dim=1).repeat(in_height*in_width,1)/2
    out_grid=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1).repeat_interleave(num_anchors,dim=0)
    out_grid=out_grid+wh_grid

    return out_grid.unsqueeze(0)

def box_iou(boxes1,boxes2):
    area1=(boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])
    area2=(boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])

    upleft_xy=torch.max(boxes1[:,None,:2],boxes2[:,:2])
    downright_xy=torch.min(boxes1[:,None,2:],boxes2[:,2:])

    inner=downright_xy-inner_upperlefts
    inner=inner*(inner>=0) # 如果 宽高有负数，那么 面积相乘会是0
    inner_area=inner[:,:,0]*inner[:,:,1]
    union_area=(area1[:,None]+area2)-inner_area

    return inner_area/union_area

def assign_anchor_to_bbox(data,num_anchor,ground_truth,anchors):
    in_height,in_width=data[-2:]
    num_anchors=len(anchors)

    step_h,step_w=1/in_height,1/in_width
    ground_truth_center=boxes_xywh_to_center(ground_truth)

    anchors_bbox_map=torch.full((num_anchors,),fill_value=-1,device=device)
    anchors_conf_map=torch.full((num_anchors,),fill_value=0,device=device)
    set_zeros=torch.zeros((4,),device=device)

    for idx,bbox in enumerate(ground_truth_center):
        if bbox[0]==-1:
            break
        bbox_cx,bbox_cy=bbox[0],bbox[1]
        idx,idy=bbox_cx//step_w,bbox_cy//step_h
        idx_anchors=idy*in_width*num_anchor+idx*num_anchor

        select_anchors=anchors[idx_anchors:idx_anchors+num_anchor]

        bboxes_iou=box_iou(select_anchors,ground_truth[idx].unsqueeze(0))
        max_iou,indice=torch.max(bboxes_iou,dim=0)

        anchor_idx=idx_anchors+indice.item()
        anchors_bbox_map[anchor_idx]=idx
        anchors_conf_map[anchor_idx]=max_iou.item()
        anchors[anchor_idx]=set_zeros


    return anchors_bbox_map,anchors_conf_map

def offset_boxes(anchors,assigned_bb,eps=1e-6):
    c_anc=boxes_corner_to_center(anchors)
    c_assigned_bb=boxes_xywh_to_center(assigned_bb)

    offset_xy=10*(c_assigned_bb[:,:2]-c_anc[:,:2])/c_anc[:,2:]
    offset_wh=5*torch.log((eps+c_assigned_bb[:,2:])/c_anc[:,2:])

    return torch.cat([offset_xy,offset_wh],axis=1)

def multibox_target(data,num_anchor,anchors,labels):
    batch_size,anchors=labels.shape[0],anchors.squeeze(0)
    num_anchors=anchors.shape[0]

    batch_bbox_mask,batch_class_mask,batch_offset,batch_conf,batch_classes_labels=[],[],[],[],[]
    for i in range(batch_size):
        label=labels[i]
        anchors_bbox_map,anchors_conf_map=assign_anchor_to_bbox(data,num_anchor,label[:,1:],anchors)
        bbox_mask=(anchors_bbox_map>=0).float().unsqueeze(-1).repeat(1,4)
        class_mask=(anchors_bbox_map>=0).float()


        box_idx=torch.nonzero(anchors_bbox_map>=0).reshape(-1)
        assigned_bb=torch.zeros((num_anchors,4),device=device,dtype=torch.float32)
        class_labels=torch.full((num_anchors,),,device=device,fill_value=-1)

        assigned_bb[box_idx]=label[box_idx,1:]
        class_labels[box_idx]=label[box_idx,0]

        offset_box=offset_boxes(anchors,assigned_bb)

        batch_bbox_mask.append(bbox_mask)
        batch_class_mask.append(class_mask)
        batch_offset.append(offset_box)
        batch_conf.append(anchors_conf_map)
        batch_classes_labels.append(class_labels)
    
    bboxes_bbox_mask=torch.stack(batch_mask)
    bboxes_class_mask=torch.stack(batch_class_mask)
    bboxes_offset=torch.stack(batch_offset)
    bboxes_conf=torch.stack(batch_conf)
    bboxes_labels=torch.stack(batch_classes_labels)

    return bboxes_bbox_mask,bboxes_class_mask,bboxes_offset,bboxes_conf,bboxes_labels

def offset_inverse(anchors,offset_pred):
    c_anc=boxes_corner_to_center(anchors)

    offset_xy=(c_anc[:,2:]*offset_pred[:,:2])/10+c_anc[:,:2]
    offset_wh=torch.exp(offset_pred[:,2:]/5)*c_anc[:,2:]

    offset_invers=torch.stack([offset_xy,offset_wh],dim=1)
    offset_invers=boxes_corner_to_center(offset_inves)
    return offsset_invers

def nms(boxes,scores,iou_threshold):
    B=torch.argsort(scores,dim=-1,descending=True)
    keep=[]
    while B.numel()>0:
        i=B[0]
        keep.append(i)
        if B.numel()==1:
            break
        boxes_iou=box_iou(boxes[i,:].unsqueeze(0),boxes[B[1:],:]).reshape(-1)

        box_idx=torch.nonzero(boxes_iou<=iou_threshold).reshape(-1)

        if box_idx.numel()==0:
            break
        B=B[box_idx+1]
    return torch.tensor(keep,device=device)

def multibox_detection(cls_probs,offset_preds,anchors,num_threahold=0.5,pos_threshold=0.00999):
    anchors,batch_size=anchors.squeeze(0),cls_probs.shape[0]
    num_classes,num_anchors=cls_probs[1],cls_probs[2]
    out=[]

    for i in range(batch_size):
        cls_prob=cls_probs[i]
        offset_pred=offset_preds[i].reshape(-1,5)
        offset_invers=offset_inverse(anchors,offset_pred[:,1:])

        box_conf=offset_pred[:,1]
        class_conf,class_idx=torch.max(cls_prob,dim=0)
        conf=box_conf*class_conf

        keep=nms(offset_invers,conf,num_threahold)
        keep_list=keep.tolist()
        non_keep=[]
        for idx in range(len(conf)):
            if idx not in keep_list:
                non_keep.append(idx)

        non_keep=torch.tensor(non_keep,device=device)
        class_idx[non_keep]=-1
        anchors_all_idx=torch.cat([keep,non_keep])

        offset_invers=offset_invers[anchors_all_idx]
        class_idx=class_idx[anchors_all_idx]
        conf=conf[anchors_all_idx]

        below_conf=torch.nonzero(conf<pos_threshold,device=device).reshape(-1)

        class_idx[below_conf]=-1

        pred=torch.cat([class_idx.unsqueeze(0),box_conf.unsqueeze(0),offset_invers],dim=1)

        out.append(pred)

    return torch.stack(out)


class Resduial(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1conv:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

        self.relu=nn.ReLU()
    
    def forward(self,X):
        Y=self.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y=Y+X
        return self.relu(Y)

def resnet_block(in_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Resduial(in_channels,num_channels,use_1conv=True,strides=2))
        else:
            blk.append(Resduial(num_channels,num_channels))
    return nn.Sequentail(*blk)

def class_conv(in_channels,num_classes,num_anchor):
    return nn.Conv2d(in_channels,num_classes*num_anchor,kernel_size=3,padding=1)

def bbox_conv(in_channels,num_anchor):
    return nn.Conv2d(in_channels,num_anchor*5,kernel_size=3,padding=1)

b1=nn.Sequentail(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b2=resnet_block(64,64,2,True)
b3=resnet_block(64,128,2)
b4=resnet_block(128,256,2)
b5=resnet_block(256,512,2)

bone_net=nn.Sequential(b1,b2,b3,b4,b5)

sizes=[0.2,0.4,0.5]
ratios=[2,1,0.5]
num_anchor=len(sizes)+len(ratios)-1

class Yolov2(nn.Module):
    def __init__(self,bone_net):
        super(Yolov2,self).__init__()
        self.bone_net=bone_net
        self.class_conv=class_conv(512,num_anchor*5)
        self.bbox_conv=bbox_conv(512,num_anchor*5)

    def forward(self,X):
        X=self.bone_net(X)
        anchors=multibox_prior(X,sizes,ratios)
        class_preds=self.class_conv(X)
        bbox_preds=self.bbox_conv(X)

        batch_size=len(class_preds)
        class_preds=class_preds.permute(0,2,3,1)
        class_preds=class_preds.reshape(batch_size,-1,num_anchor*5)

        bbox_preds=bbox_preds.permute(0,2,3,1)
        bbox_preds=bbox_preds.reshape(batch_size,-1,num_anchor*5)

        return X,anchors,class_preds,bbox_preds

c_loss=nn.CrossEntropyLoss(reduction='none')
l_loss=nn.L1Loss(reduction='none')

def calc_loss(class_preds,offset_preds,conf,bbox_mask,class_mask,bboxes_offset,bboxes_conf,bboxes_labels)
        

def boxes_to_rect(bbox,color):
    return patches.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2],height=bbox[3],fill=False,edgecolor=color,linewidth=2)



def show_bbox(image,bboxes):
    fig=plt.imshow(image)
    colors=['r','g','b','w']
    for idx,bbox in enumerate(bboxes):
        bbox=bbox.detach().cpu().numpy()
        if bbox[0]!=-1:
            rect=boxes_to_rect(bbox[1:].astype(np.float64),colors[idx%4])
            fig.axes.add_patch(rect)
    plt.savefig("/data/chenyan/pytorch_learn/data/images/yolo_bbox.png",dpi=300)
    plt.close()


images_idx,annotations_idx,categories_idx=read_cowboy_data(images,annotations,categories)
new_images_idx,ratios=image_pillow(images_idx)
new_annotations_idx=bbox_pillow(annotations_idx,ratios)
new_images_idx[0].save("/data/chenyan/pytorch_learn/data/images/yolo_test.png")

show_bbox(new_images_idx[0],new_annotations_idx[0])






