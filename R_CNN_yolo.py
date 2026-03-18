# 区域卷积神经网络：R-CNN、Faster RCNN、SSD、YOLO

# R-CNN：使用启发式搜索算法来 选择锚框；使用 预训练模型 对每个锚框抽取特征 ；训练一个 SVM对类别分类；训练一个 线性回归模型 预测边缘框偏移
# 如何 保证锚框大小不同；但是最后 出来的特征大小相同：兴趣区域池化层（ROI)，将锚框均匀 分割成 n×m块，输出 每块最大值，最后出来都是 n×m个值

# Fast RCNN：先对图片 用CNN 抽特征；再在图片上面 搜索锚框；再将锚框 映射到 CNN输出的特征；按锚框比例，在 特征图同样找出来；再对 特征图上找出的锚框 做ROI pooling，抽取固定大小；拉成向量，再过全连接层

# Faster RCNN：用神经网络替代 选择性搜索锚框；对图片做完 CNN抽取特征后；用特征图 通过卷积层得到 一系列锚框；然后 一系列锚框 进神经网络看 这个锚框是否有效 以及 偏移如何，再通过 NMS 筛掉重叠框，得到 质量比较好的框

# Mask RCNN：如果 有像素级别的标号，使用 FCN来利用这些信息；ROI 变成ROI align，就是对于一个 3×3，要得一个 2×2时，我直接 均匀切分，中间像素 分成4小块，是 整块的加权平均（整块的 一小块值）

# SSD (单发多框检测)：就只做一次 推理，对每个像素，生成 多个以它为中心的锚框；图片过来先抽特征，直接预测 锚框；然后 多个卷积层来 减半高宽，每段 都生成锚框（多段输出）；底部检测 小物体（不用 压太多像素），顶部拟合 大物体（多压 好预测）

# YOLO：SSD 锚框 重叠率是很高的；尽量让 锚框不重叠
# 把图片均匀分成 S×S个锚框（不重叠），每个锚框预测 B个边缘框，一个锚框 输出B个

# 多尺度目标检测
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from frame_draw import show_bboxes,multibox_prior,multibox_target,multibox_detection
from object_detection import load_data_bananas

def pil2tensor(image):
    image=image.convert("RGB")
    image_np=np.array(image).astype(np.float32)/255.0
    image_tensor=torch.tensor(image_np)
    return image_tensor.permute(2,0,1)

def display(fmap_w,fmap_h,s):
    famp=torch.zeros((1,10,fmap_h,fmap_w))
    anchors=multibox_prior(famp,s,ratios=[1,2,0.5])
    show_bboxes(image_path,bboxes=anchors[0],save_name="SSD_test")

import torchvision
from torch import nn
from torch.nn import functional as F

# 最后的 输出层，num_inputs 是输入通道，num_anchors 是一个像素点的 锚框数，num_classes 是类别数，这个是预测 类别置信度的
def cls_predictor(num_inputs,num_anchors,num_classes):
    # 用一个 卷积做最后输出，输出 长宽和图片 大小一致,意思就是，输出的 每个像素点 通道数就是像素点 所有框预测的 所有类别的值（对每个 像素做预测）
    return nn.Conv2d(num_inputs,num_anchors*num_classes,kernel_size=3,padding=1)

# 预测 我的锚框和真实框的偏移
def bbox_predictor(num_inputs,num_anchors):
    # 同样，输出宽高 和 图片一致,但 前面高宽和通道数 可能会发生变化
    return nn.Conv2d(num_inputs,num_anchors*4,kernel_size=3,padding=1)

# 连接多尺度 预测
def forward(x,block):
    return block(x)

def flatten_pred(pred):
    # 先把预测 从（B，C，H，W） 变成（B，H，W，C）这样展平的时候 是按照像素点的位置，保证像素点1 的锚框预测在前
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1) # torch.flatten(data,start_dim,end_dim) 默认 从0维开始展平，变成 一维张量，和 nn.Flatten() 是有区别的

def concat_preds(preds):
    # 因为 B的维度是一样的，可以 横向拼接,由于 宽高可能变化，把每一个 输出展平，横向拼接  
    return torch.cat([flatten_pred(p) for p in preds],dim=1)

# 定义神经网络 高宽减半块
def down_sample_blk(in_channels,out_channels):
    blk=[]
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels=out_channels
    blk.append(nn.MaxPool2d(2)) # stride 没定义，和kernel_size 对齐 
    return nn.Sequential(*blk)

# 从原始图片 抽特征，到第一次做锚框 
def base_net():
    blk=[]
    num_filters=[3,16,32,64] # (3,16),(16,32),(32,64)
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))
    return nn.Sequential(*blk)

# 完整单发多框检测 五个模块组成
def get_blk(i):
    if i==0:
        blk=base_net()
    elif i==1:
        blk=down_sample_blk(64,128)
    elif i==4:
        blk=nn.AdaptiveMaxPool2d((1,1))  # 最后压到（1，1）
    else:
        blk=down_sample_blk(128,128) # 数据集 不是特别复杂，没必要 继续增大通道了
    
    return blk

def blk_forward(X,blk,size,ratio,cls_predictor,bbox_predictor):
    Y=blk(X)
    anchors=multibox_prior(Y,size,ratio)
    cls_preds=cls_predictor(Y)
    bbox_preds=bbox_predictor(Y)
    return (Y,anchors,cls_preds,bbox_preds)

# 超参数
sizes=[[0.2,0.272],[0.37,0.447],[0.54,0.619],[0.71,0.79],[0.88,0.961]] # 框的大小逐步增大
ratios=[[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5]]
num_anchors=len(sizes[0])+len(ratios[0])-1

class TinySSD(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(TinySSD,self).__init__(**kwargs)
        self.num_classes=num_classes
        idx_to_channesls=[64,128,128,128,128]
        # setattr(obj，name，value) 为 名为name 的变量，赋值 value，obj 为属性 ，self.name 通常为self
        for i in range(5): 
            setattr(self,f'blk_{i}',get_blk(i))
            setattr(self,f'cls_{i}',cls_predictor(idx_to_channesls[i],num_anchors,num_classes+1))
            setattr(self,f'bbox_{i}',bbox_predictor(idx_to_channesls[i],num_anchors))

    def forward(self,X):
        anchors,cls_preds,bbox_preds=[],[],[]
        for i in range(5):
            # getattar(obj,name) obj 属性 self.之类，name 变量名
            X,anchor,cls_pred,bbox_pred=blk_forward(
                X,
                getattr(self,f'blk_{i}'),
                sizes[i],
                ratios[i],
                getattr(self,f'cls_{i}'),
                getattr(self,f'bbox_{i}')
            )
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
        anchors=torch.cat(anchors,dim=1)
        cls_preds=concat_preds(cls_preds)
        cls_preds=cls_preds.reshape(cls_preds.shape[0],-1,self.num_classes+1)
        bbox_preds=concat_preds(bbox_preds)

        return anchors,cls_preds,bbox_preds

cls_loss=nn.CrossEntropyLoss(reduction='none') # 所有样本 不取平均，损失会大一些
bbox_loss=nn.L1Loss(reduction='none') # L1Loss 是平均绝对误差，'none'不计算均值 |y1-y2|,L2Loss 是均方误差,(y1-y2)^2,防止L2Loss 过大  

def calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
    batch_size,num_classes=cls_preds.shape[0],cls_preds.shape[2]
    cls=cls_loss(cls_preds.reshape(-1,num_classes),cls_labels.reshape(-1)).reshape(batch_size,-1).mean(dim=1) # (B,M) 第一张图片 所有框损失，第二张图片 所有框损失;mean(dim=1) 就是把第一张图片 所有框预测损失 平均，（B，）
    bbox=bbox_loss(bbox_preds*bbox_masks,bbox_labels*bbox_masks).mean(dim=1) # 第一个框 损失平均，（B，）

    return cls+bbox

def cls_eval(cls_preds,cls_labels):
    return (cls_preds.argmax(dim=-1).type(cls_labels.dtype)==cls_labels).sum().item()

def bbox_eval(bbox_preds,bbox_labels,bbox_masks):
    return (abs(bbox_preds-bbox_labels)*bbox_masks).sum().item()

def train_ch3(net,train_iter,test_iter,num_epochs,updater,device):
    net=net.to(device)
    train_acc,train_loss=[],[]
    num_anchors=0
    for epoch in range(num_epochs):
        if isinstance(net,nn.Module):
            net.train()
        acc_num,l_num,num=0,0,0
        for x,y in train_iter:
            updater.zero_grad()
            x,y=x.to(device),y.to(device)
            anchors,cls_preds,bbox_preds=net(x)
            num_anchors=anchors.shape[1]
            bbox_masks,bbox_labels,cls_labels=multibox_target(anchors,y)
            l=calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks)
            l.mean().backward()
            updater.step()
            acc_num+=cls_eval(cls_preds,cls_labels)
            l_num+=bbox_eval(bbox_preds,bbox_labels,bbox_masks)
            num+=len(x)*num_anchors
        train_acc.append(acc_num/num)
        train_loss.append(l_num/num)
        print(f'epoch:{epoch+1},train_acc:{train_acc[-1]},train_loss:{train_loss[-1]}')
    
    return train_acc,train_loss

def predict(x,net,device):
    device = torch.device(device)
    net = net.to(device)
    net.eval()
    anchors,cls_preds,bbox_preds=net(x.to(device))
    cls_prob=F.softmax(cls_preds,dim=2).permute(0,2,1)
    output=multibox_detection(cls_prob,bbox_preds,anchors)
    idx=[i for i,row in enumerate(output[0]) if row[0]!=-1]
    return output[0,idx]




def draw_acc_loss(train_acc,train_loss):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_acc,label="train_acc",color="green",linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Acc Curve")
    plt.legend(loc="upper right")

    plt.subplot(1,2,2)
    plt.plot(train_loss,label="train_loss",color="blue",linestyle='-',linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend(loc="upper right")

    plt.savefig("/data/chenyan/pytorch_learn/data/images/SSD_predict.png",dpi=300)

if __name__=="__main__":
    # print(forward(torch.zeros((2,3,20,20)),base_net()).shape)


    # Y1=forward(torch.zeros((2,8,20,20)),cls_predictor(8,5,10))
    # Y2=forward(torch.zeros((2,16,10,10)),cls_predictor(16,3,10))
    # print(Y1.shape,Y2.shape)
    # print(concat_preds([Y1,Y2]).shape)



    # image_path="/data/chenyan/pytorch_learn/data/images/catdog.jpg"
    # image=Image.open(image_path)
    # w,h=image.size
    # print(w,h)

    # display(2,2,s=[0.4])

    # net=TinySSD(num_classes=1)
    # X=torch.zeros((32,3,256,256))
    # anchors,cls_preds,bbox_preds=net(X)
    # print(anchors.shape,cls_preds.shape,bbox_preds.shape)

    batch_size,num_epochs=32,10
    train_iter,test_iter=load_data_bananas(batch_size)

    device,net='cuda:0',TinySSD(num_classes=1)
    updater=torch.optim.SGD(net.parameters(),lr=0.2,weight_decay=5e-4)

    cls_loss=nn.CrossEntropyLoss(reduction='none') # 所有样本 不取平均，损失会大一些
    bbox_loss=nn.L1Loss(reduction='none') # L1Loss 是平均绝对误差，'none'不计算均值 |y1-y2|,L2Loss 是均方误差,(y1-y2)^2,防止L2Loss 过大 
    train_acc,train_loss=train_ch3(net,train_iter,test_iter,num_epochs,updater,device)
    draw_acc_loss(train_acc,train_loss)

    test_image_path='/data/chenyan/pytorch_learn/data/banana-detection/bananas_val/images/92.png'
    test_image=Image.open(test_image_path)
    test_image=pil2tensor(test_image).unsqueeze(0)
    output=predict(test_image,net,device)
    idx=[i for i,row in enumerate(output) if row[1]>=0.9]
    output=output[idx,:]
    show_bboxes(test_image_path,output[:,2:],save_name='SSD_pred_one',labels=output[:,0])





    

    

    

    


