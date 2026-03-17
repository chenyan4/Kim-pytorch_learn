# 描框：先提出 一系列框，预测每个框是否含有关注物体，如果有，预测这个框到真实框 的偏移；一张图片 可能生成上万 描框
# IOU-交并比：比较两个框之间的 相似度
# 对一张图片 描若干描框，训练时候，每一个描框是一个 训练样本；对于每一个 描框，要么标注成背景（什么都没有），要么关联 一个真实边缘框（和某个物体相关联）
# 非极大抑制（NMS）输出：在推理时，每个描框预测一个边缘框，选中非背景类的最大预测值，去掉所有其他和它的IOU值大于 δ的预测，重复过程，直到所有预测，要么被选中，要么被去掉

import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

torch.set_printoptions(2) # 表示打印tensor时 显示方式，2表示小数保留 2位

# 假定原图做了归一化，就是一个正方形（面积是1），描框的 宽度和高度 分别是 s*r^(0.5)和 s/r^(0.5)，size是 按边长缩放比例 （归一化 下）,ratio是描框的 宽高比（归一化 下）
# 会给 一个系列的 s和r，但不会 一一组合，取 s第一个 遍历所有r；取 r第一个 遍历所有s（排除 第一个（s1，r1））
# data 就是图片（以 像素点为中心，生成 不同形状的框），

def multibox_prior(data,sizes,ratios):
    in_height,in_width=data.shape[-2:] # 一般来说最后两维 是 H、W
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1 # 每个像素点 框的个数
    size_tensor=torch.tensor(sizes,device=device)
    ratio_tensor=torch.tensor(ratios,device=device)

    offset_h,offset_w=0.5,0.5 
    step_h=1.0/in_height # 坐标 已经归一化处理了，每个像素格子（像素是有宽高的，也是有面积的，做几何框时，看成格子会方便），也就是在一个小格子中填充颜色 间隔距离
    step_w = 1.0 / in_width

    # +0.5 表示每个格子在 高度方向上的中心坐标、在 宽度上的中心坐标
    center_h = (torch.arange(in_height, device=device) + offset_h)*step_h  # arange 不是 arrange
    center_w = (torch.arange(in_width, device=device) + offset_w)*step_w  # width 不是 weight

    shift_y,shift_x=torch.meshgrid(center_h,center_w) # 会得到两个（H，W），前面提供 H，后面提供 W
    shift_y,shift_x=shift_y.reshape(-1),shift_x.reshape(-1) # 展平，方便后面zip，由左到右，依次往下

    w=torch.cat((size_tensor*torch.sqrt(ratio_tensor[0]),size_tensor[0]*torch.sqrt(ratio_tensor[1:])))*in_height/in_width # wa/W=w^,wa(真实宽度)=W*w^,归一化坐标 保留 高度，动态调整宽度（解决原图 不是正方形情况，保证缩放后宽高比 一致）
    h=torch.cat((size_tensor/torch.sqrt(ratio_tensor[0]),size_tensor[0]/torch.sqrt(ratio_tensor[1:]))) # 还是归一化的 长度

    out_grid=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1).repeat_interleave(boxes_per_pixel,dim=0) # repeat_interleave(n,dim) 按照维度，进行 单个元素的复制

    anchor_manipulations=torch.stack([-w,-h,w,h],dim=1).repeat(in_height*in_width,1)/2 # torch.repeat(b,n,m),是整块复制，指现在列方向上 整块复制 m次，再在 行方向上 整块复制 m次，再在 通道方向上 整块复制 b次
    out_grid=out_grid+anchor_manipulations

    return out_grid.unsqueeze(0) # 多一个维度表示 这是第一张图片的 锚框

def box_to_rect(bbox,color):
    return patches.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],edgecolor=color,fill=False,linewidth=2)

def show_bboxes(image_path,bboxes,save_name,num_rows=2,num_cols=5,labels=None,colors=None):
    image=Image.open(image_path)
    w,h=image.size
    if colors is None:
        colors=['b','g','r','m','c']
    fig=plt.imshow(image)
    for i,bbox in enumerate(bboxes):
        bbox[0],bbox[2]=bbox[0]*w,bbox[2]*w
        bbox[1],bbox[3]=bbox[1]*h,bbox[3]*h
        color=colors[i%len(colors)]
        rect=box_to_rect(bbox.cpu().numpy(),color)
        fig.axes.add_patch(rect)
        # if labels and len(labels) > i:
        text_color='w'
        fig.axes.text(rect.xy[0],rect.xy[1],labels[i],color=text_color,fontsize=10,ha='center',va='center')

    plt.savefig(f'/data/chenyan/pytorch_learn/data/images/{save_name}.png',dpi=300)
    # plt.close()  # 不close会复用上次 的画布，和图片读取无关
    #     fig.axes.text(
    #     x, y,         # 文本的位置（数据坐标）
    #     s,            # 要显示的字符串
    #     color='w',    # 文本颜色
    #     fontsize=10,  # 字号
    #     ha='center',  # 水平对齐方式：'left'/'center'/'right'
    #     va='center'   # 垂直对齐方式：'top'/'center'/'bottom' 等
    # )

def box_iou(boxes1,boxes2):
    # 假设 boxes1有 M个框，boxes2有 N个框
    def box_area(bbox):
        return (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])
    area1=box_area(boxes1) # （M，）
    area2=box_area(boxes2) # （N，）

    # 利用 广播机制，boxes1 通过[:,None,:2]变成[M,1,2],boxes2 变成[N,2] (Xmin,Ymin)
    # boxes1 在第一个维度上扩展到N，即 将一个框复制 N次；而 boxes2 扩展一个维度，把 boxes2 的所有框 整体复制 M次
    # 这时候 变成[M,N,2] ;boxes1的第一个框，与boxes2 所有框作比；再第二个框，与boxes2 所有框作比

    inner_upperlefts=torch.max(boxes1[:,None,:2],boxes2[:,:2])
    # (Xmax,Ymax)
    inner_lowerrights=torch.min(boxes1[:,None,2:],boxes2[:,2:])

    inners=(inner_lowerrights-inner_upperlefts) #得到 内部长宽，如果没有，则长或宽 至少一个会是负数
    inners=inners*(inners>0)  # 将负数 置0，长宽相乘 面积是0
    inners_area=inners[:,:,0]*inners[:,:,1] # 
    union_area=area1[:,None]+area2-inners_area 
    return inners_area/union_area # 返回 [M,N] 就是boxes1 第一个框和boxes2 所有框的IOU；boxes1 第二个框和boxes2 所有框的IOU，以此类推

    
def assign_anchor_to_bbox(ground_truth,anchors,device,iou_threshold=0.5):
    num_anchors,num_gt_boxes=anchors.shape[0],ground_truth.shape[0]
    
    # 每个 锚框去对应一个真实框，如果 IOU大于阈值，则有映射，小于 阈值表示 这框是背景；所以 一个框要和所有真实框 做IOU，取最大
    anchors_bbox_map=torch.full((num_anchors,),fill_value=-1,device=device,dtype=torch.long) # 创建一个 [num_anchors,]，记录 框到真实框的映射，torch.full 全部填充，背景 用-1（因为 0也会映射）
    # 计算 每个框与所有真实框 IOU值
    jaccard=box_iou(anchors,ground_truth) # 每个框 和所有真实框的交并比

    max_iou,indices=torch.max(jaccard,dim=1) # 在横向上 作比，返回 最大值 和 最大值下标，两个都是 一维的

    # anc_i=torch.nonzero(max_ious>=iou_threshold).reshape(-1) # torch.nonzero()返回 成立值下标，是二维张量，要reshape
    # box_j=indices[max_iou>=iou_threshold] # 取出满足条件的值
    # anchors_bbox_map[anc_i]=box_j # 会遍历 anc_i 下标，去赋予box_j的映射
    anc_i,box_j=[],[]
    for i,iou in enumerate(max_iou):
        if iou>=iou_threshold:
            anc_i.append(i)

    for i in anc_i:
        box_j.append(indices[i])

    for idx in range(len(anc_i)):
        anchors_bbox_map[anc_i[idx]]=box_j[idx]

    # 强制 为真实框分配 锚框，防止有的锚框没有真实框对应，兜底的
    col_discard=torch.full((num_anchors,),fill_value=-1)
    row_discard=torch.full((num_gt_boxes,),fill_value=-1)

    for _ in range(num_gt_boxes):
        max_idx=torch.argmax(jaccard) #相当于 对所有元素 做最大取下标，可以理解先展平后，再取下标

        #计算 在二维中 jaccard中对应位置
        anc_idx=(max_idx/num_gt_boxes).long() # 对应行，转成整数
        box_idx=(max_idx%num_gt_boxes).long() # 对应列，转成整数

        anchors_bbox_map[anc_idx]=box_idx

        jaccard[anc_idx,:]=row_discard
        jaccard[:,box_idx]=col_discard

    return anchors_bbox_map
    

def box_corner_to_center(bbox):
    x1,y1,x2,y2=bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]

    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1

    bbox=torch.stack([cx,cy,w,h],dim=1) # 其实 本质上是一个一维的东西，在（N，）上维度
    return bbox

def box_center_to_corner(bbox):
    cx,cy,w,h=bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]

    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2

    bbox=torch.stack([x1,y1,x2,y2],axis=1)

    return bbox
    

# 计算锚框的偏移量 计算和转换，assigned_bb 是 每个锚框对应分到的真实锚框 的坐标矩阵，无效值也会计算，但不参与损失
def offset_boxes(anchors,assigned_bb,eps=1e-6):
    c_anc=box_corner_to_center(anchors)
    c_assigned_bb=box_corner_to_center(assigned_bb)

    # 中心点偏移，根据 锚框长宽进行归一化，因为 有些锚框很小，离中心点很近，损失不大；但有些锚框很大，虽然IOU是满足的，但损失很大，会 损失爆炸 （排除框的大小影响）
    offset_xy=10*(c_assigned_bb[:,:2]-c_anc[:,:2])/c_anc[:,2:]
    offset_wh=5*torch.log((eps+c_assigned_bb[:,2:])/c_anc[:,2:])
    offset=torch.cat([offset_xy,offset_wh],axis=1)

    return offset

# 使用真实边界框 标记 锚框,anchors 是（1，M，4），labels 则是（B，N，5），B是图片 批量大小，M是一张图片的 最大真实框数，没有的用[0,0,0,0,0]替代，5就是[class_id,xmin,ymin,xmax,ymax]
def multibox_target(anchors,labels):
    batch_size,anchors=labels.shape[0],anchors.squeeze(0) # 去掉第一个维度
    num_anchors,device=anchors.shape[0],anchors.device

    # 批量的偏移掩码（用来保存哪些偏移 计算是有用的），batch_offset 批量偏移,batch_classes_labels 批量 识别类别
    batch_mask,batch_offset,batch_classes_labels=[],[],[]

    # 批量 拿出图片
    for i in range(batch_size):
        label=labels[i,:,:]
        anchors_bbox_map=assign_anchor_to_bbox(label[:,1:],anchors,device)
        anchors_mask=((anchors_bbox_map>=0).float().unsqueeze(-1)).repeat(1,4) # 得到(M,4) 

        # 将 类别和边界框坐标 初始化为0
        class_label=torch.zeros((num_anchors,),device=device,dtype=torch.long)
        assigned_bb=torch.zeros((num_anchors,4),device=device,dtype=torch.float32)

        anchors_idx=torch.nonzero(anchors_bbox_map>=0).reshape(-1) #torch_nonzeros,返回的是成立 下标，是一个二维张量
        boxes_idx=anchors_bbox_map[anchors_idx]
        class_label[anchors_idx]=label[boxes_idx,0].long()+1
        assigned_bb[anchors_idx]=label[boxes_idx,1:]
        anchors_offset=offset_boxes(anchors,assigned_bb)*anchors_mask
        batch_mask.append(anchors_mask.reshape(-1))
        batch_offset.append(anchors_offset.reshape(-1))
        batch_classes_labels.append(class_label)
    
    # 返回的是 （B，4M） 所有框的 值都reshape到一起，每个框占 4个位置
    bboxes_mask=torch.stack(batch_mask)
    bboxes_offset=torch.stack(batch_offset)
    bboxes_label=torch.stack(batch_classes_labels)

    return (bboxes_mask,bboxes_offset,bboxes_label)

# 根据带有 预测偏移量的锚框 来预测边界框
def offset_inverse(anchors,offset_preds):
    anc=box_corner_to_center(anchors)
    pred_bbox_xy=(offset_preds[:,:2]*anc[:,2:]/10)+anc[:,:2]
    pred_bbox_wh=torch.exp(offset_preds[:,2:])*anc[:,2:]
    pred_bbox=torch.cat([pred_bbox_xy,pred_bbox_wh],axis=1)
    predicted_bbox=box_center_to_corner(pred_bbox)

    return predicted_bbox

# 非极大抑制（NMS） 删除框;boxes (M,4) 所有预测框，scores 预测框 分数，框内是否有 物体 (M,)
def nms(boxes,scores,iou_threshold):
    B=torch.argsort(scores,dim=-1,descending=True) # 对元素进行排序，返回下标的 排序，默认 descending=False 升序，True 为 降序；二维 以上需要指定 维度，返回值 维度相同，但下标 排列
    keep=[]
    while B.numel()>0:
        i=B[0].item() # 取出 置信度最大的 下标
        keep.append(i)
        # 如果 只剩一个框，框 保留，退出循环，不用 计算IOU了
        if B.numel()==1:
            break
        # 取出 最大置信度框，重新变成（1，4）,另外 取出剩下的框 (len(B)-1,4),reshape 确保维度，计算 IOU值，返回是一个（1，len(B)-1）
        boxes_iou=box_iou(boxes[i,:].reshape(-1,4),boxes[B[1:],:].reshape(-1,4)).reshape(-1)
        # 选择 保留下来的框在 B中的下标,由于box_iou是从 0开始，但是在 B中是从 1开始，所有要 +1
        box_idx=torch.nonzero(boxes_iou<=iou_threshold).reshape(-1) # 返回空张量
        if box_idx.numel()==0:
            break

        B=B[box_idx+1]

        # 返回 选出来 框的下标
    return torch.tensor(keep,device=boxes.device)

    # 二维张量：2行3列
    # tensor_2d = torch.tensor([[3, 1, 2], 
    #                           [6, 4, 5]])

    # 1. 按行排序（dim=1，每行内部排序）
    # idx_row = torch.argsort(tensor_2d, dim=1)
    # print("按行升序索引：\n", idx_row)
    # 输出：
    # tensor([[1, 2, 0],  # 第一行[3,1,2]升序后：1(索引1)、2(索引2)、3(索引0)
    #         [1, 2, 0]]) # 第二行[6,4,5]升序后：4(索引1)、5(索引2)、6(索引0)

    # 2. 按列排序（dim=0，每列内部排序）
    # idx_col = torch.argsort(tensor_2d, dim=0)
    # print("按列升序索引：\n", idx_col)
    # 输出：
    # tensor([[0, 0, 0],  # 第一列[3,6]升序后：3(索引0)、6(索引1)
    #         [1, 1, 1]]) # 第二列[1,4]升序后：1(索引0)、4(索引1)；第三列[2,5]同理

# 使用 非极大抑制 应用于预测边界框,cls_probs (B,C+1,M) B 就是一个批量，C+1 就是类别数（包含背景），M就是 每个锚框 的预测值；offset_preds（B，4M） 每个锚框 预测偏移量 ;num_threahold 是非极大抑制阈值，pos_threshold 排除部分 类别识别过低
def multibox_detection(cls_probs,offset_preds,anchors,nms_threshold=0.5,pos_threshold=0.009999):
    device,batch_size=cls_probs.device,cls_probs.shape[0]
    anchors=anchors.squeeze(0) # 去除第一个 维度
    num_classes,num_anchors=cls_probs.shape[1],cls_probs.shape[2]

    out=[]
    # 取出 第一张图片
    for i in range(batch_size):
        cls_prob=cls_probs[0]
        offset_pred=offset_preds[i].reshape(-1,4) # (M,4)
        conf,class_id=torch.max(cls_prob[1:,:],dim=0)
        predict_bb=offset_inverse(anchors,offset_pred) #(M,4)

        # 找到所有需要保留框的 索引
        keep=nms(predict_bb,conf,nms_threshold)
        non_keep=[]
        keep_list=keep.tolist()
        # 找到所有 不需要的框的索引 non-keep
        for idx in range(len(conf)):
            if idx not in keep_list:
                non_keep.append(idx)
        non_keep=torch.tensor(non_keep,device=keep.device)

        # 将不需要的框，类别置0
        class_id[non_keep]=-1

        # 将 需要框 和 不需要框 下标合并
        anchors_all_ids=torch.cat([keep,non_keep])

        # 这是 要过滤 置信度过低的框，比如有些 框置信度仅为0.0几（但 nms 由于IOU阈值计算 还是保留了)
        below_conf_idx=torch.nonzero(conf<pos_threshold).reshape(-1)
        class_id[below_conf_idx]=-1
        # 更新 置信度过低的框的 置信度显示（识别种类 置信度低，意味着 背景置信度高
        conf[below_conf_idx]=1-conf[below_conf_idx] # non_keep也应该修正，但已经是 -1了，无所谓

        #调整 位置,按 anchors_all_ids 排好序
        class_id=class_id[anchors_all_ids]
        predict_bb=predict_bb[anchors_all_ids]
        conf=conf[anchors_all_ids]

        predict_info=torch.cat([class_id.unsqueeze(-1),conf.unsqueeze(-1),predict_bb],axis=1)
        out.append(predict_info)
    
    return torch.stack(out) # (B,M,6)




        




if __name__=="__main__":
    img=Image.open("/data/chenyan/pytorch_learn/data/images/catdog.jpg")
    w,h=img.size
    print(w,h)

    X=torch.randn((1,3,h,w))
    Y=multibox_prior(X,[0.75,0.5,0.25],[1.5,0.5])
    print((Y[0,3,2]-Y[0,3,0])/(Y[0,3,3]-Y[0,3,1]))
    bboxes=Y[0,1000000:1000010,:]
    image_path="/data/chenyan/pytorch_learn/data/images/catdog.jpg"

    # show_bboxes(image_path,bboxes,save_name='bbox_temp')

    ground_truth=torch.tensor([[0,0.1,0.08,0.52,0.92],[1,0.55,0.2,0.9,0.88]])
    anchors=torch.tensor([[0,0.1,0.2,0.3],[0.15,0.2,0.4,0.4],[0.63,0.05,0.88,0.98],[0.66,0.45,0.84,0.92],[0.57,0.3,0.92,0.9]])

    # show_bboxes(image_path,ground_truth[:,1:],labels=['dog','cat'],save_name="true")
    # show_bboxes(image_path,anchors,labels=['1','2','3','4','5'],save_name="train")

    labels=multibox_target(anchors.unsqueeze(0),ground_truth.unsqueeze(0))
    print(labels[0],labels[1],labels[2])

    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds=torch.zeros((4,4))
    cls_probs=torch.tensor([[0,0,0,0],[0.9,0.8,0.7,0.1],[0.1,0.2,0.3,0.9]])

    out=multibox_detection(cls_probs.unsqueeze(0),offset_preds.unsqueeze(0),anchors.unsqueeze(0))
    print(out)

    for i in out[0]:
        if i[0]==-1:
            continue
        show_bboxes(image_path,i.unsqueeze(0),labels=[f'{i[0]}'],save_name="dog_cat_pred")

# center_h = torch.tensor([0.5, 1.5])   # 2 行
# center_w = torch.tensor([0.5, 1.5, 2.5])  # 3 列
# shift_y, shift_x = torch.meshgrid(center_h, center_w)
# shift_y: [[0.5, 0.5, 0.5],
#           [1.5, 1.5, 1.5]]   shape (2, 3)
# shift_x: [[0.5, 1.5, 2.5],
#           [0.5, 1.5, 2.5]]   shape (2, 3)
# 格点 (0,0) 中心 (0.5, 0.5)，(0,1) 中心 (0.5, 1.5)，(1,2) 中心 (1.5, 2.5) ...