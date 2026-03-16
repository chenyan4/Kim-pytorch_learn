# 描框：先提出 一系列框，预测每个框是否含有关注物体，如果有，预测这个框到真实框 的偏移；一张图片 可能生成上万 描框
# IOU-交并比：比较两个框之间的 相似度
# 对一张图片 描若干描框，训练时候，每一个描框是一个 训练样本；对于每一个 描框，要么标注成背景（什么都没有），要么关联 一个真实边缘框（和某个物体相关联）
# 非极大抑制（NMS）输出：在推理时，每个描框预测一个边缘框，选中非背景类的最大预测值，去掉所有其他和它的IOU值大于 δ的预测，重复过程，直到所有预测，要么被选中，要么被去掉

import torch

torch.set_printoptions(2) # 表示打印tensor时 显示方式，2表示小数保留 2位

# 描框的 宽度和高度 分别是 ws*r^(0.5)和 hs/r^(0.5)，w和s 图片宽高，scale是 占图片多少的比例（例如 80%）,ratio是描框的 高宽比
# 会给 一个系列的 s和r，但不会 一一组合，取 s第一个 遍历所有r；取 r第一个 遍历所有s（排除 第一个（s1，r1））
# data 就是图片（以 像素点为中心，生成 不同形状的框），

def multibox_prior(data,sizes,ratios):
    in_height,in_width=data.shape[-2:] # 一般来说最后两维 是 H、W
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1 # 每个像素点 框的个数
    size_tensor=torch.tensor(sizes,device=device)
    ratio_tensor=torch.tensor(ratios,device=device)

    offset_h,offset_w=0.5,0.5 # 中心店 归一化坐标
    steps_h=1.0/in_height # 坐标 已经归一化处理了，每个像素点 间隔距离
    step_w = 1.0 / in_width

    center_h = (torch.arange(in_height, device=device) + offset_h)  # arange 不是 arrange
    center_w = (torch.arange(in_width, device=device) + offset_w)  # width 不是 weight