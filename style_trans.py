# 样式迁移
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

content_image=Image.open('/data/chenyan/pytorch_learn/data/images/1.jpg')
style_image=Image.open('/data/chenyan/pytorch_learn/data/images/style_to.jpg')

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

def preprocess(img,img_shape):
    transform=transforms.Compose([transforms.Resize(img_shape),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    return transform(img).unsqueeze(0)

def postprocess(img):
    img = img[0].detach().permute(1, 2, 0)
    std = torch.tensor(rgb_std, dtype=img.dtype, device=img.device)
    mean = torch.tensor(rgb_mean, dtype=img.dtype, device=img.device)
    img = img * std + mean
    img = img.clamp(0, 1) # 把值都 限制在（0，1）之间
    return Image.fromarray((img * 255).cpu().numpy().astype(np.uint8))
    

vgg19=torchvision.models.vgg19(pretrained=True)
style_layers,content_layers=[0,5,10,19,28],[25] # 用来 匹配不同深度卷积层的输出，style可以看到是提取由 浅层到深层 特征信息；内容上，越底层 其实内容还原 越好，越往上允许变形

_vgg_feats = list(vgg19.features.children())
net = nn.Sequential(
    *[_vgg_feats[i] for i in range(max(style_layers + content_layers) + 1)]
)
# for _p in net.parameters():
#     _p.requires_grad = False
# net.eval()

# 一层一层 抽特征
def extract_features(x,content_layers,style_layers):
    contents=[]
    styles=[]
    for i in range(len(net)):
        x=net[i](x)

        if i in content_layers:
            contents.append(x)
        if i in style_layers:
            styles.append(x)
    return contents,styles

# 先 图片抽好特征
def get_contents(image_shape,device):
    content_X=preprocess(content_image,image_shape).to(device)
    content_Y,_=extract_features(content_X,content_layers,style_layers)
    return content_X,content_Y
# 抽好 样式特征
def get_styles(image_shape,device):
    style_X=preprocess(style_image,image_shape).to(device)
    _,styles_Y=extract_features(style_X,content_layers,style_layers)
    return style_X,styles_Y

# 后续不改变模型权重,Y_hat 是生成图片 在第25层的输出；Y是原图片在第 25层输出
def content_loss(Y_hat,Y):
    return torch.square(Y_hat-Y.detach()).mean() # 均方误差

# 实际上就是 输入是(1,c,h,w),先把这个 reshape成（c,h×w),gram就是X*X^T,即(c,h×w)×(h×w,c),对角线其实就是 相同通道所有元素平方 加和，这其实就包括了 本通道上的分布 X^2，和通道间的分布 X1*X2
def calc_gram(x):
    num_channels,h,w=x.shape[1],x.shape[2],x.shape[3]
    x=x.reshape(num_channels,h*w)
    # 由于 hw较大时，可能导致矩阵 元素过大
    return torch.matmul(x,x.T)/(num_channels*h*w)

# 计算损失和 内容一样,gram_Y是提前计算好的 gram矩阵
def calc_styleloss(Y_hat,gram_Y):
    return torch.square(calc_gram(Y_hat)-gram_Y.detach()).mean()

# 有时候某些像素 特别亮或暗，意味着它和相邻像素值的差距很大，这是高频噪点，把这个看成一个损失，要全变分去噪 即0.5*（(Xi+1|j-Xi|j)+(Xi|j+1-Xi|j),0.5 因为两个相加，取0.5得1（类似）
def tv_loss(Y_hat):
    l=0.5*(torch.abs(Y_hat[:,:,1:,:]-Y_hat[:,:,:-1]).mean()+torch.abs(Y_hat[:,:,:,1:]-Y_hat[:,:,:,:-1]).mean())
    return l

# 风格转移的损失是 内容损失、风格损失和 去噪损失的加权和
# tv 过大时总损失里平滑项会压过风格/内容，图会糊、风格上不去；可先试 tv_weight=1 再微调
content_weight, style_weight, tv_weight = 1, 1e3, 1

def compute_loss(X,content_Y_hat,style_Y_hat,contents_Y,styles_Y_gram):
    content_l=[content_loss(Y_hat,Y)*content_weight for Y_hat,Y in zip(content_Y_hat,contents_Y)]
    style_l=[calc_styleloss(Y_hat,Y_gram)*style_weight for Y_hat,Y_gram in zip(style_Y_hat,styles_Y_gram)]
    tv_l=tv_loss(X)*tv_weight
    # 对所有 5个风格损失，1个内容损失，1个 全分损失
    l=sum(content_l)+sum(style_l)+tv_l
    return content_l,style_l,tv_l,l

# 参数其实就是我输出的一整张图片，其实是去 训练一张图片参数
class SynthesizedImage(nn.Module):
    def __init__(self,img_shape):
        super(SynthesizedImage,self).__init__()
        self.weight=nn.Parameter(torch.rand(*img_shape)) # 这个就是我们的图片

    def forward(self):
        return self.weight

# 可以 初始化成 你的内容图片
def get_inits(X,device,lr,styles_Y):
    get_img=SynthesizedImage(X.shape).to(device) # 跟网络类似
    get_img.weight.data.copy_(X.data) # 为网络权重 复制参数
    updater=torch.optim.Adam(get_img.parameters(),lr=lr)
    style_Y_gram=[calc_gram(Y) for Y in styles_Y]
    # 返回模块本身；需要当前图像张量时在训练循环里写 X = get_img()（会调 forward → 返回 self.weight）
    return get_img, style_Y_gram, updater # 要返回的是 初始化的权重作为输入，优化器 会优化参数

def train(X,contents_Y,styles_Y,lr,num_epochs,device,lr_decay_epoch):
    get_img,style_Y_gram,updater=get_inits(X,device,lr,styles_Y)
    scheduler=torch.optim.lr_scheduler.StepLR(updater,lr_decay_epoch)

    X=get_img()
    for epoch in range(num_epochs):
        updater.zero_grad()
        content_Y_hat,styles_Y_hat=extract_features(X,content_layers,style_layers)
        content_l,style_l,tv_l,l=compute_loss(X,content_Y_hat,styles_Y_hat,contents_Y,style_Y_gram)
        l.backward()
        updater.step()
        scheduler.step()
        X=get_img()

        if (epoch+1)%50==0:
            print(f'content_l:{sum(content_l).item()},style_l:{sum(style_l).item()},tv_l:{tv_l.item()}')
            image=postprocess(X)
            image.save(f"/data/chenyan/pytorch_learn/data/images/{epoch+1}.png")
        
    return X

device,image_shape='cuda:0',(450,300)
net=net.to(device)

content_X,content_Y=get_contents(image_shape,device)
_,style_Y=get_styles(image_shape,device)
output=train(content_X,content_Y,style_Y,0.3,500,device,50)



        




