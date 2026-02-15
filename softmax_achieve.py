import torch
import torchvision
from aioitertools.types import Accumulator
from torch.utils import data
from torchvision import transforms

#1.准备数据集
def load_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4))

trans=[transforms.ToTensor()]
trans=transforms.Compose(trans)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)

batch_size=256
train_iter,test_iter=load_fashion_mnist(batch_size)
# for x,y in train_iter:
#     print(x,'\n',y)

#定义权重w，b
num_inputs=784
num_outputs=10

w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True) #拉长，全连接神经网络
b=torch.zeros(num_outputs,requires_grad=True)

#softmax
def softmax(X):
    X_exp=torch.exp(X) #每个元素求e
    # print(X_exp)
    partition=X_exp.sum(1,keepdim=True) #按行求和
    # print(partition)
    return X_exp/partition #对于第i行除以第i个元素

# ans=softmax(torch.tensor([[1,2,3],[4,5,6],[7,8,9]]))
# print(ans.sum(1))

#实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape(len(X),w.shape[0]),w)+b) #b会在列的维度上相加，每行10个输出都加一次b,-1表示该维度自己计算
# x=torch.tensor([[1,2,3,4,5],
#                 [1,2,3,4,5],
#                 [1,2,3,4,5]])
# y=torch.tensor([1,2,3,4,5])
# print(x+y)
# tensor([[ 2,  4,  6,  8, 10],
#         [ 2,  4,  6,  8, 10],
#         [ 2,  4,  6,  8, 10]])

#拿出元素对应输出
y=torch.tensor([0,2]) #标签值
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# print(y_hat[[0,1],y]) #对于第0个元素，取出对应y的下标值，对于第1个元素，取出对应y的下标
# #tensor([0.1000, 0.5000])

#定义交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y]) #对于每一行要去相应预测值，range后其实就是一个列表，里面是每行的标号
# print(cross_entropy(y_hat,y)) #得到每个样本的损失


#计算正确率accuracy
def accuracy(y_hat,y):
    #如果是一个二维矩阵，且列数大于1
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat==y
    # cmp=y_hat.type(y.dtype)==y 将y_hat转为y的数据类型，保证类型一致
    # print(cmp.type(y.dtype)) #输出是bool值,转换后tensor([0, 1])
    # return float(cmp.type(y.dtype).sum()) 转成y的数据类型，bool转int在这里
    return float(cmp.sum()) #做sum后相加0，1，bool值的相加

# print(accuracy(y_hat,y)/len(y))，预测正确概率

#计算器任意模型net的准确率
def evaluate_accuracy(net,data_iter): #计算模型在数据迭代器上的精度
    #如果是torch.nn实现的模型
    if isinstance(net,torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric=[] #正确预测数、预测总数,Accumulator是累加器（自己实现），add不断累加
    for x,y in data_iter:
        metric.append((accuracy(net(x),y),y.numel())) #求元素总数
    acc_num=0
    num=0
    for x,y in metric:
        acc_num+=x
        num+=y
    # print(metric)
    return acc_num/num

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=[]
    for x,y in train_iter:
        y_hat=net(x)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad() #梯度清零
            l.backward()
            updater.step()
            metric.append((accuracy(y_hat,y),y.numel()))
        else: #如果自己实现的话
            l.sum().backward() #求导这块是不用自己求的，在线性回归时也是自动求导，但我们还是把几个样本的损失加了起来再求导
            updater(x.shape[0]) #传入batch_size的值
            metric.append((accuracy(y_hat, y), y.numel()))

    acc_num = 0
    num = 0
    for x, y in metric:
        acc_num += x
        num += y
    # print(metric)
    return acc_num / num


def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_acc=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        print(f'epoch:{epoch+1}  train_acc:{train_acc:f}  test_acc:{test_acc:f}')

def SGD(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            # print(param.grad)
            param-=lr*(param.grad/batch_size) #反向求导其实就是
            param.grad.zero_() #梯度清零

lr=0.1
def updater(batch_size):
    return SGD([w,b],lr,batch_size)

def predict(net,test_iter):
    for x,y in test_iter:
        y_hat=net(x)
        y_hat=y_hat.argmax(axis=1)
        print(f'预测结果:{y_hat} 真实结果:{y}')
        break


if __name__=='__main__':
    num_epochs=10
    train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
    predict(net,mnist_test)



