#数据预处理
import os

#1.创建人工数据集，并存储在CSV（逗号分隔值）文件
os.makedirs(os.path.join('data'),exist_ok=True)
data_file=os.path.join('data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,pave,127500\n') #每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

#2.加载原始数据
import pandas as pd
data=pd.read_csv(os.path.join('data','house_tiny.csv')) #data是一个4行3列

#3.补充确实数据
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2] #取0-1列，2列
# 只对数值列进行均值填充
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs)

#4.如果不是数值(字符串变数值）
inputs=pd.get_dummies(inputs,dummy_na=True) #将pave和NaN分成两类，一个pave一个NaN
print(inputs)

#5.转成数值后，可以变为张量
import torch
# 将pandas DataFrame和Series转换为numpy数组，再转换为PyTorch张量
# 需要将布尔值转换为浮点数
x = torch.tensor(inputs.values.astype(float), dtype=torch.float32) #将panda Dataframe 转换成numpy数组，将布尔类型转换成浮点型
y = torch.tensor(outputs.values, dtype=torch.float32) #转成Numpy数组后就可以转化为张量了
print("输入张量 x:")
print(x)
print("输出张量 y:")
print(y)