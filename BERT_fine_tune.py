# BERT 微调
import os
import re
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from text_predo import Vocab


base_url='/workspace/Kim-pytorch_learn/data/snli_1.0'

def read_snli(data_dir,is_train):
    def extract_text(s):
        s=re.sub('\(','',s)
        s=re.sub('\)','',s)
        s=re.sub('\s{2,}',' ',s) # 匹配 两个以上空白符，变成一个空白符
        return s.strip()
    label_set={'entailment':0,'contradiction':1,'neutral':2} # 2 不知道，1 不对，0 对
    file_name=os.path.join(data_dir,'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name,'r') as f:
        rows=[row.split('\t') for row in f.readlines()[1:]] # 按制表符 进行分隔，因为 txt文件每个属性分隔会有一个制表符
    premises=[extract_text(row[1]) for row in rows if row[0] in label_set] # 每个样本的前提
    hypotheses=[extract_text(row[2]) for row in rows if row[0] in label_set] # 每个样本的假设
    labels=[label_set[row[0]] for row in rows if row[0] in label_set] # 每个样本的label
    return premises,hypotheses,labels

def tokenizer(lines,token="word"):
    if token=="word":
        return [line.split() for line in lines]
    if token=="char":
        return [list(line) for line in lines]
    else:
        print(f'未知令牌类型:{token}')

def truncate_pad(line,num_steps,value):
    if len(line)>num_steps:
        return line[:num_steps]
    else:
        return line+[value]*(num_steps-len(line))    

class SNLIDataset(Dataset):
    def __init__(self,dataset,num_steps,vocab=None):
        self.num_steps=num_steps
        all_premise_tokens=tokenizer(dataset[0])
        all_hypothesis_tokens=tokenizer(dataset[1])
        if vocab is None: # vocab 要用BERT的词表
            self.vocab=Vocab(tokens=all_premise_tokens+all_hypothesis_tokens,min_freq=5,reserved_tokens=['<pad>'])
        else:
            self.vocab=vocab

        self.premises=self._pad(all_premise_tokens)
        self.hypotheses=self._pad(all_hypothesis_tokens)
        self.labels=torch.tensor(dataset[2])
        
    def _pad(self,lines):
        return torch.tensor([truncate_pad(self.vocab.__getitem__(line),self.num_steps,value=self.vocab.__getitem__('<pad>')) for line in lines])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return (self.premises[idx],self.hypotheses[idx]),self.labels[idx]
    
def load_data_snli(batch_size,num_steps=50):
    train_data=read_snli(base_url,is_train=True)
    test_data=read_snli(base_url,is_train=False)
    train_set=SNLIDataset(train_data,num_steps)
    test_set=SNLIDataset(test_data,num_steps,train_set.vocab)
    train_iter=DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
    test_iter=DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=4)

    return train_iter,test_iter,train_set.vocab

 
# train_data=read_snli(base_url,is_train=True)
# test_data=read_snli(base_url,is_train=False)

# for x0,x1,y in zip(train_data[0][:3],train_data[1][:3],train_data[2][:3]):
#     print('permise:',x0)
#     print('hypothesis:',x1)
#     print('label:',y)

# for data in[train_data,test_data]:
#     print([data[2].count(i) for i in range(3)])

train_iter,test_iter,vocab=load_data_snli(128,50)
print(vocab.__len__())
for X,Y in train_iter:
    print(X[0].shape,X[1].shape,Y.shape)