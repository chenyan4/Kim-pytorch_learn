# 机器翻译与数据集
import os
import torch
import re
from text_predo import Vocab
from torch.utils.data import DataLoader,Dataset


def read_data_nmt():
    data_dir='/workspace/Kim-pytorch_learn/data/fra-eng/fra.txt'
    with open(data_dir,"r") as f:
        lines=f.readlines()
    # 统一一些“看起来像空格”的特殊空白字符，避免分词时切不开
    clear_lines=[
        re.sub(r'CC-BY 2.0 \(France\)(.*)','',line)
        .strip()
        .lower()
        .replace('\u202f',' ')  # NARROW NO-BREAK SPACE
        .replace('\xa0',' ')    # NO-BREAK SPACE
        for line in lines
    ]
    # clear_lines=[re.sub('\\t',' ',line).strip() for line in clear_lines]
    # clear_lines=[re.findall('[A-Za-z]+|[.,?!]+|\\t',line) for line in clear_lines]
    
    return clear_lines

raw_text=read_data_nmt()
# print(raw_text[:30])

def tokenize_nmt(text,num_examples=None):
    source,target=[],[]
    idx=0
    for line in text:
        if num_examples and idx>num_examples:
            break
        parts=line.split('\t')
        sou=parts[0]
        tar=parts[1]
        for i in range(len(sou)):
            if sou[i] in ['.',',','!','?']:
                sou=sou[:i]+' '+sou[i:]

        for i in range(len(tar)):
            if tar[i] in ['.',',','!','?']:
                tar=tar[:i]+' '+tar[i:]
 
        sou=sou.split()
        tar=tar.split() # split() 会按任意空白分开,会区分连续空白；list()才是细分每一个字符
        if source==[] or sou!=source[-1]:
            source.append(sou)
            target.append(tar)
            idx+=1
    return source,target

# 将句子放到指定长度
def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))


source,target=tokenize_nmt(raw_text)
# print(source[:10])
src_vocab=Vocab(source,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
print(src_vocab.__len__())

# print(truncate_pad(src_vocab.__getitem__(source[0]),10,src_vocab.__getitem__('<pad>')))

# 将机器翻译的文本序列转换成小批量
def build_array_nmt(lines,vocab,num_steps):
    lines=[vocab.__getitem__(l) for l in lines]
    lines=[l+[vocab.__getitem__('<eos>')] for l in lines] # <eos>是截止符
    array=torch.tensor([truncate_pad(l,num_steps,vocab.__getitem__('<pad>')) for l in lines])
    valid_len=(array!=vocab.__getitem__('<pad>')).type(torch.float32).sum(dim=1) # 每个句子的实际 有效长度
    return array,valid_len

class NMT(Dataset):
    def __init__(self,data_array):
        self.src_array,self.src_valid_len,self.tgt_array,self.tgt_valid_len=data_array
    
    def __len__(self):
        return len(self.src_array)

    def __getitem__(self,idx):
        return (self.src_array[idx],self.src_valid_len[idx],self.tgt_array[idx],self.tgt_valid_len[idx])


def load_data_nmt(batch_size,num_steps,num_examples=600):
    text=read_data_nmt()
    source,target=tokenize_nmt(text,num_examples)
    src_vocab=Vocab(source,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
    tgt_vocab=Vocab(target,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])

    src_array,src_valid_len=build_array_nmt(source,src_vocab,num_steps)
    tgt_array,tgt_valid_len=build_array_nmt(target,tgt_vocab,num_steps)

    data_array=[src_array,src_valid_len,tgt_array,tgt_valid_len]
    data_set=NMT(data_array)

    pin=torch.cuda.is_available()
    nw=min(4,(os.cpu_count() or 1))
    dl_kw=dict(batch_size=batch_size,shuffle=True,drop_last=True,pin_memory=pin,num_workers=nw)
    if nw>0:
        dl_kw["persistent_workers"]=True
    data_iter=DataLoader(data_set,**dl_kw)
    return data_iter,src_vocab,tgt_vocab

# train_iter,src_vocab,tgt_vocab=load_data_nmt(batch_size=2,num_steps=8)
# for X,X_len,Y,Y_len in train_iter:
#     print("X:",X)
#     print("X_len:",X_len)
#     print("Y:",Y)
#     print("Y_len",Y_len)
#     break

