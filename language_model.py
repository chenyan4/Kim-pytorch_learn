# 语言模型
import random
import torch
import re
from text_predo import Vocab
import matplotlib.pyplot as plt

base_url="/data/chenyan/pytorch_learn/data/timemachine.txt"

def read_time_machine():
    with open(base_url,'r') as f:
        lines=f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line) for line in lines]

def tokenize(lines,token="word"):
    if token=="word":
        return [line.split() for line in lines]
    elif token=="char":
        return [list(line) for line in lines]
    else:
        print(f"错误:未知令牌类型{token}")

tokens=tokenize(read_time_machine())
corpus=[]
for line in tokens:
    for token in line:
        corpus.append(token)

vocab=Vocab(corpus)
print(vocab.token_freqs[:10])

freqs=[freq for token,freq in vocab.token_freqs]
plt.figure(figsize=(12,6))
plt.plot(freqs,label='freq',color='b',linestyle='-',linewidth=2)
plt.xlabel('token:x')
plt.ylabel('frequency:n(x)')
plt.xscale('log')
plt.yscale('log')
plt.savefig("/data/chenyan/pytorch_learn/data/images/text_freq.png",dpi=300)