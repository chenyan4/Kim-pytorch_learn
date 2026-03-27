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
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

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

# freqs=[freq for token,freq in vocab.token_freqs]
# plt.figure(figsize=(12,6))
# plt.plot(freqs,label='freq',color='b',linestyle='-',linewidth=2)
# plt.xlabel('token:x')
# plt.ylabel('frequency:n(x)')
# plt.xscale('log')
# plt.yscale('log') # 常选值有 linear（默认）、symlog,logit
# plt.savefig("/data/chenyan/pytorch_learn/data/images/text_freq.png",dpi=300)

# 二元 序列
bigram_tokens=[pair for pair in zip(corpus[:-1],corpus[1:])]
bigram_vocab=Vocab(bigram_tokens)
bigram_freqs=[freq for token,freq in bigram_vocab.token_freqs]
# print(bigram_vocab.token_freqs[:10])

# 三元 序列
trigram_tokens=[triple for triple in zip(corpus[:-2],corpus[1:-1],corpus[2:])]
trigram_vocab=Vocab(trigram_tokens)
trigram_freqs=[freq for token,freq in trigram_vocab.token_freqs]
print(trigram_vocab.token_freqs[:10])

freqs=[freq for token,freq in vocab.token_freqs]
plt.figure(figsize=(12,6))
plt.plot(freqs,label='freq',color='b',linestyle='-',linewidth=2)
plt.plot(bigram_freqs,label='bigram_freq',color='g',linestyle='--',linewidth=2)
plt.plot(trigram_freqs,label='trigram_freq',color='r',linestyle='-.',linewidth=2)
plt.xlabel('token:x')
plt.ylabel('frequency:n(x)')
plt.xscale('log')
plt.yscale('log') # 常选值有 linear（默认）、symlog,logit
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig("/data/chenyan/pytorch_learn/data/images/text_freq.png",dpi=300)