# 文本预处理
import collections
import re
import torch
import random

base_path="/data/chenyan/pytorch_learn/data/timemachine.txt"
def read_time_machine():
    with open(base_path,'r') as f:
        lines=f.readlines() # readlines() 按行 一行一行读文件，list[str]
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines] # strip()删除字符串首尾空白字符（空格、换行），lower()变成小写

lines=read_time_machine()
# print(lines[10])

# 将每个文本序列 拆分成一个标记列表
def tokenize(lines,token='word'):
    # 是一个句子,拆分成单词
    if token=='word':
        return [line.split() for line in lines]
    # 是一个字符串
    elif token=='char':
        return [list(line) for line in lines] # 字符串是可索引，list就拆分成一个一个 字母
    else:
        print('错误：未知令牌类型:'+token)

tokens=tokenize(lines)
# for i in range(11):
#     print(tokens[i])

class Vocab:
    # min_freq 表示一个 token出现小于min_freq次，就丢掉;reserved_tokens 是自己传进来的 额外保留的token，可以是自己定义的起止符
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens=[]
        if reserved_tokens is None:
            reserved_tokens=[]
        counter=count_corpus(tokens)
        self.token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True) # key 是按照什么排序，reverse=True 表示降序
        # self.token_freqs=counter.items()
        # token_freqs_line=len(self.token_freqs)
        # for i in range(token_freqs_line,1,-1):
        #     for j in range(i-1):
        #         if self.token_freqs[j][1]<self.token_freqs[j+1][1]:
        #             temp=self.token_freqs[j]
        #             self.token_freqs[j]=self.token_freqs[j+1]
        #             self.token_freqs[j+1]=temp

        #uniq_tokens 是一个特殊tokens列表，以<unk>开头；self.unk是这些不用token的id
        self.unk,uniq_tokens=0,['<unk>']+reserved_tokens
        # 把一些出现 频率高于阈值的token 加入uniq_tokens（我的词表）
        uniq_tokens+=[token for token,freq in self.token_freqs if freq>=min_freq and token not in uniq_tokens]

        self.idx_to_token,self.token_to_idx=[],{}
        for token in uniq_tokens:
            self.idx_to_token.append(token) # id:0 第一个就是不知道的token
            self.token_to_idx[token]=len(self.idx_to_token)-1

    

    def __len__(self):
        return len(self.idx_to_token)

    # 给一个tokens 或者是 tokens列表
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            # 如果 get(token,default) 表示如果有就取值，没有用default(即self.unk)
            return self.token_to_idx.get(tokens,self.unk) 
        # 如果是一个tokens列表
        return [self.__getitem__(token) for token in tokens]
        # return [self.token_to_idx.get(token,self.unk) for token in tokens]
        
    def to_tokens(self,indices):
        if not isinstance(indices,(tuple,list)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[idx] for idx in indices]


def count_corpus(tokens):
    tokens_count=[]
    if len(tokens)==0 :
        return collections.Counter()
    if isinstance(tokens[0],list):
        for line in tokens:
            for token in line:
                tokens_count.append(token)
        return collections.Counter(tokens_count)
    else:
        return collections.Counter(tokens)

vocab=Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])

# for i in [0,10]:
#     print('words:',tokens[i])
#     print('indices:',vocab.__getitem__(tokens[i]))

def load_corpus_time_machine(max_tokens=-1):
    lines=read_time_machine()
    tokens=tokenize(lines,'char')
    vocab=Vocab(tokens)
    corpus=[]
    # corpus 相当于把文章拉成 一长条，所有词元的下标
    for line in tokens:
        for token in line:
            corpus.append(vocab.__getitem__(token))
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return corpus,vocab

# 随机抽样生成一个小批量子序列,num_steps是子序列长度
def seq_data_iter_random(corpus,batch_size,num_steps):
    # 随机丢弃 前0-num_steps个数据,打破 固定偏移，让模型能看到不同开头序列;random.randint两端都包含，要有num_steps种，必需num_steps-1;num_steps等价于丢掉一个完整块，这没必要
    corpus=corpus[random.randint(0,num_steps-1):]
    # 一共有多少段序列，向下取整,-1是 为了保留 预测数据Y，比如 12段，num_steps=3,这个时候 如果分成 4段时，在取预测时会越界
    num_subseqs=(len(corpus)-1)//num_steps
    # 取出每一段开头的 起始下标，比如num_steps=3,分成 4段,那么得到[0,3,6,9] 四段的起始下标；12是开区间不取
    initial_indices=list(range(0,num_subseqs*num_steps,num_steps))
    # 要将 提取顺序打乱,比如[3,0,9,6]，这是候取的段落就会被 打乱，模型能学到更强的信息
    random.shuffle(initial_indices)

    # 取出 子序列
    def data(pos):
        return corpus[pos:pos+num_steps]

    # 可以得到多少 batch，batch_size表示 打包多少个 子序列为一组
    num_batches=num_subseqs//batch_size
    for i in range(0,num_batches*batch_size,batch_size):
        initial_indices_per_batch=initial_indices[i:i+batch_size]
        X=[data(j) for j in initial_indices_per_batch]
        Y=[data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y) # yield每次循环返回一个值；在函数外 可以 for X,Y in seq_data_iter_random()一批一批拿取数据


# 使用 顺序分区生成一个小批量子序列
# def seq_data_iter_sequential(corpus,batch_size,num_steps):
#     # 前向偏移
#     offset=random.randint(0,num_steps-1)
#     corpus=corpus[offset:]
#     num_subseqs=(len(corpus)-1)//num_steps
#     initial_indices=list(range(0,num_subseqs*num_steps,num_steps))

#     def data(pos):
#         return corpus[pos:pos+num_steps]

#     num_batches=num_subseqs//batch_size
#     for i in range(0,num_batches*batch_size,batch_size):
#         initial_indices_per_batch=initial_indices[i:i+batch_size]
#         X=[data(j) for j in initial_indices_per_batch]
#         Y=[data(j+1) for j in initial_indices_per_batch]
#         yield torch.tensor(X),torch.tensor(Y)

# 也就是 小批量里的两个数据，在下一个小批量里的两个数据，在位置方向上是连续的
def seq_data_iter_sequential(corpus,batch_size,num_steps):
    offset=random.randint(0,num_steps-1)
    num_tokens=((len(corpus)-offset-1)//batch_size)*batch_size
    Xs=torch.tensor(corpus[offset:offset+num_tokens])
    Ys=torch.tensor(corpus[offset+1:offset+num_tokens+1])
    Xs,Ys=Xs.reshape(batch_size,-1),Ys.reshape(batch_size,-1)
    num_batches=Xs.shape[1]//num_steps
    for i in range(0,num_batches*num_steps,num_steps):
        X=Xs[:,i:i+num_steps]
        Y=Ys[:,i:i+num_steps]
        yield X,Y

class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.data_iter_fn=seq_data_iter_random
        else:
            self.data_iter_fn=seq_data_iter_sequential
        self.corpus.self.vocab=load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps=batch_size,num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)
    
def load_data_time_machine(batch_size,num_steps,use_random_iter=False,max_tokens=10000):
    data_iter=SeqDataLoader(batch_size,num_steps,use_random_iter,max_tokens)

    return data_iter,data_iter.vocab

my_seq=list(range(35))
for X,Y in seq_data_iter_sequential(my_seq,batch_size=2,num_steps=5):
    print('X:',X)
    print('Y:',Y)


# corpus,vocab=load_corpus_time_machine()
# print(len(corpus),vocab.__len__())