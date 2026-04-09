# 加载 WiKiText数据集

import os 
import random
import torch


base_url="/workspace/Kim-pytorch_learn/data/db1ec-main/wikitext-2"

def _read_wiki(data_dir):
    file_name=os.path.join(data_dir,'wiki.train.tokens')
    with open(file_name,'r') as f:
        lines=f.readlines()
    paragraphs=[line.strip().lower().split(' . ') for line in lines if len(line.split(' . '))>=2] # 过滤掉比较短的文本
    random.shuffle(paragraphs)
    return paragraphs

print(_read_wiki(base_url)[:5])