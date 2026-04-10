# 加载 WiKiText数据集

import os 
import random
import torch


base_url="/workspace/Kim-pytorch_learn/data/db1ec-main/wikitext-2"

def _read_wiki(data_dir):
    file_name=os.path.join(data_dir,'wiki.train.tokens')
    with open(file_name,'r') as f:
        lines=f.readlines()
    paragraphs=[line.strip().lower().split(' . ') for line in lines if len(line.split(' . '))>=2] # 过滤掉比较短的文本,为了做 下一句子预测
    random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence,next_sentence,paragraphs):
    if random.random()<0.5:
        is_next=True
    else:
        next_sentence=random_choice(random.choice(paragraphs)) # paragraphs是二维数组，随机选择句子
        is_next=False
    return sentence,next_sentence,is_next

def get_tokens_and_segments(tokens_a,tokens_b=None):
    tokens=['<cls>']+tokens_a+['sep']
    segments=[0]*(len(tokens_a)+2)
    if tokens_b is not None:
        tokens=tokens+tokens_b+['sep']
        segments=segments+[1]*(len(tokens_b)+1)
    return tokens,segments

def _get_nsp_data_from_paragraph(paragraph,paragraphs,vocab,max_len):
    nsp_data_from_paragraph=[]
    for i in range(len(paragraph)-1):
        tokens_a,tokens_b,is_next=_get_next_sentence(paragraph[i],paragraph[i+1],paragraphs)

        tokens_a=tokens_a.split()
        tokens_b=tokens_b.split()
        # 考虑一个 <cls> 和 两个 <sep>
        if len(tokens_a)+len(tokens_b)+3>max_len:
            continue
        tokens,segments=get_tokens_and_segments(tokens_a,tokens_b)
        nsp_data_from_paragraph.append((tokens,segments,is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens,candidate_pred_positions,num_mlm_preds,vocab):
    # 复制一个 tokens备份，不修改原来的tokens
    mlm_input_tokens=[token for token in tokens]
    pred_positions_and_labels=[]
    # candidate_pred_positions 是可替换的 编码的下标;随机打乱
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        # 如果打到预测上限
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token=None
        if random.random()<0.8:
            masked_token='<mask>'
        else:
            if random.random()<0.5:
                masked_token=tokens[mlm_pred_position]
            else:
                random_idx=random_randint(0,vocab.__len__()-1)
                masked_token=vocab.to_tokens(random_idx)

        mlm_input_tokens[mlm_pred_position]=masked_token
        pred_positions_and_labels.append((mlm_pred_position,token[mlm_pred_position])) # 保存替换的词 和 该地方正确的词
    return mlm_input_tokens,pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens,vocab):
    candidate_pred_positions=[]
    for i,token in enumerate(tokens):
        if token in ['<cls>','<seq>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds=max(1,round(len(tokens)*0.15)) # round 是四舍五入
    mlm_input_tokens,pred_positions_and_labels=_replace_mlm_tokens(tokens,candidate_pred_positions,num_mlm_preds,vocab)

    pred_positions_and_labels=sorted(pred_positions_and_labels,key=lambda x:x[0],reverse=True) # 还是按照 下标从低到高排列
    pred_positions=[v[0] for v in pred_positions_and_labels]
    mlm_pred_labels=[v[1] for v in pred_positions_and_labels]
    return vocab.__getitem__(mlm_input_tokens),pred_positions,vocab.__getitem__(mlm_pred_labels)
    
def _pad_bert_inputs(examples,max_len,vocab):
    max_num_mlm_preds=round(max_len*0.15)
    all_token_ids,all_segments,valid_lens=[],[],[]
    all_pred_positions,all_mlm_weights,all_mlm_labels=[],[],[]
    nsp_labels=[]
    for (token_ids,pred_positions,mlm_pred_label_ids,segments,is_next) in examples:
        all_token_ids.append(torch.tensor(vocab.__getitem__(token_ids+['<pad>']*(max_len-len(token_ids))),dtype=torch.long()))
        all_segments.append(torch.tensor(segments+[0]*(max_len-len(segments)),dtype=torch.long))
        

print(_read_wiki(base_url)[:5])