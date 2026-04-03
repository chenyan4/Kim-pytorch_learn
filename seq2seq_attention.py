# seq2seq 使用注意力机制

# 编码器的对每一个词的输出值 作为key和value
# 解码器 将上一个预测词的隐层作为query，这样其实 query和key 在一个记忆空间

import torch
from torch import nn
import math
from seq2seq import draw_loss
import collections
from transfer_data_predo import load_data_nmt

# 注意力解码器 架构
class AttentionDecoder(nn.Module):
    def __init__(self):
        super(AttentionDecoder,self).__init__()

    def attention_weights(self):
        raise NotImplementedError

def sequence_mask(X,valid_lens,value):
    num_queries=len(X)
    for i in range(num_queries):
        # valid_len 常为 float（如 dataset 里 .sum()），切片必须是 int
        X[i, int(valid_lens[i].item()) :] = value
    return X

def masked_softmax(X,valid_lens=None):
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    shape=X.shape
    if valid_lens.dim()==1:
        valid_lens=valid_lens.repeat_interleave(X.shape[1])
    else:
        valid_lens=valid_lens.reshape(-1)
    X=sequence_mask(X.reshape(-1,X.shape[2]),valid_lens,value=-1e6)
    return nn.functional.softmax(X.reshape(shape),dim=-1)
    

class AdditiveAttention(nn.Module):
    def __init__(self,query_size,key_size,num_hiddens,dropout):
        super(AdditiveAttention,self).__init__()
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens):
        queries,keys=self.W_q(queries),self.W_k(keys)
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        scores=self.w_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

class Seq2SeqEncoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super(Seq2SeqEncoder,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.GRU=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    
    def forward(self,X):
        X=self.embedding(X).permute(1,0,2)
        output,state=self.GRU(X)
        return output,state

class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super(Seq2SeqAttentionDecoder,self).__init__()
        self.attention=AdditiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout=dropout)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.GRU=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens):
        outputs,hidden_state=enc_outputs
        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)

    def forward(self,X,state):
        enc_outputs,hidden_state,enc_valid_lens=state
        X=self.embedding(X).permute(1,0,2)
        outputs,self._attention_weights=[],[]
        for x in X:
            query=torch.unsqueeze(hidden_state[-1],dim=1) # 取最后一个深度的state(靠近输出) 作为我的decoder初始state [batch_size,num_hiddens],加一个num_queries维度，[batch_size,1,num_hiddens]
            context=self.attention(query,enc_outputs,enc_outputs,enc_valid_lens) # 输出就是[batch_size,num_queries,value_size],在这里就是 [batch_size,1,num_hiddes]
            # context后拼接输入的x，得到新的x
            x=torch.cat((context,x.unsqueeze(1)),dim=-1)  
            out,hidden_state=self.GRU(x.permute(1,0,2),hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs=self.dense(torch.cat(outputs,dim=0))
        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]

    def attention_weights(self):
        return self._attention_weights  

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,X,dec_X,enc_valid_lens):
        enc_outputs=self.encoder(X)
        state=self.decoder.init_state(enc_outputs,enc_valid_lens)
        output,state=self.decoder(dec_X,state)
        return output,state

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_lens):
        weights=torch.ones_like(label)
        weights=sequence_mask(weights,valid_lens,value=0)
        self.reduction = 'none'  # 否则父类返回标量，无法按 batch×步长 reshape
        batch_size,num_classes=pred.shape[0],pred.shape[-1]
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.reshape(-1,num_classes),label.reshape(-1)).reshape(batch_size,-1)
        weighted_loss=(weights*unweighted_loss).mean(dim=1)
        return weighted_loss

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters()]
    else:
        params=net.params
    norm=torch.sqrt(sum((torch.sum(p.grad**2) for p in params)))
    if norm>theta:
        for param in params:
            param.grad*=theta/norm

def train_seq2seq(net,train_iter,lr,num_epochs,vocab,device):
    def xavier_init_weights(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    updater=torch.optim.Adam(net.parameters(),lr=lr)
    loss=MaskedSoftmaxCELoss()

    train_loss=[]
    for epoch in range(num_epochs):
        l_num,num=0,0
        for batch in train_iter:
            X,X_valid_len,Y,Y_valid_len=[x.to(device) for x in batch]
            bos=torch.tensor([vocab.__getitem__('<bos>')]*Y.shape[0],device=device).reshape(-1,1)
            dec_X=torch.cat([bos,Y[:,:-1]],dim=-1)

            updater.zero_grad()
            Y_hat,_=net(X,dec_X,X_valid_len)
            l=loss(Y_hat,Y,Y_valid_len)
            l.sum().backward()
            updater.step()

            l_num += l.sum().item()
            num += len(X)

        train_loss.append(l_num / num)
        if (epoch+1)%10==0:
            print(f'epoch:{epoch+1},train_loss:{train_loss[-1]}')
    return train_loss

def truncate_pad(line,num_steps,value):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[value]*(num_steps-len(line))

def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device,save_attention_weights=False):
    net.eval()
    src_tokens=src_vocab.__getitem__(src_sentence.split()+['eos'])
    enc_valid_len=torch.tensor([min(len(src_tokens),num_steps)],device=device)
    src_tokens=truncate_pad(src_tokens,num_steps,value=src_vocab.__getitem__('<pad>'))
    enc_X=torch.tensor(src_tokens,device=device).unsqueeze(0)

    dec_X=torch.tensor([tgt_vocab.__getitem__('<bos>')],device=device).unsqueeze(0)

    enc_outputs=net.encoder(enc_X)
    state=net.decoder.init_state(enc_outputs,enc_valid_len)
    output_seq,attention_weight_seq=[],[]
    for i in range(num_steps):
        dec_X,state=net.decoder(dec_X,state)
        dec_X=dec_X.argmax(dim=-1)
        pred=dec_X.squeeze(0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention.attention_weights)
        if pred==tgt_vocab.__getitem__('<eos>'):
            break
        output_seq.append(pred)
    # 用空格拼接，否则 ''.join 中间无空格，整句会被 split() 当成一个词，len_pred=1 时 n=2 会 denom=0 除零
    translation = ' '.join([tgt_vocab.to_tokens(idx) for idx in output_seq])
    return translation, attention_weight_seq

def blue(pred_seq,label_seq,k):
    pred_seq,label_seq=pred_seq.split(),label_seq.split()
    len_pred,len_label=len(pred_seq),len(label_seq)
    score=math.exp(min(0,1-(len_label/len_pred)))
    for n in range(1,k+1):
        num_matches,label_subs=0,collections.defaultdict(int)
        for i in range(len_label-n+1):
            label_subs[''.join(label_seq[i:i+n])]+=1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred_seq[i:i+n])] and label_subs[''.join(pred_seq[i:i+n])]>0:
                num_matches+=1
                label_subs[''.join(pred_seq[i:i+n])]-=1
        score=score*math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score

if __name__=="__main__":
    embed_size,num_hiddens,num_layers,dropout=32,32,2,0.1
    batch_size,num_steps=64,10
    lr,num_epochs,device=0.005,300,'cuda:0'

    train_iter,src_vocab,tgt_vocab=load_data_nmt(batch_size,num_steps)
    encoder=Seq2SeqEncoder(src_vocab.__len__(),embed_size,num_hiddens,num_layers,dropout)
    decoder=Seq2SeqAttentionDecoder(tgt_vocab.__len__(),embed_size,num_hiddens,num_layers,dropout)

    net=EncoderDecoder(encoder,decoder)
    train_loss=train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
    draw_loss(train_loss,'seq2seq_attention')

    engs=['go .','i lost .','he\'s calm .','i\'m home .']
    fras=['va !','j\'ai perdu .','il est calme .','je suis chez moi .']
    for eng,fra in zip(engs,fras):
        translation,attention_weight_seq=predict_seq2seq(net,eng,src_vocab,tgt_vocab,num_steps,device,save_attention_weights=True)
        print(f'{eng}=>{translation},blue:{blue(translation,fra,k=2)}')






