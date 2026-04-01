# 编码器和解码器
# 编码器：将输入编程成中间表达形式，将文本表示成向量
# 解码器：将中间表示解码成输出，将向量表示成输出
# 编码器处理输出，解码器生成输出 ，input->Encoder->state->Decoder(<-input 额外输入)->output

from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

    def forward(self,X):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

    # enc_outputs 是encoder所有的输出；初始化状态
    def init_state(self,enc_outputs):
        raise NotImplementedError

    # 来一个额外的输入，可以不断更新 架构
    def forward(self,X,state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,enc_X,dec_X):
        enc_outputs=self.encoder(enc_X)
        dec_state=self.decoder.init_state(enc_outputs) # 解码器状态
        return self.decoder(dec_X,dec_state) # 解码器输入和状态 得到结果