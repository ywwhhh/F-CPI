''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer,DecoderLayer
from .SubLayers import MultiHeadAttention,PositionwiseFeedForward
import  torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.autograd import Variable
from collections import OrderedDict


__author__ = "Yu-Hsiang Huang"

import torch
from torch.nn.utils.rnn import pad_sequence





def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)



def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=256):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        #a = self.pos_table[:, :x.size(1)].clone().detach()
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder_p(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=256, scale_emb=False):

        super().__init__()

        self.emb = nn.Linear(768, 512, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.emb(src_seq)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)



        return enc_output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=256, scale_emb=False):

        super().__init__()

        self.emb = nn.Linear(384, 512, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.emb(src_seq)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)



        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_gin_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=256, dropout=0.1, scale_emb=False):

        super().__init__()


        # self.fnf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self.qnf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        ###########
        self.ll = nn.Linear(512, 256, bias=False)

        self.react_emb = nn.Embedding(7, 256)
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        #############
        self.attn_p = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn_p = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


        ####
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, f_seq,f_mask, nf_seq,nf_mask, enc_output, src_mask, react,return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        f_seq = self.ffn_p(self.attn_p(f_seq,enc_output,enc_output,mask=src_mask))
        nf_seq = self.ffn_p(self.attn_p(nf_seq,enc_output,enc_output,mask=src_mask))

        qf_output = self.ffn(self.attn(f_seq, nf_seq, nf_seq, mask=nf_mask).mean(1))
        qnf_output = self.ffn(self.attn(nf_seq, f_seq, f_seq, mask=f_mask).mean(1))

        react = self.react_emb(react)
        f_output = self.mlp(torch.cat((self.ll(f_seq.mean(1)), react), 1))
        nf_output = self.mlp(torch.cat((self.ll(nf_seq.mean(1)), react), 1))
        # f_output = self.mlp(f_output.mean(1))
        # nf_output = self.mlp(nf_output.mean(1))

        return qf_output - qnf_output, f_output, nf_output





class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab=23, n_trg_vocab=39, src_pad_idx=0, trg_pad_idx=0,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers_e=3, n_layers_d=3, n_gin_layers=2, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=1024,
            trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False,
            scale_emb_or_prj='prj',from_pre=None,pre_train=None):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder_p = Encoder_p(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_e, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)
        self.encoder = Encoder(
            n_src_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_e-1, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_d, n_gin_layers=n_gin_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, 2, bias=False)
        self.trg_act_prj = nn.Linear(d_model, 1, bias=False)
        # self.active_prj = nn.Linear(d_model, 1, bias=False)

        #对所有参数矩阵的初始化方法
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
        if from_pre:
            checkpoint = torch.load(from_pre, map_location=torch.device('cpu'))

            state_dict = checkpoint['model']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    name = k[7:]  # remove `module.model.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            self.load_state_dict(new_state_dict)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        # if emb_src_trg_weight_sharing:
        #     self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, pro_all, f_all, nf_all, react):
        """
        Args:
            src_seq: (N,S)
            trg_seq: (N,T)
        """
        #pro_mask = get_pad_mask(pro_seq, self.src_pad_idx)
        f_seq = f_all
        nf_seq = nf_all
        pro_seq = pro_all
        pro_mask = get_pad_mask(pro_seq.sum(-1), self.trg_pad_idx)
        f_mask = get_pad_mask(f_seq.sum(-1), self.trg_pad_idx)
        nf_mask = get_pad_mask(nf_seq.sum(-1), self.trg_pad_idx)
        #print("encoder input shape:",src_seq.shape,src_mask.shape)#(N,S) (N,1,S)
        #enc_output = self.encoder(pro_seq, pro_mask)
        enc_output = self.encoder_p(pro_seq,pro_mask)
        f_out = self.encoder(f_seq,f_mask)
        nf_out = self.encoder(nf_seq, nf_mask)
        #print("decoder input shape:", trg_seq.shape, trg_mask.shape,enc_output.shape,src_mask.shape)  # (N,T), (N,T,T), (N,S,E), (N,1,S)
        dec_output,f_out,nf_out = self.decoder(f_out, f_mask, nf_out, nf_mask, enc_output,pro_mask, react)
        #print("decoder output shape", dec_output.shape)#(N,T,E)
        #dec_output = dec_output.mean(-2)

        seq_logit = self.trg_word_prj(dec_output)
        f_out = self.trg_act_prj(f_out)
        nf_out = self.trg_act_prj(nf_out)


        seq_logit *= self.d_model ** -0.5
        #print("transformer out shape:",seq_logit.shape)#(N,T,V)
        return seq_logit.view(-1, seq_logit.size(-1)), f_out.view(-1, f_out.size(-1)),nf_out.view(-1, nf_out.size(-1))

    