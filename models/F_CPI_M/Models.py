
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer,DecoderLayer
from .SubLayers import MultiHeadAttention,PositionwiseFeedForward
import  torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.autograd import Variable
from collections import OrderedDict


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


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=256, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)



        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_gin_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=256, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        #self.react_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # self.fnf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self.qnf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        ###########
        self.ll = nn.Linear(512, 192, bias=False)
        self.pp = nn.Linear(512, 192, bias=False)
        self.react_emb = nn.Embedding(8, 128)
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        #############
        self.fuse = PositionwiseFeedForward(d_model*3, d_inner, dropout=dropout)
        self.down_sample = nn.Linear(d_model*3, d_model)
        ###
        self.mo_fuse = PositionwiseFeedForward(d_model*2, d_inner, dropout=dropout)
        self.mo_sample = nn.Linear(d_model*2, d_model)
        #####
        self.mol1 = nn.Linear(512, 512, bias=False)
        self.mol2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        ####
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, f_seq, f_mask, nf_seq, nf_mask, enc_output, src_mask, react, f_mo, nf_mo,return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []
        f_output = self.trg_word_emb(f_seq)
        nf_output = self.trg_word_emb(nf_seq)




        # -- Forward
        if self.scale_emb:
            f_output *= self.d_model ** 0.5
            nf_output *= self.d_model ** 0.5
        f_output = self.dropout(self.position_enc(f_output))
        f_output = self.layer_norm(f_output)
        nf_output = self.dropout(self.position_enc(nf_output))
        nf_output = self.layer_norm(nf_output)

        ########################获得邻接矩阵

        ########################

        for dec_layer in self.layer_stack:
            f_output = dec_layer(
                f_output, slf_attn_mask=f_mask)
        for dec_layer in self.layer_stack:
            nf_output = dec_layer(
                nf_output, slf_attn_mask=nf_mask)
        f_mo = self.mol2(self.mol1(f_mo))
        nf_mo = self.mol2(self.mol1(nf_mo))

        f_output = self.mo_sample(self.mo_fuse(torch.cat((f_output.mean(1), f_mo), -1)))
        nf_output = self.mo_sample(self.mo_fuse(torch.cat((nf_output.mean(1), nf_mo), -1)))

        out_put = self.down_sample(self.fuse(torch.cat((enc_output,f_output, nf_output), -1)))

        react = self.react_emb(react)
        f_output = self.mlp(torch.cat((self.pp(enc_output), self.ll(f_output), react), 1))
        nf_output = self.mlp(torch.cat((self.pp(enc_output),self.ll(nf_output), react), 1))
        # f_output = self.mlp(f_output.mean(1))
        # nf_output = self.mlp(nf_output.mean(1))

        return out_put,f_output, nf_output




class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab=35, n_trg_vocab=39, src_pad_idx=0, trg_pad_idx=0,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers_e=4, n_layers_d=3, n_gin_layers=2, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=1024,
            trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False,
            scale_emb_or_prj='prj',pre_train=None):

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

        # self.encoder = Encoder(
        #     n_src_vocab=n_src_vocab, n_position=n_position,
        #     d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
        #     n_layers=n_layers_e, n_head=n_head, d_k=d_k, d_v=d_v,
        #     pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_d, n_gin_layers=n_gin_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, 2, bias=False)
        self.trg_act_prj = nn.Linear(d_model, 1, bias=False)
        # self.active_prj = nn.Linear(d_model, 1, bias=False)
        self.pro = nn.Linear(768, 512, bias=False)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pssm_ll = nn.Linear(400, 512, bias=False)
        self.pssm_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pro_fuse = PositionwiseFeedForward(d_model * 2, d_inner, dropout=dropout)
        self.pro_sample = nn.Linear(d_model * 2, d_model)
        #对所有参数矩阵的初始化方法
        if pre_train:
            checkpoint = torch.load(pre_train, map_location=torch.device('cpu'))

            state_dict = checkpoint['model']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    name = k[7:]  # remove `module.model.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            self.load_state_dict(new_state_dict)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_uniform_(p)

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
        f_seq = f_all[0]
        nf_seq = nf_all[0]
        f_mo = f_all[1]
        nf_mo = nf_all[1]
        pro_seq = pro_all[0]
        pssm = pro_all[1]
        pro_mask = torch.ones(pro_seq.size(0),1,1).to(pro_seq)
        f_mask = get_pad_mask(f_seq, self.trg_pad_idx)
        nf_mask = get_pad_mask(nf_seq, self.trg_pad_idx)
        #print("encoder input shape:",src_seq.shape,src_mask.shape)#(N,S) (N,1,S)
        #enc_output = self.encoder(pro_seq, pro_mask)
        enc_output = self.pro(pro_seq)
        enc_output = self.pos_ffn(enc_output)
        pssm = self.pssm_ffn(self.pssm_ll(pssm))
        enc_output = self.pro_sample(self.pro_fuse(torch.cat((enc_output, pssm), -1)))
        #print("decoder input shape:", trg_seq.shape, trg_mask.shape,enc_output.shape,src_mask.shape)  # (N,T), (N,T,T), (N,S,E), (N,1,S)
        dec_output,f_out,nf_out = self.decoder(f_seq, f_mask, nf_seq, nf_mask, enc_output,pro_mask, react,f_mo,nf_mo)
        #print("decoder output shape", dec_output.shape)#(N,T,E)
        #dec_output = dec_output.mean(-2)

        seq_logit = self.trg_word_prj(dec_output)
        f_out = self.trg_act_prj(f_out)
        nf_out = self.trg_act_prj(nf_out)


        seq_logit *= self.d_model ** -0.5
        #print("transformer out shape:",seq_logit.shape)#(N,T,V)
        return seq_logit.view(-1, seq_logit.size(-1)), f_out.view(-1, f_out.size(-1)),nf_out.view(-1, nf_out.size(-1))

    