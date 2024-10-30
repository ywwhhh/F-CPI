
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Modules import ScaledDotProductAttention
import torch

class Attention_sum(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, dropout=0.1):
        super().__init__()




        self.q = torch.nn.Parameter(torch.ones(1,1,512), requires_grad=True)

        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(temperature=d_model ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x, mask=None):




        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        q = self.q
        k = self.w_ks(x)
        v = self.w_vs(x)

        # Transpose for attention dot product: b x n x lq x dv
        # For head axis broadcasting.
#attn是没和v乘之前的softmax
        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)

        return q.view(-1, q.size(-1))

class Attention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


