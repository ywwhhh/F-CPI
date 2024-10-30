import torch
import torch.nn.functional as F


class MSE(torch.nn.Module):
    def __init__(self, a,corr):
        super(MSE, self).__init__()
        self.a = a
        self.corr = corr

    def forward(self, pred, gold):

        gold = gold.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        loss = F.mse_loss(pred, gold, reduction='sum')
        return loss * self.a*self.corr
