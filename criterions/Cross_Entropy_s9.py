import torch
import torch.nn.functional as F

#不平滑
class Cross_Entropy_s9(torch.nn.Module):
    def __init__(self, gamma, alpha,corr,smoothing=False, reduction='sum'):
        super(Cross_Entropy_s9, self).__init__()

        self.smoothing = smoothing
        assert reduction in ['sum', 'mean']
        self.reduction = reduction
        self.bound_t = 0.5
        self.bound_b = 0
        self.bias = 0.1
        self.gamma = gamma
        self.alpha = alpha
        self.corr = corr
    def forward(self, pred, gold):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        # pred(N*T,V)
        # gold(N*T)
        gold = gold.contiguous().view(-1)


        eps = 0.1
        n_class = pred.size(1)

        # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        one_hot = torch.zeros_like(pred)

        for i in range(one_hot.size(0)):
            one_hot[i][1] = self.f(gold[i], bound_b=self.bound_b, bound_t=self.bound_t, bias=self.bias)*self.alpha
            one_hot[i][0] = (1 - one_hot[i][1])*(1-self.alpha)

        prb = F.softmax(pred, dim=1)
        prb_1 = (torch.ones_like(prb)-prb).pow(self.gamma)
        log_prb = torch.log(prb)

        loss = -(one_hot * log_prb * prb_1).sum(dim=1)
        loss = loss.sum()*self.corr  # average later


        return loss
    def f(self, src, bound_b, bound_t, bias):
        # if bias<0.5:
        #     d = (1/(0.5-bias)) * (bound_t-bound_b)
        #     b = 1 - bound_t / d
        #     if src > 0:
        #         src = src/d+b
        #     elif src == 0:
        #         src = 0.5
        #     else:
        #         src = src/d-b+1
        if src>=0.5:
            src = 1
        else:
            src = 0
        return src

