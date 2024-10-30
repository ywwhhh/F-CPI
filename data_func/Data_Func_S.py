

import itertools
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from utils import Transform
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm


class Data_Func_S(Dataset):

    def __init__(self, data, mode, ind):
        super(Data_Func_S,self).__init__()
        self.data_set = data
        self.mode = mode
        self.dict = {'Ki': 1, 'Potency': 2, 'IC50': 3, 'EC50': 4, 'Kd': 5, 'AC50': 6}
        if mode=='inf':
            self.pro = np.load('/home/ywh/flourine_smile_68test/data/data'+ind+'/inf_pro_feature.npz', allow_pickle=True)
            self.pssm = np.load('/home/ywh/flourine_smile_68test/data/data'+ind+'/pssm_400.npz', allow_pickle=True)
            self.mol = [token.strip().split('_') for token in
                        open('/home/ywh/flourine_smile_68test/data/data' + ind + '/mol_' + mode)]

            self.data = self.process_inf()
        else:
            self.pro = np.load('data/all_pro_feature_full.npz', allow_pickle=True)
            self.mol = np.load('data/data' + ind + '/' + mode + '_full_emb.npz', allow_pickle=True)
            self.data = self.process()





    def process(self):
        data = []
        transform_ob = Transform()
        bias = {'ligand': 1, 'protein': 1, 'reaction': 1}
        i = 0
        for line in tqdm(self.data_set, mininterval=2, desc='loading', leave=False):
            f = self.mol[str(i)+'_f']
            nf = self.mol[str(i)+'_nf']

            pro = line[4]

            #pro = np.reshape(self.pro[line[4]], [1])[0]['seq'][0] #cls
            f = torch.tensor(f, dtype=torch.float)
            nf = torch.tensor(nf, dtype=torch.float)

            #######################
            r = line[6].split(' ')[0]
            react = torch.tensor(self.dict[r]) if r in self.dict else torch.tensor(0)
            #react = torch.tensor(self.dict[line[6].split(' ')[0]])
            # react = torch.tensor(1.0)
            #######################
            gold = [float(line[-1].split(' ')[1]), float(line[-1].split(' ')[2]),float(line[-1].split(' ')[3])]
            data.append((pro,f, nf,react, gold))
            i+=1
        return data
    def process_inf(self):
        data = []
        transform_ob = Transform()
        bias = {'ligand': 1, 'protein': 1, 'reaction': 1}
        for i, line in enumerate(self.data_set):
            f = transform_ob.get_enc(line[0], 'ligand', 1, bias['ligand'])
            nf = transform_ob.get_enc(line[2], 'ligand', 1, bias['ligand'])
            f_mo = list(map(int, self.mol[i][0].split()))
            nf_mo = list(map(int, self.mol[i][1].split()))
            #####################
            # pro = np.reshape(self.pro[line[4]], [1])[0]['avg']
            pro = np.reshape(self.pro['sp|P0DTD1|R1AB_SARS2|3264-3569'], [1])[0]['avg']
            pssm = self.pssm[line[4]]
            #pro = np.reshape(self.pro[line[4]], [1])[0]['seq'][0] #cls
            f = torch.tensor(f, dtype=torch.long)
            nf = torch.tensor(nf, dtype=torch.long)
            f_mo = torch.tensor(f_mo, dtype=torch.float)
            nf_mo = torch.tensor(nf_mo, dtype=torch.float)
            pro = torch.tensor(pro, dtype=torch.float)
            pssm = torch.tensor(pssm, dtype=torch.float)
            #######################
            # r = line[6].split(' ')[0]
            # react = torch.tensor(self.dict[r]) if r in self.dict else torch.tensor(0)
            #react = torch.tensor(self.dict[line[6].split(' ')[0]])
            # react = torch.tensor(1.0)
            #######################
            # gold = [float(line[-1].split(' ')[1]), float(line[-1].split(' ')[2]),float(line[-1].split(' ')[3])]
            react = torch.tensor(1,dtype=torch.long)
            #######################
            gold = [i, i, i]
            data.append((pro,pssm, f, nf,f_mo,nf_mo, react, gold))
        return data

    def rand_sample(self,ratio = 0.1):
        """从总数据集中随机抽取ratio比例进行验证,近在valid模式中起作用
        Arg:
            ratio: rand sample ratio
        """
        k = int(len(self.full_validate)*ratio)
        #print("before validate sample:",len(self.data))
        # print("before validate sample:",self.data[0][0].shape)
        self.data = random.sample(self.full_validate,k)
        #print("after validate sample:",len(self.data))
        # print("after validate sample:",self.data[0][0].shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]


def generate_batch_S(data_batch):
    pro_batch = []
    f_batch = []
    nf_batch = []

    react_batch = []
    gold_batch = []

    for pro_item,f_item, nf_item,react, gold in data_batch:  # 开始对一个batch重的每一个样本进行处理
        pro_batch.append(pro_item)  # 编码器输入序列不需要加起止符
        f_batch.append(f_item)
        nf_batch.append(nf_item)

        react_batch.append(react)
        gold_batch.append(gold)




    f_batch = pad_sequence(f_batch, padding_value=0).transpose(0, 1)
    nf_batch = pad_sequence(nf_batch, padding_value=0).transpose(0, 1)

    react_batch = torch.tensor(react_batch)

    gold_batch = torch.tensor(gold_batch)
    gold_batch = gold_batch.view(-1, 3)
    return pro_batch, f_batch, nf_batch, react_batch, gold_batch

class Data_Func_S_Metric(object):
    """
    用于单机或分布式训练的评价标准的类
    需要评价训练结果时候在每个机器中单独计算正确词数量和总词数，用distributed_sum函数把不同机器上的参数汇总到0号机器中计算最终的评价结果
    """

    def __init__(self, trg_pad_idx=0):
        super(Data_Func_S_Metric, self).__init__()

        self.n_word, self.n_correct = 0, 0
        self.total_loss1 = 0
        self.total_loss2 = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


    def update(self, pred, gold, loss1, loss2):
        self.total_loss1 += loss1
        self.total_loss2 += loss2
        self.n_word += len(gold)
        correct = 0
        pred = pred.max(1)[1]
        for i in range(len(gold)):
            if gold[i][2] < 0.5:
                gold[i][2] = 0
            else:
                gold[i][2] = 1
        gold = gold[:, 2].view(-1)
        correct = pred.eq(gold).sum().item()
        self.tp += (pred*gold).sum().item()
        self.tn += ((torch.ones_like(pred)-pred)*(torch.ones_like(gold)-gold)).sum().item()
        self.fp += (pred*(torch.ones_like(gold)-gold)).sum().item()
        self.fn += (gold*(torch.ones_like(pred)-pred)).sum().item()
        # print(self.tp)
        # print(self.tn)
        # print(self.fp)
        # print(self.fn)
        self.n_correct += correct

    def compute(self):
        loss_per_word = self.total_loss1 / self.n_word
        loss_per_label = self.total_loss2 / self.n_word
        accuracy = self.n_correct / self.n_word
        recall = self.tp/(self.tp+self.fn) if not (self.tp+self.fn)==0 else 0
        precision = self.tp/(self.tp+self.fp) if not (self.tp+self.fp)==0 else 0
        recall_b = self.tn/(self.tn+self.fp) if not (self.tn+self.fp)==0 else 0
        ba = (recall + recall_b)/2

        return loss_per_word, loss_per_label, accuracy,recall,precision,ba
