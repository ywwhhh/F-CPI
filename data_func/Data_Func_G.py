

import itertools
import time

import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from utils import Transform
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Data_Func_G(Dataset):

    def __init__(self, data, mode, ind,whole):
        super(Data_Func_G,self).__init__()
        self.data_set = data
        self.mode = mode
        self.dict = {'Ki': 1, 'Potency': 2, 'IC50': 3, 'EC50': 4, 'Kd': 5, 'AC50': 6}
        self.pro = np.load('data/data0/contact_map.npz', allow_pickle=True)
        self.mol = [token.strip().split('_') for token in open('data/data0/'+mode+'_graph')]
        self.data = self.process()




    def process(self):
        data = []
        transform_ob = Transform()
        bias = {'ligand': 1, 'protein': 1, 'reaction': 1}
        for i, line in enumerate(self.data_set):
            f = self.mol[i][0].split('$')
            del f[-1]
            f_ind = self.mol[i][1].split('$')
            del f_ind[-1]
            nf = self.mol[i][2].split('$')
            del nf[-1]
            nf_ind = self.mol[i][3].split('$')
            del nf_ind[-1]

            f_feature = []
            nf_feature = []
            for j in range(len(f)):
                b = [int(k) for k in f[j].split(' ')]
                f_feature.append(b)
            for j in range(len(nf)):
                b = [int(k) for k in nf[j].split(' ')]
                nf_feature.append(b)


            f_index = []
            for i in range(len(f_ind)):
                d = f_ind[i].split(' ')
                f_index.append([int(d[0]), int(d[1])])
            nf_index = []
            for i in range(len(nf_ind)):
                d = nf_ind[i].split(' ')
                nf_index.append([int(d[0]), int(d[1])])

            pro = transform_ob.get_enc(line[5], 'protein', 1, 1)
            if len(pro)>1024:
                pro = pro[:1024]
            pro_index = self.pro[line[4]]
            # pro = np.reshape(self.pro[line[4]], [1])[0]['avg']
            #pro = np.reshape(self.pro[line[4]], [1])[0]['seq'][0] #cls

            f_feature = torch.tensor(f_feature, dtype=torch.long)
            f_index = torch.LongTensor(f_index)
            nf_feature = torch.tensor(nf_feature, dtype=torch.long)
            nf_index = torch.LongTensor(nf_index)
            pro = torch.tensor(pro, dtype=torch.long)
            pro_index = torch.LongTensor(pro_index)
            #######################
            r = line[6].split(' ')[0]
            react = torch.tensor(self.dict[r]) if r in self.dict else torch.tensor(0)
            #react = torch.tensor(self.dict[line[6].split(' ')[0]])
            # react = torch.tensor(1.0)
            #######################
            gold = [float(line[-1].split(' ')[1]), float(line[-1].split(' ')[2]),float(line[-1].split(' ')[3])]
            data.append((pro,pro_index, f_feature, f_index,nf_feature,nf_index, react, gold))
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


def generate_batch_G(data_batch):
    pro_batch = []
    f_batch = []
    nf_batch = []
    f_index_batch = []
    nf_index_batch = []
    pro_index_batch = []
    react_batch = []
    gold_batch = []

    for pro_item,pro_ind, f_item, f_ind,nf_item, nf_ind, react, gold in data_batch:  # 开始对一个batch重的每一个样本进行处理
        pro_batch.append(pro_item)  # 编码器输入序列不需要加起止符
        f_batch.append(f_item)
        nf_batch.append(nf_item)
        f_index_batch.append(f_ind)
        nf_index_batch.append(nf_ind)
        pro_index_batch.append(pro_ind)
        react_batch.append(react)
        gold_batch.append(gold)

    pro_batch = pad_sequence(pro_batch, padding_value=0).transpose(0, 1)
    f_batch = pad_sequence(f_batch, padding_value=0).transpose(0, 1)
    nf_batch = pad_sequence(nf_batch, padding_value=0).transpose(0, 1)
    # f_index_batch = pad_sequence(f_index_batch, padding_value=-1).transpose(0, 1)
    # nf_index_batch = pad_sequence(nf_index_batch, padding_value=-1).transpose(0, 1)
    # pro_index_batch = pad_sequence(pro_index_batch, padding_value=-1).transpose(0, 1)


    len_f = f_batch.size(1)
    f_matrix_batch = []
    for ind in f_index_batch:
        val = torch.ones(len(ind))
        matrix = torch.sparse_coo_tensor(ind.t(), val, (len_f, len_f))
        matrix = matrix.to_dense()
        f_matrix_batch.append(matrix)
    f_matrix_batch = torch.stack(f_matrix_batch)

    len_nf = nf_batch.size(1)
    nf_matrix_batch = []
    for ind in nf_index_batch:
        val = torch.ones(len(ind))
        matrix = torch.sparse_coo_tensor(ind.t(), val, (len_nf, len_nf))
        matrix = matrix.to_dense()
        nf_matrix_batch.append(matrix)
    nf_matrix_batch = torch.stack(nf_matrix_batch)

    len_pro = pro_batch.size(1)
    pro_matrix_batch = []
    for ind in pro_index_batch:
        val = torch.ones(len(ind))
        matrix = torch.sparse_coo_tensor(ind.t(), val, (len_pro, len_pro))
        matrix = matrix.to_dense()
        pro_matrix_batch.append(matrix)
    pro_matrix_batch = torch.stack(pro_matrix_batch)



    react_batch = torch.tensor(react_batch)
    gold_batch = torch.tensor(gold_batch)
    gold_batch = gold_batch.view(-1, 3)


    return [pro_batch,pro_matrix_batch], [f_batch,f_matrix_batch], [nf_batch,nf_matrix_batch], react_batch, gold_batch

class Data_Func_G_Metric(object):
    """
    用于单机或分布式训练的评价标准的类
    需要评价训练结果时候在每个机器中单独计算正确词数量和总词数，用distributed_sum函数把不同机器上的参数汇总到0号机器中计算最终的评价结果
    """

    def __init__(self, trg_pad_idx=0):
        super(Data_Func_G_Metric, self).__init__()

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
