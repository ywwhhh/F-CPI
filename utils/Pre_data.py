from torch.utils.data import Dataset
import data_func
from torch.utils.data import dataloader, RandomSampler, SequentialSampler
import os
from torch.nn.utils.rnn import pad_sequence
import torch

class Pre_data(Dataset):

    def __init__(self, data_root, data_pre_func, batch_size,rank=-1, inf_data = None,whole=False):
        super(Pre_data, self).__init__()
        self.inf_data = inf_data
        self.root = data_root
        self.rank = rank
        self.data_pre_func =data_pre_func
        self.batch_size = batch_size

        self.loaders = {}
        self.whole = whole
        if data_root == 'inf':
            self.loaders['inf'] = self.get_data_loader('inf')
        else:
            self.modes = ['train', 'valid', 'test']
            for mode in self.modes:
                self.loaders[mode] = self.get_data_loader(mode)

    def get_data_loader(self, mode):
        if mode == 'inf':
            with open(self.inf_data+'/inf_set') as f:
                data = [token.strip().split('_') for token in f]
            ind = self.inf_data[9:]
        else:
            data = self.prepare(mode)
            ind = self.root[9:] if not self.root[9:] in ['10','11','12','13'] else '9'

        dataset = eval('data_func.'+self.data_pre_func)(data, mode, ind,self.whole)

        if mode == 'train':
            if self.rank != -1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            else:
                sampler = RandomSampler(dataset, replacement=False)

            drop_last = True
        else:
            if self.rank != -1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            else:
                sampler = SequentialSampler(dataset)
            drop_last = False

        loader = dataloader.DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       sampler=sampler,
                                       num_workers=8,
                                       pin_memory=True,
                                       drop_last=drop_last,
                                       collate_fn=eval('data_func.generate_batch_'+self.data_pre_func[10:]))

        return loader




    def prepare(self, mode):

        with open(self.root+'/'+mode + '_set') as f:
            vocab = [token.strip().split('_') for token in f]

        return vocab


    def get_loaders(self):
        return self.loaders







