import torch
import data_func
import models
import criterions
from utils import ScheduledOptim, get_output_dir, set_logger, Pre_data
import time
from collections import OrderedDict
from tqdm import tqdm
import logging
import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_root = 'data/data24_1'
model_name = 'Model_51_1'
data_pre_func = 'Data_Func_18'
pretrained_path = "experiment/Model_51_1_Data_Func_18_data18_3_Cross_Entropy_s9_MSE/2024-04-22-06-00/model_accuracy.chkpt"
batch_size = 1
end_epoch = 7000
device_ids = 0
dropout = 0.1
n_layer_d = 1
n_layer_g = 3
loss_a = 0.7
corr = 1
##focal loss
gamma = 5

alpha = 0.4

model = eval("models." + model_name)(dropout=dropout,n_layers_d=n_layer_d,n_gin_layers=n_layer_g)


checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))

state_dict = checkpoint['model']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k[:7] == 'module.':
        name = k[7:]
        new_state_dict[name] = v
model.load_state_dict(new_state_dict)





model.eval()
metric = eval('data_func.'+data_pre_func+'_Metric')()
tic = time.time()
desc = '  - (Testing)   '
indx = 0



data = Pre_data('inf', data_pre_func, batch_size, inf_data=data_root)
loaders = data.get_loaders()
#valid
inf_loader = loaders['inf']
suc = []
with torch.no_grad():
    for batch in tqdm(inf_loader, mininterval=2, desc=desc, leave=False):
        indx += 1
        pro_seq, f_seq, nf_seq, react, gold = batch
        # pro_seq = pro_seq.to(torch.float).cuda()
        pro_seq = pro_seq
        f_seq = f_seq
        nf_seq = nf_seq
        react = react
        gold = gold

        # pred,_,_ = model(pro_seq, f_seq, nf_seq, react)
        pred,_,_ = model(pro_seq, f_seq, nf_seq, react)
        ##################################################
        print(pred)
        if pred[0][0]<pred[0][1]:
            prb = F.softmax(pred, dim=1)

            suc.append([int(gold[0][0].item()),prb[0][1].item()])

# source = line.split('$')[0]
# reps = line.split('$')[1].split('_')
print(suc)


o = open(data_root+'/positive_pre','w')
vocab = [token.strip() for token in open(data_root+'/inf_set')]
o_list = []
for i in suc:
    o_list.append((i[1],vocab[i[0]]))
o_list = sorted(o_list,key=lambda x:(x[0],x[1]),reverse=True)
print(o_list)
for score,tx in o_list:
    o.write(str(score)+'_'+tx+'\n')
