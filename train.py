import data_func
import models
import criterions
import torch
from utils import ScheduledOptim, get_output_dir, set_logger, Pre_data, train_epoch_label, valid_epoch_label,test_epoch_label
import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False



def train():

    pre_train = None
    model_name = 'F_CPI_M'
    data_pre_func = 'Data_Func_M'
    data_root = 'data0'
    criterion_name_1 = 'Cross_Entropy_s9'
    criterion_name_2 = 'MSE'
    loss_a = 0.7
    batch_size = 144
    corr = 1
    ngpus = 1
    end_epoch = 1000
    grad_accu = 2
    n_warm = 2000
    ##focal loss
    gamma = 5
    alpha = 0.4
    lr = 0.3
    dropout = 0.1
    n_layer_d = 1
    n_layer_g = 3
    n_layer_e = 2
    device_ids = 3
    seed = 233

    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument("--local_rank", default=-1, type=int,
                        help="Local rank for distributed training.")
    args = parser.parse_args()
    rank = args.local_rank
    set_seed(seed)
    if rank != -1:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl',
        )
    # else:
    #     torch.cuda.set_device(device_ids)

    output_dir = get_output_dir(model_name, data_pre_func, data_root, criterion_name_1, criterion_name_2,rank)

    logger = set_logger(output_dir,rank)
    logging.info('batch_size: ' + str(batch_size))
    logging.info('grad_accu: ' + str(grad_accu))
    logging.info('lr: ' + str(lr))
    logging.info('n_warm: ' + str(n_warm))
    logging.info('dropout: ' + str(dropout))
    logging.info('n_layer_e: ' + str(n_layer_e))
    logging.info('n_layer_d: ' + str(n_layer_d))
    logging.info('n_layer_g: ' + str(n_layer_g))
    logging.info('loss_a: ' + str(loss_a))
    logging.info('alpha: ' + str(alpha))
    logging.info('gamma: ' + str(gamma))
    logging.info('corr: ' + str(corr))
    logging.info('seed: ' + str(seed))
    logging.info('pretrain: ' + str(pre_train))




    writer_dict = {
        'writer': SummaryWriter(output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'test_global_steps': 0,
    }


    model = eval("models." + model_name)(dropout=dropout,n_layers_d=n_layer_d,n_gin_layers=n_layer_g,n_layers_e=n_layer_e,pre_train=pre_train)

    # model = torch.nn.DataParallel(model, device_ids=[device_ids]).to(device_ids)

    if rank != -1:
        model = DDP(model.cuda(),
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model)


    data = Pre_data('data/' + data_root, data_pre_func, batch_size,rank)
    loaders = data.get_loaders()
    train_loader = loaders['train']
    valid_loader = loaders['valid']
    test_loader = loaders['test']


    criterion_1 = eval('criterions.' + criterion_name_1)(gamma=gamma,alpha=alpha,corr=corr)
    criterion_2 = eval('criterions.' + criterion_name_2)(loss_a,corr=corr)


    optimizer = ScheduledOptim(
        optimizer=torch.optim.Adam(model.parameters(), betas=(0.9, 0.997),eps=1e-9),
        lr_mul=lr, d_model=512,
        n_warmup_steps=n_warm)


    start_time = timeit.default_timer()
    start_epoch = 0
    cur_iter = 0
    iter_per_epoch = len(train_loader.dataset) // (batch_size*ngpus) // grad_accu
    global_iter = 0

    best_accuracy = 0
    best_recall = 0
    best_precision = 0
    # if data_pre_func == 'Data_Func_S_pre':
    #     pro_emb = np.load('data/all_pro_feature_full.npz', allow_pickle=True)
    #     pro_e = {}
    #     for pro in pro_emb:
    #         p = np.reshape(pro_emb[pro], [1])[0]['seq']
    #         pro_e[pro]=p
    #     pro_emb = pro_e
    # else:
    pro_emb = None
    for epoch in range(start_epoch, end_epoch):
        # Train a epoch
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        train_loss_per_word, train_accuracy, global_iter = train_epoch_label(model, train_loader, optimizer,
                                                                           criterion_1, criterion_2, global_iter,
                                                                           epoch, end_epoch, writer_dict, data_pre_func,
                                                                           device_ids, grad_accu, rank,pro_emb=pro_emb)


        cur_iter += iter_per_epoch

        #Validate
        if ((epoch + 1) % 1) == 0:
            # Rand sample

            valid_loss_per_word, valid_accuracy, valid_recall, valid_precision, valid_ba = valid_epoch_label(model,
                                                                                                             valid_loader,
                                                                                                             criterion_1,
                                                                                                             criterion_2,
                                                                                                             writer_dict,
                                                                                                             data_pre_func,
                                                                                                             device_ids,
                                                                                                             rank,
                                                                                                             pro_emb=pro_emb)

        if rank < 1:
            if valid_recall > best_recall:
                checkpoint = {'epoch': epoch, 'model': model.state_dict()}
                best_recall = valid_recall
                model_path = os.path.join(output_dir, 'model_recall.chkpt')
                torch.save(checkpoint, model_path)
                logger.info('The best recall checkpoint file has been updated.')
                print('Best accuracy: {:3.3f} %,Best recall: {:3.3f},Best precision: {:3.3f}, cur iter: {}'.format(best_accuracy,best_recall,best_precision, cur_iter))

            if valid_accuracy > best_accuracy:
                checkpoint = {'epoch': epoch, 'model': model.state_dict()}
                best_accuracy = valid_accuracy
                model_path = os.path.join(output_dir, 'model_accuracy.chkpt')
                torch.save(checkpoint, model_path)
                logger.info('The best accuracy checkpoint file has been updated.')
                print('Best accuracy: {:3.3f} %,Best recall: {:3.3f},Best precision: {:3.3f}, cur iter: {}'.format(best_accuracy,best_recall,best_precision, cur_iter))
            if valid_precision > best_precision:
                checkpoint = {'epoch': epoch, 'model': model.state_dict()}
                best_precision = valid_precision
                model_path = os.path.join(output_dir, 'model_precision.chkpt')
                torch.save(checkpoint, model_path)
                logger.info('The best precision checkpoint file has been updated.')
                print('Best accuracy: {:3.3f} %,Best recall: {:3.3f},Best precision: {:3.3f}, cur iter: {}'.format(best_accuracy,best_recall,best_precision, cur_iter))

            if ((epoch + 1) % 10) == 0:
                checkpoint = {'epoch': epoch, 'model': model.state_dict()}
                model_path = os.path.join(output_dir, 'model_{}.chkpt'.format(epoch))
                torch.save(checkpoint, model_path)

    writer_dict['writer'].close()
    end_time = timeit.default_timer()
    logging.info('Elapse time: %d hour %d minute !' % (int((end_time - start_time) / 3600), int((end_time - start_time) % 3600 / 60)))
    logging.info('Done!')


if __name__ == '__main__':
    train()
