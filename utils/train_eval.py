import time
from tqdm import tqdm
import torch
import logging
import data_func
from contextlib import suppress as nullcontext
import numpy as np
from torch.nn.utils.rnn import pad_sequence


DATA_FUNCS_COMPOUND= []
DATA_FUNCS = ['Data_Func_M','Data_Func_G']
DATA_FUNCS_EMB = ['Data_Func_S']




def train_epoch_label(model, training_data, optimizer, criterion_1, criterion_2, cur_iter, cur_epoch, end_epoch, writer_dict, data_pre_func, device_ids, grad_accu,rank,pro_emb=None):

    # Training mode
    model.train()
    tic = time.time()
    metric = eval('data_func.'+data_pre_func+'_Metric')()
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        pro_seq, f_seq, nf_seq, react, gold = batch


        if data_pre_func in DATA_FUNCS_COMPOUND:
            f_seq[0] = f_seq[0].cuda()
            f_seq[1] = f_seq[1].cuda()
            nf_seq[0] = nf_seq[0].cuda()
            nf_seq[1] = nf_seq[1].cuda()
            pro_seq = pro_seq.cuda()
        elif data_pre_func in DATA_FUNCS:
            f_seq[0] = f_seq[0].cuda()
            f_seq[1] = f_seq[1].cuda()
            nf_seq[0] = nf_seq[0].cuda()
            nf_seq[1] = nf_seq[1].cuda()
            pro_seq[0] = pro_seq[0].cuda()
            pro_seq[1] = pro_seq[1].cuda()
        elif data_pre_func in DATA_FUNCS_EMB:

            f_seq = f_seq.cuda()
            nf_seq = nf_seq.cuda()
            pro_batch = []
            for pro in pro_seq:
                pro = pro_emb[pro]
                if np.size(pro, 0) > 768:
                    pro = pro[:768]
                pro = torch.tensor(pro, dtype=torch.float)
                pro_batch.append(pro)
            pro_seq = pad_sequence(pro_batch, padding_value=0).transpose(0, 1).cuda()

        else:
            f_seq = f_seq.cuda()
            nf_seq = nf_seq.cuda()
            pro_seq = pro_seq.cuda()
        react = react.cuda()
        gold = gold.cuda()

        my_context = model.no_sync if rank != -1 and (cur_iter + 1) % grad_accu != 0 else nullcontext
        with my_context():
            pred,f_pred,nf_pred = model(pro_seq, f_seq, nf_seq, react)

            #########################################
            # Backward
            loss1 = criterion_1(pred, gold[:, 2])/grad_accu
            loss2 = (criterion_2(f_pred, gold[:, 0]) + criterion_2(nf_pred, gold[:, 1]))/(grad_accu*2)

            loss = loss1 + loss2
            ##flood
            #
            # b = 0.4 * len(gold)
            # flood = (loss-b).abs()+b
            # flood.backward()
            ##
            loss.backward()

            ##


        if ((cur_iter + 1) % grad_accu) == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
            #logging.info("global iter:{}, cur iter:{},update parameter".format(optimizer.n_steps, cur_iter))
        # Update metric
        if criterion_2.a==0:
            metric.update(pred, gold, loss1.item() * grad_accu, 0)
        else:
            metric.update(pred, gold, loss1.item()*grad_accu,loss2.item()*grad_accu/criterion_2.a)
        cur_iter += 1
        # if ((cur_iter + 1) % 8192) == 0:
        #     _, accuracy = metric.compute()
        #     logging.info('{}'.format(accuracy))
    # optimizer.zero_grad()
    # Feedback on console
    loss_per_word,loss_per_label, accuracy,recall,precision, ba = metric.compute()
    accuracy = accuracy*100
    recall = recall * 100
    precision = precision * 100
    ba = ba * 100

    msg = 'Epoch: [{}/{}], loss: {:5.5f}, loss_mse: {:5.5f}, accuracy: {:3.3f} %,recall: {:3.3f} %,precision: {:3.3f} %,ba: {:3.3f} %,lr: {:8.5f}, ' \
          'elapse: {:3.3f} min'.format(
        cur_epoch, end_epoch,  loss_per_word,loss_per_label, accuracy,recall, precision,ba,
        optimizer._optimizer.param_groups[0]['lr'], (time.time()-tic)/60)



    logging.info(msg)

    #tensorboard
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    if rank < 1:

        writer.add_scalar('train_loss', loss_per_word, global_steps)
        writer.add_scalar('train_loss_label', loss_per_label, global_steps)
        writer.add_scalar('train_accuracy', accuracy, global_steps)
        writer.add_scalar('train_recall', recall, global_steps)
        writer.add_scalar('train_precision', precision, global_steps)
        writer.add_scalar('train_ba', ba, global_steps)
        writer.add_scalar('learning rate', optimizer._optimizer.param_groups[0]['lr'], global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


    return loss_per_word, accuracy, cur_iter


def valid_epoch_label(model, validation_data, criterion_1, criterion_2, writer_dict, data_pre_func, device_ids,rank,pro_emb=None):
    ''' Epoch operation in evaluation phase '''
    # eval mode
    model.eval()

    metric = eval('data_func.'+data_pre_func+'_Metric')()
    desc = '  - (Validation) '
    tic = time.time()
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            ##############################################
            # src_seq, trg_seq, gold = batch
            # # Prepare data
            # src_seq = src_seq.to(device_ids)
            # trg_seq = trg_seq.to(device_ids)
            # gold = gold.to(device_ids)
            #
            # # Forward
            # pred = model(src_seq, trg_seq)
            ##############################################
            # nf_seq, f_seq, p_seq, gold = batch
            # nf_seq = nf_seq.cuda()
            # f_seq = f_seq.cuda()
            # p_seq = p_seq.cuda()
            # gold = gold.cuda()
            # pred = model(nf_seq, f_seq, p_seq)
            ##################################################
            pro_seq, f_seq, nf_seq, react, gold = batch
            #pro_seq = pro_seq.to(torch.float).cuda()

            if data_pre_func in DATA_FUNCS_COMPOUND:
                f_seq[0] = f_seq[0].cuda()
                f_seq[1] = f_seq[1].cuda()
                nf_seq[0] = nf_seq[0].cuda()
                nf_seq[1] = nf_seq[1].cuda()
                pro_seq = pro_seq.cuda()
            elif data_pre_func in DATA_FUNCS:
                f_seq[0] = f_seq[0].cuda()
                f_seq[1] = f_seq[1].cuda()
                nf_seq[0] = nf_seq[0].cuda()
                nf_seq[1] = nf_seq[1].cuda()
                pro_seq[0] = pro_seq[0].cuda()
                pro_seq[1] = pro_seq[1].cuda()
            elif data_pre_func in DATA_FUNCS_EMB:

                f_seq = f_seq.cuda()
                nf_seq = nf_seq.cuda()
                pro_batch = []
                for pro in pro_seq:
                    pro = pro_emb[pro]
                    if np.size(pro, 0) > 768:
                        pro = pro[:768]
                    pro = torch.tensor(pro, dtype=torch.float)
                    pro_batch.append(pro)
                pro_seq = pad_sequence(pro_batch, padding_value=0).transpose(0, 1).cuda()

            else:
                f_seq = f_seq.cuda()
                nf_seq = nf_seq.cuda()
                pro_seq = pro_seq.cuda()
            react = react.cuda()
            gold = gold.cuda()

            pred,f_pred, nf_pred = model(pro_seq, f_seq, nf_seq, react)

            ##################################################


            loss1 = criterion_1(pred, gold[:, 2])
            loss2 = (criterion_2(f_pred, gold[:, 0]) + criterion_2(nf_pred, gold[:, 1])) / 2

            # Update metric
            if criterion_2.a == 0:
                metric.update(pred, gold, loss1.item(), 0)
            else:
                metric.update(pred, gold, loss1.item(),loss2.item()/criterion_2.a)



    loss_per_word,loss_per_label, accuracy,recall,precision, ba = metric.compute()
    accuracy = accuracy * 100
    recall = recall*100
    precision = precision*100
    ba = ba*100

    msg = 'Validating result:, loss: {:8.5f}, loss_mse: {:8.5f}, accuracy: {:3.3f} %,recall: {:3.3f} %,precision: {:3.3f} %,ba: {:3.3f} %,  ' \
          'elapse: {:3.3f} min'.format(
        loss_per_word,loss_per_label, accuracy,recall, precision,ba,(time.time() - tic) / 60)

    logging.info(msg)

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    if rank < 1:

        writer.add_scalar('valid_loss', loss_per_word, global_steps)
        writer.add_scalar('valid_loss_mse', loss_per_label, global_steps)
        writer.add_scalar('valid_accuracy', accuracy, global_steps)
        writer.add_scalar('valid_recall', recall, global_steps)
        writer.add_scalar('valid_precision', precision, global_steps)
        writer.add_scalar('valid_ba', ba, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return loss_per_word, accuracy,recall, precision,ba

def test_epoch_label(model, validation_data, criterion_1, criterion_2, writer_dict, data_pre_func, device_ids,rank,pro_emb=None):
    ''' Epoch operation in evaluation phase '''
    # eval mode
    model.eval()

    metric = eval('data_func.'+data_pre_func+'_Metric')()
    desc = '  - (Testing) '
    tic = time.time()
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            ##############################################
            # src_seq, trg_seq, gold = batch
            # # Prepare data
            # src_seq = src_seq.to(device_ids)
            # trg_seq = trg_seq.to(device_ids)
            # gold = gold.to(device_ids)
            #
            # # Forward
            # pred = model(src_seq, trg_seq)
            ##############################################
            # nf_seq, f_seq, p_seq, gold = batch
            # nf_seq = nf_seq.cuda()
            # f_seq = f_seq.cuda()
            # p_seq = p_seq.cuda()
            # gold = gold.cuda()
            # pred = model(nf_seq, f_seq, p_seq)
            ##################################################
            pro_seq, f_seq, nf_seq, react, gold = batch
            #pro_seq = pro_seq.to(torch.float).cuda()

            if data_pre_func in DATA_FUNCS_COMPOUND:
                f_seq[0] = f_seq[0].cuda()
                f_seq[1] = f_seq[1].cuda()
                nf_seq[0] = nf_seq[0].cuda()
                nf_seq[1] = nf_seq[1].cuda()
                pro_seq = pro_seq.cuda()
            elif data_pre_func in DATA_FUNCS:
                f_seq[0] = f_seq[0].cuda()
                f_seq[1] = f_seq[1].cuda()
                nf_seq[0] = nf_seq[0].cuda()
                nf_seq[1] = nf_seq[1].cuda()
                pro_seq[0] = pro_seq[0].cuda()
                pro_seq[1] = pro_seq[1].cuda()
            elif data_pre_func in DATA_FUNCS_EMB:

                f_seq = f_seq.cuda()
                nf_seq = nf_seq.cuda()
                pro_batch = []
                for pro in pro_seq:
                    pro = pro_emb[pro]
                    if np.size(pro, 0) > 768:
                        pro = pro[:768]
                    pro = torch.tensor(pro, dtype=torch.float)
                    pro_batch.append(pro)
                pro_seq = pad_sequence(pro_batch, padding_value=0).transpose(0, 1).cuda()

            else:
                f_seq = f_seq.cuda()
                nf_seq = nf_seq.cuda()
                pro_seq = pro_seq.cuda()
            react = react.cuda()
            gold = gold.cuda()

            pred,f_pred, nf_pred = model(pro_seq, f_seq, nf_seq, react)

            ##################################################


            loss1 = criterion_1(pred, gold[:, 2])
            loss2 = (criterion_2(f_pred, gold[:, 0]) + criterion_2(nf_pred, gold[:, 1])) / 2

            # Update metric
            if criterion_2.a == 0:
                metric.update(pred, gold, loss1.item(), 0)
            else:
                metric.update(pred, gold, loss1.item(), loss2.item() / criterion_2.a)


    loss_per_word,loss_per_label, accuracy,recall,precision, ba = metric.compute()
    accuracy = accuracy * 100
    recall = recall*100
    precision = precision*100
    ba = ba*100

    msg = 'Testing result:, loss: {:8.5f}, loss_mse: {:8.5f}, accuracy: {:3.3f} %,recall: {:3.3f} %,precision: {:3.3f} %,ba: {:3.3f} %,  ' \
          'elapse: {:3.3f} min'.format(
        loss_per_word,loss_per_label, accuracy,recall, precision,ba,(time.time() - tic) / 60)

    logging.info(msg)

    writer = writer_dict['writer']
    global_steps = writer_dict['test_global_steps']
    if rank < 1:

        writer.add_scalar('test_loss', loss_per_word, global_steps)
        writer.add_scalar('test_loss_mse', loss_per_label, global_steps)
        writer.add_scalar('test_accuracy', accuracy, global_steps)
        writer.add_scalar('test_recall', recall, global_steps)
        writer.add_scalar('test_precision', precision, global_steps)
        writer.add_scalar('test_ba', ba, global_steps)
        writer_dict['test_global_steps'] = global_steps + 1

    return loss_per_word, accuracy,recall, precision,ba




