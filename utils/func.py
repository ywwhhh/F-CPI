import logging
import os
import time


def set_logger(output_dir = None,rank = -1):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank < 1 else logging.WARNING)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank < 1 else logging.WARNING)
    logger.addHandler(console)

    if output_dir is not None:
        fileter = logging.FileHandler(output_dir + '/log.txt')
        fileter.setLevel(logging.INFO if rank < 1 else logging.WARNING)
        logger.addHandler(fileter)

    return logger


def get_output_dir(model_name, data_pre_func, data_root, criterion_name_1, criterion_name_2=None,rank=-1):
    if criterion_name_2 == None:
        output_dir = 'experiment/' + model_name + '_' + data_pre_func + '_' + data_root + '_' + criterion_name_1
    else:
        output_dir = 'experiment/' + model_name + '_' + data_pre_func + '_' + data_root + '_' + criterion_name_1 + '_' + criterion_name_2
    if rank<1:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    time_dir = output_dir + '/' + time_str
    if rank<1:
        if not os.path.exists(time_dir):
            os.mkdir(time_dir)
    return time_dir