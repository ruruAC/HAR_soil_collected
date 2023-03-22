import os
import sys

from zmq import device
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
sys.path.append("/home/zhangjunru/newwork/Trans_szs/dltime-torch-dev/dltime-torch-dev")
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import pandas as pd
from test_program.config import TrainConfig
from dltime.data.ts_datasets import Soil_Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from .data_process import handle_dataset_3dims
from .utils import get_logger, get_scheduler, load_pkl, weight_init
def get_soil_data(batch_size):
    # outputs dir
    OUTPUT_DIR = './soil_outputs/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CFG = TrainConfig()

    now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')
    model_name = 'inception-attn'
    mode = 'all'
    pretrain = False

    LOGGER = get_logger(OUTPUT_DIR + model_name + '_' + now)
    LOGGER.info(f'========= Training =========')

    # data process
    data_for_train = ['zwy', 'zwy2', 'zwy3', 'zwy4', 'j11', 'j11_2', 'j11_md', 'j11_527', 'yqcc', 'yqcc2', 'sky', 'syf', 'syf2', 'zyq', 'zyq2']
    data_for_train = ['zwy', 'zwy2', 'zwy3', 'zwy4']
    # data_for_train = ['sky']
    train_data = []
    test_data = []
    path = '/home/zhangjunru/newwork/Trans_szs/dltime-torch-dev/dltime-torch-dev/soil_disturbance/dataset//'
    for data_name in data_for_train:
        try:
            train_data.extend(load_pkl(path+data_name+'_train'+'.pkl'))
            test_data.extend(load_pkl(path+data_name+'_test'+'.pkl'))
        except:
            continue

    train_data = shuffle(train_data)
    train_x, train_label = handle_dataset_3dims(train_data, mode=mode)
    test_x, test_label = handle_dataset_3dims(test_data, mode=mode)
    train_x = np.swapaxes(train_x, 2, 1)
    test_x = np.swapaxes(test_x, 2, 1)
    print('Train data size:', train_x.shape, train_label.shape)
    print('Test data size:', test_x.shape, test_label.shape)

    train_dataset = Soil_Dataset(train_x, train_label, normalize=None, channel_first=True)
    test_dataset = Soil_Dataset(test_x, test_label, normalize=None, channel_first=True)

    feat_dim = train_x.shape[-1]
    max_len = train_dataset.max_len
    num_classes = 3
    batch_size =len(train_label)
    batch_size_test = len(test_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader, feat_dim,num_classes,max_len
