import imp
import sys
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import re
import string
import pickle

def get_logger(filename='train'):
    
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    sh_handler = StreamHandler(stream=sys.stdout)
    sh_handler.setFormatter(Formatter("%(message)s"))
    fh_handler = FileHandler(filename=f"{filename}.log")
    fh_handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(sh_handler)
    logger.addHandler(fh_handler)
    return logger

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    # param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.transformer_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.transformer_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "transformer_encoder" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

def load_pretrained_state_dict(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "output_layer" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
