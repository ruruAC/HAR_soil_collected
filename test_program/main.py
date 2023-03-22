import os
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib  
matplotlib.use('Agg')
from zmq import device
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
sys.path.append('E://jupter//A-JRZ//UCR3//Trans_szs//dltime-torch-dev//dltime-torch-dev')

import torch
import torch.nn as nn
import time
import datetime
import wandb
import gc
import pandas as pd
from train_helper import train_fn, valid_fn
from config import TrainConfig
from utils import get_logger, get_optimizer_params, get_scheduler, load_pretrained_state_dict
from dltime.data.ts_datasets import UCR_UEADataset
from dltime.data.tsc_dataset_names import *
from dltime.models.ts_transformer import TSTransformerEncoderConvClassifier
from dltime.models.GTN import GTN
from transformers import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = './outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CFG = TrainConfig()
CFG.extract_path = "E://jupter//A-JRZ//UCR3//TCE//datasets//Multivariate2018_ts//Multivariate_ts"

now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')
model_name = 'GTN'
LOGGER = get_logger(OUTPUT_DIR + now)
LOGGER.info(f'========= Training =========')

multivariate_dataset = ["ArticularyWordRecognition","EthanolConcentration","PEMS-SF","AtrialFibrillation", "JapaneseVowels","AtrialFibrillation", "BasicMotions", \
    "CharacterTrajectories", "FaceDetection", "HandMovementDirection", "Heartbeat", "NATOPS", "SpokenArabicDigits"]
multivariate_dataset = ["DuckDuckGeese","Epilepsy"]
multivariate_dataset = ["DuckDuckGeese"]
# univariate_dataset = ["CricketX", "ECG200", "Wafer"]

# for dataset_name in multivariate_dataset[:2]:
for dataset_name in multivariate_dataset:
    
    # try:
    LOGGER.info(f"Train for dataset {dataset_name}")
    
    train_dataset = UCR_UEADataset(dataset_name, split="train", extract_path=CFG.extract_path)
    test_dataset = UCR_UEADataset(dataset_name, split="test", extract_path=CFG.extract_path)

    feat_dim = train_dataset[0]['input'].shape[-1]
    max_len = train_dataset.max_len
    num_classes = len(train_dataset.y2label)

    LOGGER.info(f"Train for Dataset {dataset_name}, feat_dim: {feat_dim}, max_len: {max_len}, num_class: {num_classes}")
    LOGGER.info(f"Train Size: {len(train_dataset)} Test Size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    test_labels = test_dataset.y
    colors=['#699d4c','#aa2704','#107ab0','red','blue'] #绿色，红色，蓝色
    # except:
    #     LOGGER.info(f"Load Dataset error!")
    #     continue
    dir_name = "D://pictures//"+dataset_name
    import os
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if 1:
        sp= 2 #特征 train_dataset[0]['input'].shape[-1]
        #[len,channel]
        for c in range(len(train_dataset)):
            plt.figure(figsize=(20,10))
            for sp in range(train_dataset[c]['input'].shape[1],train_dataset[c]['input'].shape[1] - 5, -1): #特征
                sp = train_dataset[c]['input'].shape[1] - sp
          #  for sp in range(min(train_dataset[c]['input'].shape[1],5)): #特征

                plt.plot(np.arange(train_dataset[c]['input'].shape[0]),train_dataset[c]['input'][:,sp].numpy(),# 线条的颜色
               # color=colors[int(train_dataset[c]['label'].item())],  
               color=colors[sp],  
                        linewidth=2.0,  
                        linestyle='-' ,
                       #label='Class '+str(train_dataset[c]['label'].item())  
                       label='F'+str(sp)         
                    )
            plt.legend()
            plt.title(str(train_dataset[c]['label'].item()))
            #plt.show()
            plt.savefig("D://pictures//"+dataset_name+"//"+str(train_dataset[c]['label'].item())+"&"+str(c)+".png")
        print(train_dataset[c]['label'].item())
        print("ff")

    if CFG.wandb:
        # 启用 weights & bias
        import wandb
        wandb.login()
        anony = None

        def class2dict(f):
            return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

        run = wandb.init(project='dltime', 
                        name=CFG.project,
                        config=class2dict(CFG),
                        group=dataset_name + '_train',
                        job_type="train",
                        anonymous=anony)

    # model = TSTransformerEncoderConvClassifier(
    #     feat_dim=feat_dim, 
    #     max_len=max_len, 
    #     d_model=256, 
    #     num_layers=4, 
    #     dim_feedforward=512, 
    #     num_classes=num_classes).to(CFG.device)
    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    N = 8
    dropout = 0.2
    pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
    mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
    d_input = max_len  # 时间部数量
    d_channel = feat_dim  # 时间序列维度
    d_output = num_classes  # 分类类别
    model = GTN(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=CFG.device).to(CFG.device)

    optimizer_parameters = model.parameters()
    # optimizer_parameters = get_optimizer_params(model, CFG.encoder_lr, CFG.decoder_lr)
   # optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    optimizer=optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-3)  
#Adamw 即 Adam + weight decate ,效果与 Adam + L2正则化相同,但是计算效率更高,因为L2正则化需要在loss中加入正则项,之后再算梯度,最后在反向传播,而Adamw直接将正则项的梯度加入反向传播的公式中,省去了手动在loss中加正则项这一步
    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss, avg_acc = train_fn(CFG, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = valid_fn(CFG, test_loader, model, criterion, CFG.device)
        
        # scoring
        score = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='macro')

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Acc: {score:.4f} - F1: {f1:.4f}')
        if CFG.wandb:
            wandb.log({"epoch": epoch+1, 
                    "avg_train_loss": avg_loss, 
                    "avg_train_acc": avg_acc,
                    "avg_val_loss": avg_val_loss,
                    "score": score})
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR+f"{dataset_name}_{model_name}_best_retrain.pth")

    torch.cuda.empty_cache()
    gc.collect()
    if CFG.wandb:
        wandb.finish()
