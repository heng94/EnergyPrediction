import os
import torch
import argparse
import datetime
import shutil
import wandb
import random
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets.dataset import TimeSerialDataset
from torch.nn.utils import clip_grad_norm_
from models.DeepAR import DeepAR


def main(args):
    
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = False
    
    # load data
    train_dataset = TimeSerialDataset(args, split='train')
    feature_scaler, target_scaler = train_dataset.get_scaler()
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train.batch_size, 
        shuffle=True, 
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    logger.info(f'Number of training samples: {len(train_dataloader)}')
    
    val_dataset = TimeSerialDataset(args, split='val', feature_scaler=feature_scaler, target_scaler=target_scaler)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.val.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )
    logger.info(f'Number of validation samples: {len(val_dataloader)}')
    
    test_dataset = TimeSerialDataset(args, split='test', feature_scaler=feature_scaler, target_scaler=target_scaler)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.test.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )
    logger.info(f'Number of test samples: {len(test_dataloader)}')
    
    # load model
    device = torch.device(f'cuda:{args.train.gpu_index}' if torch.cuda.is_available() else 'cpu')
    model = DeepAR(args, device=device).to(device)
    loss_fn = model.loss_fn
    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.train.lr, 
        weight_decay=float(args.train.weight_decay)
    )
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.train.T_max, 
        eta_min=args.train.eta_min
    )
    
    best_val_r2 = -1000
    
    # train loop 
    for epoch in range(args.train.num_epochs):
        train_loss = 0
        model.train()
        for data, label in train_dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.train.gradient_clip_val)
            optimizer.step()
            train_loss += loss.item()
            wandb.log({'train_loss_step': loss.item()})
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train, validation and test for energy prediction')
    parser.add_argument("--cfg_file", type=str, default="./configs/deepar_time.yaml")
    
    args = OmegaConf.load(parser.parse_args().cfg_file)
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.join(args.log.log_path, args.log.project_name, args.log.run_name)
    os.makedirs(log_root_path, exist_ok=True)
    
    log_path = os.path.join(log_root_path, time_stamp)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, 'log.txt')
    logger.add(log_file, format="{time} {level} {message}", level="INFO")
    
    # copy config file to log dir
    shutil.copy(args.cfg_file, log_path)
    
    result_root_path = os.path.join(args.results.results_path, args.log.project_name, args.log.run_name)
    os.makedirs(result_root_path, exist_ok=True)
    result_path = os.path.join(result_root_path, time_stamp)
    os.makedirs(result_path, exist_ok=True)
    
    # init wandb
    wandb.init(name=args.log.run_name, project=args.log.project_name, sync_tensorboard=True, dir=log_path)
    
    main(args)