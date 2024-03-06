import os
import torch
import argparse
import datetime
import shutil
import wandb
import random
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets.dataset import TimeSerialDataset
from datasets.utils import *
from torch.nn.utils import clip_grad_norm_
from models.lstm import MultivariableLSTM
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def main(args):
    
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True
    
    #* Load data
    train_dataset = TimeSerialDataset(args, split='train')
    args.model.input_size = train_dataset.get_input_dim()
    assert args.model.input_size is not None
    feature_scaler, target_scaler = train_dataset.get_scaler()
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    logger.info(f'Number of training samples: {len(train_dataloader)}')
    
    val_dataset = TimeSerialDataset(
        args, split='val', 
        feature_scaler=feature_scaler, 
        target_scaler=target_scaler,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.val.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )
    logger.info(f'Number of validation samples: {len(val_dataloader)}')
    
    #* Load model
    device = torch.device(
        f'cuda:{args.train.gpu_index}' if torch.cuda.is_available() else 'cpu'
    )
    model = MultivariableLSTM(args, device=device).to(device)
    loss_fn = nn.MSELoss().to(device)
    
    #* Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.train.lr, 
        weight_decay=float(args.train.weight_decay)
    )
    
    #* Lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.train.T_max, 
        eta_min=float(args.train.eta_min)
    )
    
    best_val_r2 = 0
    
    #* Train-val loop 
    for epoch in range(args.train.num_epochs):
        
        #* Training
        model.train()
        train_loss = 0

        for train_data, train_label in train_dataloader:
            h_0 = torch.zeros(args.model.num_layers, args.train.batch_size, args.model.hidden_size).to(device)
            c_0 = torch.zeros(args.model.num_layers, args.train.batch_size, args.model.hidden_size).to(device)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            pred = model(train_data, h_0, c_0)
            loss = loss_fn(pred.reshape(-1), train_label.reshape(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), args.train.gradient_clip_val)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()
        wandb.log(
            {'train_loss_epoch': train_loss / len(train_dataloader), 
             'epoch': epoch}
        )
        wandb.log({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})
        logger.info(f'Epoch {epoch} train loss: {train_loss / len(train_dataloader)}')
        
        #* validation
        val_loss = 0
        pred_list, label_list = [], []
        model.eval()
        with torch.no_grad():
            for val_data, val_label in val_dataloader:
                h_0 = torch.zeros(args.model.num_layers, args.val.batch_size, args.model.hidden_size).to(device)
                c_0 = torch.zeros(args.model.num_layers, args.val.batch_size, args.model.hidden_size).to(device)
                val_data = val_data.to(device)
                val_label = val_label.to(device)
                pred = model(val_data, h_0, c_0)
                loss = loss_fn(pred.reshape(-1), val_label.reshape(-1))
                val_loss += loss.item()
                pred_list.append(pred.cpu().numpy().reshape(-1, 1))
                label_list.append(val_label.cpu().numpy().reshape(-1, 1))
        wandb.log(
            {'val_loss_epoch': val_loss / len(val_dataloader), 
             'epoch': epoch}
        )
        logger.info(f'Epoch {epoch} val loss: {val_loss / len(val_dataloader)}')
        
        #* Calculate metrics
        pred = target_scaler.inverse_transform(np.concatenate(pred_list, axis=0))
        actual = target_scaler.inverse_transform(np.concatenate(label_list, axis=0))
        val_mae = np.mean(np.abs(pred - actual))
        wandb.log({'val_mae': val_mae, 'epoch': epoch})
        logger.info(f'Epoch {epoch} validation MAE: {val_mae}')
        
        val_rmse = np.sqrt(np.mean((pred - actual) ** 2))
        wandb.log({'val_rmse': val_rmse, 'epoch': epoch})
        logger.info(f'Epoch {epoch} validation RMSE: {val_rmse}')
        
        val_r2 = r2_score(actual, pred)
        wandb.log({'val_r2': val_r2, 'epoch': epoch})
        logger.info(f'Epoch {epoch} validation R2: {val_r2}')
        
        val_mape = mean_absolute_percentage_error(actual, pred)
        wandb.log({'val_mape': val_mape, 'epoch': epoch})
        logger.info(f'Epoch {epoch} validation MAPE: {val_mape}')
        
        #* save the best model accrording to val_mape
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(log_path, 'best_model.pth'))
            logger.info(f'Epoch {epoch} best model saved, val_r2: {val_r2}')
   
    #* Test the best model
    model.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
    model.eval()
    pred_list, label_list = [], []
    
    #* make non-overlapping predictions
    df = pd.read_csv(args.data.file_path)
    test_data = df[args.data.val_cutoff:].to_numpy().astype(np.float32)
    with torch.no_grad():
        for i in range(0, test_data.shape[0] - args.data.window_size, args.data.future_steps):
            per_data = feature_scaler.transform(
                test_data[i: i + args.data.window_size, 1:]
            )
            per_label_norm = target_scaler.transform(
                test_data[i: i + args.data.window_size, 0].reshape(-1, 1)
            ).reshape(-1, 1)
            per_input = np.concatenate([per_data, per_label_norm], axis=1)
            per_input = torch.from_numpy(per_input.reshape(1, args.data.window_size, -1)).to(device)
            h_0 = torch.zeros(args.model.num_layers, 1, args.model.hidden_size).to(device)
            c_0 = torch.zeros(args.model.num_layers, 1, args.model.hidden_size).to(device)
            pred = model(per_input, h_0, c_0)
            pred_list.append(pred.cpu().numpy().reshape(-1, 1))
            label_list.append(
                test_data[
                    i + args.data.window_size: i + args.data.window_size + args.data.future_steps, 0
                ].reshape(-1, 1)
            )
    test_pred = target_scaler.inverse_transform(np.concatenate(pred_list, axis=0)).reshape(-1)
    test_actual = np.concatenate(label_list, axis=0).reshape(-1)
    test_mae = np.mean(np.abs(test_pred - test_actual))
    wandb.log({'test_mae': test_mae})
    logger.info(f'Test MAE: {test_mae}')
    
    test_rmse = np.sqrt(np.mean((test_pred - test_actual) ** 2))
    wandb.log({'test_rmse': test_rmse})
    logger.info(f'Test RMSE: {test_rmse}')
    
    test_r2 = r2_score(test_actual, test_pred)
    wandb.log({'test_r2': test_r2})
    logger.info(f'Test R2: {test_r2}')
    
    test_mape = mean_absolute_percentage_error(test_actual, test_pred)
    wandb.log({'test_mape': test_mape})
    logger.info(f'Test MAPE: {test_mape}')
    
    #* Save results
    time_idx = np.arange(args.data.val_cutoff + args.data.window_size, args.data.total_num)
    test_results = pd.DataFrame(
        {
            'time_idx': time_idx,
            'actual': test_actual, 
            'pred': test_pred
        }
    )
    test_results.to_csv(os.path.join(result_path, 'result.csv'), index=False)
    
    #* visualize
    plt.plot(time_idx, test_actual, label='actual')
    plt.plot(time_idx, test_pred, label='prediction')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'result.png'))
    plt.close()
    wandb.log({'result': wandb.Image(os.path.join(result_path, 'result.png'))})
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train, validation and test for energy prediction'
    )
    parser.add_argument(
        "--cfg_file", 
        type=str, 
        default="./configs/deepar_original.yaml"
    )
    args = OmegaConf.load(parser.parse_args().cfg_file)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.join(
        args.log.log_path, 
        args.log.project_name, 
        args.log.run_name
    )
    os.makedirs(log_root_path, exist_ok=True)
    
    log_path = os.path.join(log_root_path, time_stamp)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, 'log.txt')
    logger.add(log_file, format="{time} {level} {message}", level="INFO")
    
    # copy config file to log dir
    shutil.copy(parser.parse_args().cfg_file, log_path)
    
    result_root_path = os.path.join(
        args.results.results_path, 
        args.log.project_name, 
        args.log.run_name
    )
    os.makedirs(result_root_path, exist_ok=True)
    result_path = os.path.join(result_root_path, time_stamp)
    os.makedirs(result_path, exist_ok=True)
    
    # init wandb
    wandb.init(
        name=args.log.run_name, 
        project=args.log.project_name, 
        sync_tensorboard=True, 
        dir=log_path
    )
    
    main(args)