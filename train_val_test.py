import os
import torch
import argparse
import datetime
import shutil
import wandb
import random
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets.dataset import TimeSerialDataset
from datasets.utils import *
from torch.nn.utils import clip_grad_norm_
from models.DeepAR import DeepAR
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from matplotlib import pyplot as plt


def main(args):
    
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True
    
    # load data
    train_dataset = TimeSerialDataset(args, split='train')
    args.model.input_size = train_dataset.get_input_dim()
    assert args.model.input_size is not None
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
    loss_fn = model.loss
    
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
        eta_min=float(args.train.eta_min)
    )
    
    best_val_r2 = -1000
    
    # train-val loop 
    for epoch in range(args.train.num_epochs):
        
        # training
        train_loss = 0
        model.train()
        for data, label in train_dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            mean, var = model(data)
            loss = loss_fn(mean, var, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.train.gradient_clip_val)
            optimizer.step()
            train_loss += loss.item()
            wandb.log({'train_loss_step': loss.item()})
        lr_scheduler.step()
        wandb.log({'train_loss_epoch': train_loss / len(train_dataloader), 'epoch': epoch})
        wandb.log({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})
        logger.info(f'Epoch {epoch} train loss: {train_loss / len(train_dataloader)}')
        
        # validation
        val_loss = 0
        pred_list = []
        label_list = []
        model.eval()
        with torch.no_grad():
            for data, label in val_dataloader:
                data, label = data.to(device), label.to(device)
                mean, var = model(data)
                loss = loss_fn(mean, var, label)
                val_loss += loss.item()
                wandb.log({'val_loss_step': loss.item()})
                pred_list.append(
                    mean.cpu().numpy().reshape(-1, 1)
                )  # (batch_size * window_size, 1)
                label_list.append(
                    label.cpu().numpy().reshape(-1, 1)
                )  # (batch_size * window_size, 1)
        wandb.log({'val_loss_epoch': val_loss / len(val_dataloader), 'epoch': epoch})
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
        
        # save the best model accrording to val_r2
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(log_path, 'best_model.pth'))
            logger.info(f'Epoch {epoch} best model saved, val_r2: {val_r2}')
            
    # test
    model.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
    model.to(device)
    model.eval()

    pred_list = []
    label_list = []
    with torch.no_grad():
        for data, label in val_dataloader:
            data = data.to(device)
            mean, var = model(data)
            pred_list.append(mean.cpu().numpy().reshape(-1, 1))
            label_list.append(label.numpy().reshape(-1, 1))
        
        pred = target_scaler.inverse_transform(np.concatenate(pred_list, axis=0))
        actual = target_scaler.inverse_transform(np.concatenate(label_list, axis=0))
        test_mae = np.mean(np.abs(pred - actual))
        wandb.log({'test_mae': test_mae})
        logger.info(f'Test MAE: {test_mae}')
        
        test_rmse = np.sqrt(np.mean((pred - actual) ** 2))
        wandb.log({'test_rmse': test_rmse})
        logger.info(f'Test RMSE: {test_rmse}')
        
        test_r2 = r2_score(actual, pred)
        wandb.log({'test_r2': test_r2})
        logger.info(f'Test R2: {test_r2}')
        
        test_mape = mean_absolute_percentage_error(actual, pred)
        wandb.log({'test_mape': test_mape})
        logger.info(f'Test MAPE: {test_mape}')
        
    # predict
    model.eval()
    predict_data = pd.read_csv(args.data.file_path)
    if args.data.data_type == 'correlation':
        predict_data = predict_data[train_dataset.get_cor_index()]
    predict_data = predict_data.to_numpy().astype(np.float32)[args.data.val_cutoff:, :]
    prediction = []
    label = []
    for k in range(int((args.data.total_num - args.data.val_cutoff - args.data.window_size)/24)):
        input_data = predict_data[k * 24: k * 24 + args.data.window_size, :]
        input_data = feature_scaler.transform(input_data)
        
        if args.data.use_time:
            assert time_feature_weight.shape[-1] == input_data.shape[-1]
            input_data = input_data * time_feature_weight
            
        input_label = predict_data[k * 24 + args.data.window_size: k * 24 + args.data.window_size + args.data.future_steps, 0]
        label.append(input_label.reshape(-1, 1))
        
        input_data = torch.from_numpy(input_data.reshape(1, args.data.window_size, args.model.input_size)).to(device)
        
        with torch.no_grad():
            mean, var = model(input_data)
            pred = target_scaler.inverse_transform(mean.cpu().numpy().reshape(-1, 1))
            prediction.append(pred)
    
    prediction = np.concatenate(prediction, axis=0)
    label = np.concatenate(label, axis=0)
    test_mae = np.mean(np.abs(prediction - label))
    wandb.log({'predict_mae': test_mae})
    logger.info(f'Predict MAE: {test_mae}')
    
    test_rmse = np.sqrt(np.mean((prediction - label) ** 2))
    wandb.log({'predict_rmse': test_rmse})
    logger.info(f'Predict RMSE: {test_rmse}')
    
    test_r2 = r2_score(label, prediction)
    wandb.log({'predict_r2': test_r2})
    logger.info(f'Predict R2: {test_r2}')
    
    test_mape = mean_absolute_percentage_error(label, prediction)
    wandb.log({'predict_mape': test_mape})
    logger.info(f'Predict MAPE: {test_mape}')
    
    # save results
    time_idx = list(range(
        args.data.val_cutoff + args.data.window_size,
        args.data.total_num,
        1
    ))
    result = pd.DataFrame({
        'time_idx': time_idx,
        'actual': label.reshape(-1),
        'prediction': prediction.reshape(-1),
    })  
    result.to_csv(os.path.join(result_path, 'result.csv'), index=False)
    
    # visualize
    plt.plot(time_idx, label, label='actual')
    plt.plot(time_idx, prediction, label='prediction')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'result.png'))
    plt.close()
    wandb.log({'result': wandb.Image(os.path.join(result_path, 'result.png'))})
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train, validation and test for energy prediction')
    parser.add_argument("--cfg_file", type=str, default="./configs/deepar_original.yaml")
    
    args = OmegaConf.load(parser.parse_args().cfg_file)
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.join(args.log.log_path, args.log.project_name, args.log.run_name)
    os.makedirs(log_root_path, exist_ok=True)
    
    log_path = os.path.join(log_root_path, time_stamp)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, 'log.txt')
    logger.add(log_file, format="{time} {level} {message}", level="INFO")
    
    # copy config file to log dir
    shutil.copy(parser.parse_args().cfg_file, log_path)
    
    result_root_path = os.path.join(args.results.results_path, args.log.project_name, args.log.run_name)
    os.makedirs(result_root_path, exist_ok=True)
    result_path = os.path.join(result_root_path, time_stamp)
    os.makedirs(result_path, exist_ok=True)
    
    # init wandb
    wandb.init(name=args.log.run_name, project=args.log.project_name, sync_tensorboard=True, dir=log_path)
    
    main(args)