import os
import torch
import wandb
import datetime
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from  models.build import build_model
from datasets.data_factory import data_provider
from utils.util import EarlyStopping, metric
from torch import optim
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self._set_seeds()
        self.device = self._acquire_device()
        self.log_path, self.logger, self.wandb = self._set_logger()
        self.model = self._build_model().to(self.device)
        self.optimimer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.f_dim = -1 if self.args.data.features == 'MS' else 0

    def _set_seeds(self):
        np.random.seed(self.args.train.seed)
        torch.manual_seed(self.args.train.seed)
        torch.cuda.manual_seed(self.args.train.seed)
        torch.cuda.manual_seed_all(self.args.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _set_logger(self):
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root_path = os.path.join('./logs', self.args.mdoel.name, self.args.logger.run_name)
        os.makedirs(log_root_path, exist_ok=True)
        log_path = os.path.join(log_root_path, time_stamp)
        os.makedirs(log_path, exist_ok=True)
        logger.add(os.path.join(log_path, 'train.log'), format="{time} {level} {message}", level="INFO")   

        #* save the config file to the log path
        with open(os.path.join(log_path, f'config_{self.args.logger.run_name}.yaml'), 'w') as f:
            yaml.dump(self.args, f)

        #* set up wandb
        wandb.init(name=self.args.logger.run_name, project=self.args.logger.project_name, config=self.args, dir=log_path)
        return log_path, logger, wandb

    def _build_model(self):
        model = build_model(self.args)
        if self.args.train.use_multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model
    
    def _get_data(self, flag):
        dataset, data_loader = data_provider(self.args, flag)
        return dataset, data_loader

    def _acquire_device(self):
        if self.args.train.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.train.gpu_index) if not self.args.train.use_multi_gpu else self.args.train.devices
            device = torch.device(f'cuda:{self.args.train.gpu_index}')
            print(f'Use GPU: cuda:{self.args.train.gpu_index}')
        else:
            device = torch.device('cpu')
            print(f'Use CPU')
        return device
    
    def _select_optimizer(self):
        if self.args.optimizer.name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=float(self.args.optimizer.learning_rate),
                weight_decay=float(self.args.optimizer.weight_decay),
            )
        elif self.args.optimizer.name == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=float(self.args.optimizer.learning_rate),
                weight_decay=float(self.args.optimizer.weight_decay),
            )
        elif self.args.optimizer.name == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=float(self.args.optimizer.learning_rate),
                momentum=float(self.args.optimizer.momentum),
                weight_decay=float(self.args.optimizer.weight_decay),
            )
        else:
            raise ValueError('Optimizer not supported')
        return optimizer

    def _select_scheduler(self):
        if self.args.scheduler.name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.model.parameters(), 
                T_max=int(self.args.scheduler.T_max), 
                eta_min=float(self.args.scheduler.eta_min),
            )
        elif self.args.scheduler.name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.model.parameters(), 
                milestones=self.args.scheduler.milestones, 
                gamma=float(self.args.scheduler.gamma),
            )
        return scheduler

    def exponential_warmup(self, epoch):
        warmup_epochs = self.args.train.warmup_epochs
        if epoch < warmup_epochs:
            warmup_lr = (self.args.optimizer.learning_rate / self.optimizer.defaults['lr']) \
                ** (epoch / warmup_epochs) * self.optimizer.defaults['lr']
            return warmup_lr / self.optimizer.defaults['lr']
        return self.args.optimizer.learning_rate

    def _warmup(self, epoch):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.exponential_warmup(epoch))

    def _select_criterion(self):
        criterion = torch.nn.MSELoss().to(self.device)
        return criterion

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(args=self.args, logger=self.logger, save_path=self.log_path)
        
        if self.args.train.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
        self.logger.info(f'Training starts...')
        for epoch in range(self.args.train.max_epochs):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_actuals = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.optimimer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.train.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)
                        loss = self.criterion(
                            outputs[:, -self.args.data.pred_len:, self.f_dim:], 
                            batch_y[:, -self.args.data.pred_len:, self.f_dim:],
                        )
                    grad_scaler.scale(loss).backward()
                    clip_grad_norm_(self.model.parameters(), self.args.train.gradient_clip_val)
                    grad_scaler.step(self.optimimer)
                    grad_scaler.update()
                else:
                    if self.args.model.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)
                    loss = self.criterion(
                        outputs[:, -self.args.data.pred_len:, self.f_dim:],
                        batch_y[:, -self.args.data.pred_len:, self.f_dim:],
                    )
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.args.train.gradient_clip_val)
                    self.optimimer.step()
                #* calculate the loss
                train_loss.append(loss.item())
                #* calculate the metrcis during training
                pred_outs = outputs[:, -self.args.data.pred_len:, :].detach().cpu().numpy()
                norm_outs = train_data.inverse_transform(pred_outs.reshape(-1, pred_outs.shape[-1])).reshape(-1, pred_outs.shape[-1])
                train_preds.append(norm_outs[:, self.f_dim:])
                act = batch_y[:, -self.args.data.pred_len:, :].detach().cpu().numpy()
                act_norm = train_data.inverse_transform(act.reshape(-1, act.shape[-1])).reshape(-1, act.shape[-1])
                train_actuals.append(act_norm[:, self.f_dim:])
                if (i + 1) % 100 == 0:
                    self.logger.info(
                        f'\t Epoch: {epoch + 1}/{self.args.train.max_epochs}, Iter: {i + 1}/{train_steps}, Loss: {loss.item()}'
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train.max_epochs - epoch) * train_steps - i)
                    self.logger.info(f'\t Speed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            #* scheduler
            if epoch <= self.args.scheduler.warmup_epochs:
                scheduler = self._warmup(epoch)
            else:
                scheduler = self._select_scheduler()
            scheduler.step()
            self.wandb.log({'learning_rate': self.optimimer.param_groups[0]['lr'], 'epoch': epoch+1})

            train_mae, train_mse, train_rmse, train_mape, train_mspe = metric(
                np.concatenate(np.array(train_preds), axis=0),
                np.concatenate(np.array(train_actuals), axis=0),
            )
            train_loss = np.average(train_loss)
            self.wandb.log({
                'train_loss_epoch': train_loss,
                'train_MAE_epoch': train_mae,
                'train_MSE_epoch': train_mse,
                'train_RMSE_epoch': train_rmse,
                'train_MAPE_epoch': train_mape,
                'train_MSPE_epoch': train_mspe,
                'epoch': epoch,
            })
            self.logger.info(f"Epoch: {epoch} cost time: {time.time() - epoch_time}")

            #* validation
            vali_loss, val_preds, val_trues = self.val(val_data, val_loader)
            val_mae, val_mse, val_rmse, val_mape, val_mspe = metric(val_preds, val_trues)
            self.wandb.log({
                'val_loss_epoch': vali_loss,
                'val_MAE_epoch': val_mae,
                'val_MSE_epoch': val_mse,
                'val_RMSE_epoch': val_rmse,
                'val_MAPE_epoch': val_mape,
                'val_MSPE_epoch': val_mspe,
                'epoch': epoch,
            })
            self.logger.info(f'Epoch: {epoch + 1}/{self.args.train.max_epochs}, Train Loss: {train_loss}, Val Loss: {vali_loss}')

            #* early stopping
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info('Early stopping')
                break
            
        self.logger.info(f'Training finished, best val loss: {early_stopping.val_loss_min}')
        
        #* test
        self.logger.info(f'Testing...')
        best_model_path = os.path.join(self.log_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        test_data, test_loader = self._get_data(flag='test')
        test_loss, test_preds, test_trues = self.val(test_data, test_loader)
        test_mae, test_mse, test_rmse, test_mape, test_mspe = metric(test_preds, test_trues)
        self.logger.info(
            f'''
            Test Loss: {test_loss}, 
            Test MAE: {test_mae}, 
            Test MSE: {test_mse}, 
            Test RMSE: {test_rmse}, 
            Test MAPE: {test_mape}, 
            Test MSPE: {test_mspe},
            '''
        )
        self.wandb.log({
            'test_loss': test_loss,
            'test_MAE': test_mae,
            'test_MSE': test_mse,
            'test_RMSE': test_rmse,
            'test_MAPE': test_mape,
            'test_MSPE': test_mspe,
        })

        #* non-overlapping prediction
        self.logger.info(f'Non-overlapping prediction...')
        self.predict()

    @torch.no_grad()
    def val(self, val_data, val_loader):
        loss_list, preds, trues = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.train.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)
                        loss = self.criterion(
                            outputs[:, -self.args.data.pred_len:, self.f_dim:], 
                            batch_y[:, -self.args.data.pred_len:, self.f_dim:],
                        )
                else:
                    if self.args.model.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)
                    loss = self.criterion(
                        outputs[:, -self.args.data.pred_len:, self.f_dim:],
                        batch_y[:, -self.args.data.pred_len:, self.f_dim:],
                    )
                loss_list.append(loss.item())
                pred_outs = outputs[:, -self.args.data.pred_len:, :].detach().cpu().numpy()
                norm_outs = val_data.inverse_transform(pred_outs.reshape(-1, pred_outs.shape[-1])).reshape(-1, pred_outs.shape[-1])
                preds.append(norm_outs[:, self.f_dim:])
                act = batch_y[:, -self.args.data.pred_len:, :].detach().cpu().numpy()
                act_norm = val_data.inverse_transform(act.reshape(-1, act.shape[-1])).reshape(-1, act.shape[-1])
                trues.append(act_norm[:, self.f_dim:])
        return loss_list, np.concatenate(np.array(preds), axis=0), np.concatenate(np.array(trues), axis=0)

    @torch.no_grad()
    def predict(self):
        test_data, test_loader = self._get_data(flag='test')
        df = pd.read_csv(os.path.join(self.args.data.root_path, self.args.data.data_path))
        prediction_data = df.tail(
            int(len(df) * (1 - self.args.data.train_scale - self.args.data.val_scale))
        ).to_numpy()[:, 1:].astype(np.float32)
        predictions, actuals = [], []
        
        with torch.no_grad():
            for i in range(0, prediction_data.shape[0] - self.args.seq_len, self.args.pred_len):
                per_data = test_data.scaler.transform(prediction_data[i:i + self.args.seq_len, :])
                per_input = torch.from_numpy(per_data.reshape(1, self.args.data.seq_len, -1)).to(self.device)
                if self.args.train.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model.output_attention:
                            output = self.model(per_input)[0]
                        else:
                            output = self.model(per_input)
                else:
                    if self.args.model.output_attention:
                        output = self.model(per_input)[0]
                    else:
                        output = self.model(per_input)
                output = output[:, -self.args.data.pred_len:, :].cpu().numpy()
                shape = output.shape
                output = test_data.inverse_transform(output.squeeze(0)).reshape(shape)
                output = output[:, :, self.f_dim:].reshape(-1)
                predictions.append(output)
                actual = prediction_data[i + self.args.data.seq_len: i + self.args.data.seq_len + self.args.data.pred_len, -1].reshape(-1)
                actuals.append(actual)
        predictions = np.concatenate(np.array(predictions), axis=0)
        actuals = np.concatenate(np.array(actuals), axis=0)
        pred_mae, pred_mse, pred_rmse, pred_mape, pred_mspe = metric(predictions, actuals)
        self.wandb.log({'pred_MAE': pred_mae, 'pred_MSE': pred_mse, 'pred_RMSE': pred_rmse, 'pred_MAPE': pred_mape, 'pred_MSPE': pred_mspe})
        self.logger.info(f'Pred_MAE:{pred_mae}, Pred_MSE:{pred_mse}, Pred_RMSE:{pred_rmse}, Pred_MAPE:{pred_mape}, Pred_MSP:{pred_mspe}')
        
        test_results = pd.DataFrame({'actual': actuals.reshape(-1), 'pred': predictions})
        test_results.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)
        
        time_idx = np.arange(0, actuals.shape[0], 1)
        plt.plot(time_idx, actuals, label='actual')
        plt.plot(time_idx, predictions, label='prediction')
        plt.legend()
        plt.savefig(os.path.join(self.log_path, 'result.png'))
        plt.close()
        wandb.log({'result': wandb.Image(os.path.join(self.log_path, 'result.png'))})