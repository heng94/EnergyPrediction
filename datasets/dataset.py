import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from .utils import *


class TimeSerialDataset(Dataset):
    def __init__(self, args, split='train', target_scaler=None, feature_scaler=None):
        self.args = args
        self.split = split
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.data, self.label = self.create_data()
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, index):
        if self.args.data.use_time:
            assert time_feature_weight.shape[-1] == self.data[index].shape[-1]
            self.data[index] = self.data[index] * time_feature_weight
        return self.data[index], self.label[index]
    
    def get_scaler(self,):
        return self.feature_scaler, self.target_scaler
    
    def get_input_dim(self,):
        return self.data.shape[-1]
        
    def create_data(self,):
        df = pd.read_csv(self.args.data.file_path)
        df = df.to_numpy().astype(np.float32)
        raw_data = self.data_split(df) 
        data, target = self.load_data(raw_data)
        return data, target
        
    def data_split(self, df):
        if self.split == 'train':
            raw_data = df[: self.args.data.train_cutoff]
        elif self.split == 'val':
            raw_data = df[self.args.data.train_cutoff: self.args.data.val_cutoff]
        else:
            raw_data = df[self.args.data.val_cutoff:]
        return raw_data
        
    def load_data(self, raw_data: np.ndarray):
        if self.feature_scaler is None and self.target_scaler is None and self.split == 'train':
            self.target_scaler = MinMaxScaler()
            self.feature_scaler = MinMaxScaler()
        #* [num_samples, 1]
        target_norm = self.target_scaler.fit_transform(raw_data[:, 0].reshape(-1, 1))
        #* [num_samples, num_features]
        data_norm = self.feature_scaler.fit_transform(raw_data[:, 1:])  
            
        data_list, target_list = [], []
        for idx in range(
            0, 
            data_norm.shape[0] - self.args.data.window_size - self.args.data.future_steps + 1, 
            self.args.data.shift
        ):
            #* [window_size, num_features + 1]
            data_list.append(
                np.concatenate(
                    [data_norm[idx: idx + self.args.data.window_size, :],
                    target_norm[idx: idx + self.args.data.window_size, :].reshape(-1, 1)],
                    axis=1
                )
            )
            #* [future_steps, 1]
            target_list.append(
                target_norm[
                    idx + self.args.data.window_size: \
                        idx + self.args.data.window_size + self.args.data.future_steps, :
                ].reshape(-1, 1)
            )
        data, target = np.array(data_list), np.array(target_list)
        data = torch.from_numpy(data.reshape(-1, self.args.data.window_size, data_norm.shape[1] + 1))
        target = torch.from_numpy(target.reshape(-1, self.args.data.future_steps, target_norm.shape[1]))
        return data, target