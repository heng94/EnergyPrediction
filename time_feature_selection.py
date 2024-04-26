import os
import torch
import random
import argparse
import datetime
import pandas as pd
from omegaconf import OmegaConf
from Projects.EnergyPrediction.datasets.data_loaders import TimeSerialDataset
from datasets.utils import *
from models.lstm import MultivariableLSTM
from anamod import TemporalModelAnalyzer


def main(args):
    
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True
    
    #* Load data
    df = pd.read_csv(args.data.file_path)
    feature_name_list = df.columns.tolist()
    train_dataset = TimeSerialDataset(args, split='train')
    feature_scaler, target_scaler = train_dataset.get_scaler()
    data = df[args.data.val_cutoff:].to_numpy().astype(np.float32)
    input_data, input_label = [], []
    for i in range(0, data.shape[0] - args.data.window_size, args.data.future_steps):
        per_data = feature_scaler.transform(data[i: i + args.data.window_size, 1:])
        per_label_norm = target_scaler.transform(
            data[i: i + args.data.window_size, 0].reshape(-1, 1)
        ).reshape(-1, 1)
        per_input = np.concatenate([per_data, per_label_norm], axis=1)
        input_data.append(per_input)
        per_label = np.array(
            data[
                i + args.data.window_size: i + args.data.window_size + args.data.future_steps, 0
            ].reshape(-1, 1)
        )
        # per_label = target_scaler.transform(
        #     data[
        #         i + args.data.window_size: i + args.data.window_size + args.data.future_steps, 0
        #     ].reshape(-1, 1)
        # ).reshape(-1)
        input_label.append(per_label)

    #* [num_samples, seq_len, input_size]
    input_data = np.array(input_data)
    args.model.input_size = input_data.shape[-1]
    #* [num_samples * future_steps]
    input_label = np.array(input_label)
    
    #* Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultivariableLSTM(args, device)
    model.load_state_dict(torch.load(args.model.best_model_path))
    model.to(device)
    model.eval()
    
    analyzer = TemporalModelAnalyzer(
        model=model, 
        data=np.transpose(input_data, (0, 2, 1)),  #* [num_samples, input_size, seq_len]
        targets=input_label.reshape(-1, 1),
        feature_names=feature_name_list,
        num_permutations=args.anamod.num_permutations, 
        scaler=target_scaler,
        # seed=args.train.seed,
        importance_significance_level=args.anamod.importance_significance_level,
        # loss_function="absolute_difference_loss", quadratic_loss
        loss_function="quadratic_loss",
        output_dir=log_path,
        args=args,
    )
    
    features = analyzer.analyze()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train, validation and test for energy prediction')
    parser.add_argument("--cfg_file", type=str, default="./configs/lstm_time.yaml")
    args = OmegaConf.load(parser.parse_args().cfg_file)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.join(args.log.log_path, args.log.project_name, args.log.run_name)
    os.makedirs(log_root_path, exist_ok=True)
    
    log_path = os.path.join(log_root_path, time_stamp)
    os.makedirs(log_path, exist_ok=True)
    
    main(args)