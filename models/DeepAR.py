import torch 
import torch.nn as nn
import torch.nn.functional as F


class DeepAR(nn.Module):
    
    def __init__(self, args, device):
        
        '''
        If output_size = 1, then this model achieves single point prediction,
        otherwise, this model is used for probability prediction.
        '''
        super(DeepAR, self).__init__()

        self.args = args
        self.device = device
        
        self.lstm = nn.LSTM(
            input_size=self.args.model.input_size, 
            hidden_size=self.args.model.hidden_size, 
            num_layers=self.args.model.num_layers, 
            dropout=self.args.model.dropout,
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=self.args.model.hidden_size,
            out_features=self.args.model.output_size * self.args.data.future_steps
        )
        
    def hidden_init(self,):
        
        ''' Initialize hidden state and cell state at the beginning of each epoch'''
        
        h_0 = torch.zeros(
            self.args.model.num_layers, 
            self.args.model.batch_size, 
            self.args.model.hidden_size
        ).to(self.device)
        
        c_0 = torch.zeros(
            self.args.model.num_layers, 
            self.args.model.batch_size, 
            self.args.model.hidden_size
        ).to(self.device)
        
        return h_0, c_0

    def forward(self, x):
        
        # One time step
        output, _ = self.lstm(x)  # output: (B, L, H)
        
        if self.args.train.use_all:
            
            # Use all time steps
            output = torch.mean(output, dim=1)  # output: (B, H)
        else:
            
            # Use the last time step
            output = output[:, -1, :]
            
        out = self.linear(output)  # out: (B, output_size*future_steps)
        out = out.reshape(-1, self.args.data.future_steps, self.args.model.output_size)  # out: (B, future_steps, output_size)
        mean = out[:, :, 0]  # mean: (B, future_steps)
        var = out[:, :, 1]  # var: (B, future_steps)
        
        if self.args.train.single_pred:
            
            return mean, var
        else:
            
            mean = out[:, :, 0]
            var = F.softplus(out[:, :, 1])
            
            return mean, var
        
    def loss(self, mean, var, y):
        
        ''' Calculate the loss function '''
        
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        if self.args.train.single_pred:
            
            loss = torch.mean((mean - y) ** 2)
        else:
            
            dist = torch.distributions.normal.Normal(mean, var)
            loss = - dist.log_prob(y).mean()
            
        return loss
