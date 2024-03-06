import torch.nn as nn


class MultivariableLSTM(nn.Module):
    def __init__(self, args, device):
        super(MultivariableLSTM, self).__init__()
        self.args = args
        self.device = device
        self.lstm = nn.LSTM(
            input_size=self.args.model.input_size, 
            hidden_size=self.args.model.hidden_size, 
            num_layers=self.args.model.num_layers, 
            dropout=self.args.model.dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(
            in_features=self.args.model.hidden_size, 
            out_features=self.args.data.future_steps
        )
        self.relu = nn.ReLU()
    
    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        #* [batch_size, future_steps]
        linear_input = out[:, -1, :].view(-1, self.args.model.hidden_size).contiguous()
        out = self.relu(self.linear(linear_input))
        return out
