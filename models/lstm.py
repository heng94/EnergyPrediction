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
        self.linear1 = nn.Linear(
            in_features=self.args.model.hidden_size, 
            out_features=self.args.data.future_steps*2,
        )
        self.linear2 = nn.Linear(
            in_features=self.args.data.future_steps*2, 
            out_features=self.args.data.future_steps
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # self.ln1 = nn.LayerNorm(self.args.model.hidden_size)
        # self.ln2 = nn.LayerNorm(self.args.data.future_steps*2)
        # self.ln3 = nn.LayerNorm(self.args.data.future_steps)
    
    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        #* [batch_size, future_steps]
        linear_input = out[:, -1, :].view(-1, self.args.model.hidden_size).contiguous()
        out = self.relu1(self.linear1(linear_input))
        out = self.relu2(self.linear2(out))
        # out = self.relu1(self.linear1(self.ln1(linear_input)))
        # out = self.relu2(self.linear2(out))
        return out
