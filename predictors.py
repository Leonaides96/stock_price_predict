import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # To checking the machine are obtain GPU or CPU

class PredictionModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = DEVICE)

        out, (hn, cn) = self.lstm(x, (h0.detach()), c0.detach())

        out = self.fc(out[:, -1, :])
        return out

