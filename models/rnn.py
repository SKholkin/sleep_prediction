from torch import nn
import torch
from torch.nn.modules.activation import ReLU
from models.layernorm_lstm import LayerNormLSTM, LSTM
import math

class RNN(nn.Module):
    def __init__(self, hidden_size=16, num_lstm_layers=1, classifier_output=False) -> None:
        super().__init__()
        self.classifier_output = classifier_output
        self.hidden_size = hidden_size
        self.output_dim = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.preprocess_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=hidden_size), nn.ReLU())
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.lstm = LayerNormLSTM(input_size=hidden_size, hidden_size=hidden_size)
        if self.classifier_output:
            self.classifier = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=1), nn.Sigmoid())
        self.windows_size = 5
        self.cls_conv = nn.Sequential(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.windows_size, stride=self.windows_size), nn.ReLU())

    def forward(self, x):
        # x: (N, L, 1)
        # lstm inputs: (L, H, N)
        batch_size = x.size(0)
        x = self.preprocess_mlp(x)
        x = self.cls_conv(x.permute([0, 2, 1])).permute([2, 1, 0])
        h_n, c_n = torch.empty(self.hidden_size, batch_size).normal_(0, math.sqrt(self.hidden_size)), torch.zeros(self.hidden_size, batch_size)
        
        lstm_1_output_seq = []
        for i in range(x.size(0)):
            out, (h_n, c_n) = self.lstm(x[i, :, :].squeeze(), (h_n, c_n))
            lstm_1_output_seq.append(out)

        if self.classifier_output:
            return self.classifier(h_n).squeeze()
<<<<<<< HEAD
        return torch.mean(torch.cat(lstm_1_output_seq, dim=0), dim=0)
        return h_n.squeeze()
=======
        return h_n.squeeze()
>>>>>>> fb5f111edc5686d5b1c8f7dcbd66775a9d85ea87
