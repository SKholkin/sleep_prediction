import torch
from torch import nn
from models.layernorm_lstm import LayerNormLSTM
import math

class SequencedLSTMs(nn.Module):
    def __init__(self, hidden_dims, classifier_output=False):
        super().__init__()
        self.classifier_output = classifier_output
        self.hidden_dim1, self.hidden_dim2 = hidden_dims
        self.output_dim = self.hidden_dim2
        self.lstm_1 = LayerNormLSTM(input_size=1, hidden_size=self.hidden_dim1)
        self.lstm_2 = LayerNormLSTM(input_size=self.hidden_dim1, hidden_size=self.hidden_dim2)
        if self.classifier_output:
            self.classifier = nn.Sequential(nn.Linear(in_features=self.hidden_dim2, out_features=1), nn.Sigmoid())

    def forward(self, x):
        # x: (N, L, 1)
        # lstm inputs: (L, H, N)
        batch_size = x.size(0)
        x = x.permute([1, 2, 0])
        h_n_lstm_1, c_n_lstm_1 = torch.empty(self.hidden_dim1, batch_size).normal_(0, math.sqrt(self.hidden_dim1)), torch.zeros(self.hidden_dim1, batch_size)
        lstm_1_output_seq = []
        for i in range(x.size(0)):
            out, (h_n_lstm_1, c_n_lstm_1) = self.lstm_1(x[i, :, :], (h_n_lstm_1, c_n_lstm_1))
            lstm_1_output_seq.append(out)
        x = torch.cat(lstm_1_output_seq, dim=0).permute([0, 2, 1])
        
        lstm_2_output_seq = []
        h_n_lstm_2, c_n_lstm_2 = torch.empty(self.hidden_dim2, batch_size).normal_(0, math.sqrt(self.hidden_dim1)), torch.zeros(self.hidden_dim2, batch_size)
        for i in range(x.size(0)):
            out, (h_n_lstm_2, c_n_lstm_2) = self.lstm_2(x[i, :, :], (h_n_lstm_2, c_n_lstm_2))
            lstm_2_output_seq.append(out)

        lstm_2_output_seq = torch.cat(lstm_2_output_seq, dim=0).permute([0, 2, 1])
        sum_of_all_representations = torch.sum(lstm_2_output_seq, dim=0).permute([1, 0])

        if self.classifier_output:
            return self.classifier(sum_of_all_representations).squeeze()
        return sum_of_all_representations.squeeze()
