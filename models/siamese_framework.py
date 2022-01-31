import torch
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self, core_network):
        super().__init__()
        self.core_network = core_network
        self.criterion = nn.MSELoss()

    def forward(self, x, target):
        # x:  (N, 2, L)
        # target: (N, 2)
        input_1 = x[:, 0, :].squeeze()
        input_2 = x[:, 1, :].squeeze()
        x1 = self.core_network(input_1.unsqueeze(2))
        x2 = self.core_network(input_2.unsqueeze(2))
        target = torch.abs(target[:, 0] - target[:, 1])
        loss = self.criterion(target, torch.abs(x1 - x2))
        print(loss)
        return loss, (x1, x2)
