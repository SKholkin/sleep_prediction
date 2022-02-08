import torch
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self, core_network, classifier_output=True):
        super().__init__()
        self.core_network = core_network
        self.classifier_output = classifier_output
        if self.classifier_output:
            self.classifier = nn.Sequential(nn.Linear(in_features=self.core_network.output_dim, out_features=1), nn.Sigmoid())
            self.criterion = nn.BCELoss()
            self.criterion_mse = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x, target):
        # x:  (N, 2, L)
        # target: (N, 2)
        input_1 = x[:, 0, :].squeeze()
        input_2 = x[:, 1, :].squeeze()
        x1 = self.core_network(input_1.unsqueeze(2))
        x2 = self.core_network(input_2.unsqueeze(2))
        if self.classifier_output:
            x = torch.cat([x1, x2], dim=1)
            x = self.classifier(x1 - x2).squeeze()
            # print(torch.abs(target[:, 0] - target[:, 1]))
            # print(-(2 * torch.abs(target[:, 0] - target[:, 1]) - 1))
            # loss = torch.mean(torch.abs(target[:, 0] - target[:, 1]) * self.criterion_mse(x1, x2))
            loss = self.criterion(x, torch.abs(target[:, 0] - target[:, 1]).squeeze())
            return loss, x
        loss = self.criterion(torch.abs(target[:, 0] - target[:, 1]), torch.abs(x1 - x2))
        return loss, (x1, x2)
