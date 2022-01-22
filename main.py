import functools
import torch
from dataset import SegmentsIBIDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from functools import reduce
from utils import AverageMeter
from rnn import RNN

def compute_acc(pred, target):
    return sum(torch.eq(pred.round(), target).float()) / target.size(0)
    

def main():
    batch_size = 128
    num_epochs = 200
    hidden_size = 16
    print_step = 100
    lr = 1e-3
    seg_len = 120
    dataset = SegmentsIBIDataset('records_corrected_2', final_seg_len=seg_len, augmentation=True)
    dataloader = DataLoader(dataset, batch_size)
    model = RNN(hidden_size=hidden_size)
    model.load_state_dict(torch.load('conv_to_rnn_@90.pth'))
    print(f'Model parameters {sum([p.numel() for p in (model.parameters())])}')
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for epoch in range(num_epochs):
        acc_meter.reset()
        loss_meter.reset()
        for iter, (input_, target) in enumerate(dataloader):
            pred = model(torch.unsqueeze(input_, 2))
            loss = criterion(pred, target.squeeze())
            acc = compute_acc(pred, target.squeeze())
            acc_meter.update(acc)
            loss_meter.update(loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            # total_norm = 0
            # for p in model.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # print(f'gradient norm {total_norm}')
            optimizer.step()
            #if iter % print_step == 0:
            #    print(f'Iter {iter}: loss {loss} acc {acc}')
        print(f'Epoch: {epoch} acc {acc_meter.avg} loss {loss_meter.avg}')

if __name__ == '__main__':
    main()
