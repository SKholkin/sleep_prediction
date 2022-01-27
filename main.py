import functools
import torch
from dataset import SegmentsIBIDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from functools import reduce
from argparse import ArgumentParser

from utils import AverageMeter
from rnn import RNN
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tb_logs')


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--data', type=str, help='dataset path')
    return parser

def compute_acc(pred, target):
    return sum(torch.eq(pred.round(), target).float()) / target.size(0)


def main(args):
    batch_size = 128
    num_epochs = 1000
    hidden_size = 32
    print_step = 100
    test_freq= 5
    lr = 1e-3
    seg_len = 120
    train_dataloader = DataLoader(SegmentsIBIDataset(args.data, train=True, final_seg_len=seg_len, augmentation=False), batch_size)
    val_dataloader = DataLoader(SegmentsIBIDataset(args.data, train=False, final_seg_len=seg_len, augmentation=False), batch_size)
    model = RNN(hidden_size=hidden_size)
    torch.save(model, 'model.pth')
    # from torchviz import make_dot
    # make_dot(torch.zeros([128, 120, 1]), params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    # writer.add_graph(model, torch.zeros([128, 120, 1]))
    #torch.onnx.export(model, torch.zeros([128, 120, 1]), 'model.onnx', opset_version=13)
    #model.load_state_dict(torch.load('conv_to_rnn_@90.pth'))
    print(f'Model parameters {sum([p.numel() for p in (model.parameters())])}')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-04)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        print(f'Training epoch {epoch}')
        train_acc , train_loss = train_epoch(model, optimizer, train_dataloader, criterion)
        print(f'Train: acc {train_acc} loss {train_loss}')
        if (epoch + 1) % test_freq == 0:
            print('Validating...')
            val_acc , val_loss = val(model, val_dataloader, criterion)
            print(f'Val: acc {val_acc} loss {val_loss}')
        

def train_epoch(model, optimizer, train_dataloader, criterion):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for iter, (input_, target) in enumerate(train_dataloader):
        print('Input size')
        print(torch.unsqueeze(input_, 2).size())
        pred = model(torch.unsqueeze(input_, 2))
        loss = criterion(pred, target.squeeze())
        acc = compute_acc(pred, target.squeeze())
        acc_meter.update(float(acc))
        loss_meter.update(float(loss))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        print(iter)
        #if iter % print_step == 0:
        #    print(f'Iter {iter}: loss {loss} acc {acc}')
    return acc_meter.avg, loss_meter.avg

def val(model, val_dataloader, criterion):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for iter, (input_, target) in enumerate(val_dataloader):
        pred = model(torch.unsqueeze(input_, 2))
        loss = criterion(pred, target.squeeze())
        acc = compute_acc(pred, target.squeeze())
        acc_meter.update(float(acc))
        loss_meter.update(float(loss))
    return acc_meter.avg, loss_meter.avg

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
