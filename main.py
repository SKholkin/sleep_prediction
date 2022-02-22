import torch
from dataset import SegmentsIBIDataset, SegmentsIBIContrastiveDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from argparse import ArgumentParser

from utils import AverageMeter
from models.rnn import RNN
from models.two_rnn_model import SequencedLSTMs
from models.siamese_framework import SiameseNetwork


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--data', type=str, help='dataset path')
    return parser

def compute_acc(pred, target):
    return sum(torch.eq(pred.round(), target).float()) / target.size(0)  

def compute_siamese_acc(x1, x2, target):
    # print(torch.abs(x1 - x2).round())
    # print(target)
    return sum(torch.eq(torch.abs(x1 - x2).round(), target)) / target.size(0)

def main(data_path):
    batch_size = 32
    num_epochs = 1000
    hidden_size = 32
    print_step = 100
    test_freq= 5
    lr = 1e-3
    seg_len = 120
    train_dataloader = DataLoader(SegmentsIBIContrastiveDataset(data_path, train=True, final_seg_len=seg_len, augmentation=True), batch_size)
    val_dataloader = DataLoader(SegmentsIBIContrastiveDataset(data_path, train=False, final_seg_len=seg_len, augmentation=True), batch_size)
    print(f'Train len {len(list(train_dataloader))}')
    print(f'Val len {len(val_dataloader)}')

    model = RNN(hidden_size=hidden_size, classifier_output=False)
    #model = SequencedLSTMs((64, 16), classifier_output=False)
    model = SiameseNetwork(model, classifier_output=True)

    print(f'Model parameters {sum([p.numel() for p in (model.parameters())])}')
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        print(f'Training epoch {epoch}')
        train_acc , train_loss = train_epoch(model, optimizer, train_dataloader, criterion)
        print(f'Train: acc {train_acc} loss {train_loss}')
        if (epoch + 1) % test_freq == 0:
            print('Validating...')
            val_acc , val_loss = val(model, val_dataloader, criterion)
            print(f'Val: acc {val_acc} loss {val_loss}')
        if (epoch + 1) % 20 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
        

def train_epoch(model, optimizer, train_dataloader, criterion):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for iter, (input_, target) in enumerate(train_dataloader):
        loss, outputs = model(input_, target)
        acc = compute_acc(outputs, torch.abs(target[:, 0] - target[:, 1]).squeeze())

        loss_meter.update(float(loss))
        acc_meter.update(float(acc))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    return acc_meter.avg, loss_meter.avg

def val(model, val_dataloader, criterion):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for iter, (input_, target) in enumerate(val_dataloader):
        loss, outputs = model(input_, target)
        # acc = compute_siamese_acc(*outputs, torch.abs(target[:, 0] - target[:, 1]))
        # loss = criterion(pred, target.squeeze())
        acc = compute_acc(outputs, torch.abs(target[:, 0] - target[:, 1]).squeeze())
        # acc_meter.update(float(acc))
        loss_meter.update(float(loss))
        acc_meter.update(float(acc))
    return acc_meter.avg, loss_meter.avg

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args.data)
