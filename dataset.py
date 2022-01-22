import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import json
import random

class SegmentsIBIDataset(Dataset):
    def __init__(self, data_path, final_seg_len=120, augmentation=True) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.pos_data = []
        self.neg_data = []
        self.final_seg_len = final_seg_len
        for i, record_id in enumerate(os.listdir(data_path)):
            with open(osp.join(data_path, record_id), 'r') as f:
                data_json = json.load(f)
                [self.pos_data.append(item) for item in data_json['pos_segments']]
                [self.neg_data.append((item)) for item in data_json['neg_segments']]

        self.number_of_pos_augmentations = 0
        for item in self.pos_data:
            self.number_of_pos_augmentations += min(len(item) - self.final_seg_len, [item[1] for item in item].count('sleep'))
        print(f'Total number of pos segments {len(self.pos_data)}')
        print(f'Total number of neg segments {len(self.neg_data)}')
        self.data = [(item, True) for item in self.pos_data] + [(item , False) for item in self.neg_data]
        random.shuffle(self.data)
        print(f'Total number of pos augmentations {self.number_of_pos_augmentations}')

    def _random_sleep_augment(self, seg):
        numb = random.randint(0, min(len(seg) - self.final_seg_len, [item[1] for item in seg].count('sleep')))
        augmented_seg = []
        for i in range(len(seg) - self.final_seg_len - numb, len(seg) - numb):
            augmented_seg.append(seg[i])
        return augmented_seg

    def _random_wake_augment(self, seg):
        numb = random.randint(0, len(seg) - self.final_seg_len)
        augmented_seg = []
        for i in range(0, self.final_seg_len):
            augmented_seg.append(seg[i + numb])
        return augmented_seg

    def _without_augment(self, seq):
        return seq[len(seq) - self.final_seg_len:]

    def __getitem__(self, idx):
        if self.data[idx][1]:
            if self.augmentation:
                return torch.Tensor([item[0] for item in self._random_sleep_augment(self.data[idx][0])]), torch.FloatTensor([1.0])
            else:
                return torch.Tensor([item[0] for item in self._without_augment(self.data[idx][0])]), torch.FloatTensor([1.0])

        if self.augmentation:
            return torch.Tensor([item[0] for item in self._random_wake_augment(self.data[idx][0])]), torch.FloatTensor([0.0])
        return torch.Tensor([item[0] for item in self._without_augment(self.data[idx][0])]), torch.FloatTensor([0.0])

    def __len__(self):
        return len(self.data)
