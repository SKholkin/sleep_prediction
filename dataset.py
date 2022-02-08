import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import json
import random

class SegmentsIBIDataset(Dataset):
    def __init__(self, data_path, train=True, final_seg_len=120, augmentation=True) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.pos_data = []
        self.neg_data = []
        self.final_seg_len = final_seg_len
        self.mode = 'train' if train else 'val'
        data_path = osp.join(data_path, self.mode)
        for i, record_id in enumerate(os.listdir(data_path)):
            with open(osp.join(data_path, record_id), 'r') as f:
                data_json = json.load(f)
                [self.pos_data.append(item) for item in data_json['pos_segments']]
                [self.neg_data.append(item) for item in data_json['neg_segments']]

        print(f'Total number of {self.mode} pos segments {len(self.pos_data)}')
        print(f'Total number of {self.mode} neg segments {len(self.neg_data)}')
        self.data = [(item, True) for item in self.pos_data] + [(item , False) for item in self.neg_data]
        random.shuffle(self.data)
        print(f'Total number of {self.mode} pos/neg augmentations {self.count_number_of_all_augments()}')

    def count_number_of_all_augments(self):
        number_of_pos_augments = 0
        number_of_neg_augments = 0
        for item in self.pos_data:
            number_of_pos_augments += self.count_number_of_pos_aug(item)
        for item in self.neg_data:
            number_of_neg_augments += self.count_number_of_neg_aug(item)
    
        return number_of_pos_augments, number_of_neg_augments

    def count_number_of_pos_aug(self, seg):
        return min(len(seg) - self.final_seg_len, [item[1] for item in seg].count('sleep'))

    def count_number_of_neg_aug(self, seg):
        return len(seg) - self.final_seg_len

    def _random_sleep_augment(self, seg):
        numb = random.randint(0, min(len(seg) - self.final_seg_len, [item[1] for item in seg].count('sleep')))
        augmented_seg = []
        for i in range(len(seg) - self.final_seg_len - numb, len(seg) - numb):
            augmented_seg.append(seg[i])
        return self._augment_output_to_seq(augmented_seg)

    def _random_wake_augment(self, seg):
        numb = random.randint(0, len(seg) - self.final_seg_len)
        augmented_seg = []
        for i in range(0, self.final_seg_len):
            augmented_seg.append(seg[i + numb])
        return self._augment_output_to_seq(augmented_seg)

    def _without_augment(self, seq):
        return self._augment_output_to_seq(seq[len(seq) - self.final_seg_len:])

    @staticmethod
    def _augment_output_to_seq(x):
        return [item[0] for item in x]
 
    def __getitem__(self, idx):
        if self.data[idx][1]:
            if self.augmentation:
                return torch.Tensor(self._random_sleep_augment(self.data[idx][0])), torch.FloatTensor([1.0])
            else:
                return torch.Tensor(self._without_augment(self.data[idx][0])), torch.FloatTensor([1.0])

        if self.augmentation:
            return torch.Tensor(self._random_wake_augment(self.data[idx][0])), torch.FloatTensor([0.0])
        return torch.Tensor(self._without_augment(self.data[idx][0])), torch.FloatTensor([0.0])

    def __len__(self):
        return len(self.data)

class SegmentsIBIContrastiveDataset(SegmentsIBIDataset):
    def __init__(self, data_path, train=True, people_paired=True, final_seg_len=120, augmentation=True) -> None:
        super().__init__(data_path, train, final_seg_len, augmentation)
        self.augmentation = augmentation
        self.people_paired = people_paired
        self.pos_data = []
        self.neg_data = []
        self.final_seg_len = final_seg_len
        self.mode = 'train' if train else 'val'
        self.record_data = {}
        self.data = []
        data_path = osp.join(data_path, self.mode)
        for i, record_id in enumerate(os.listdir(data_path)):
            with open(osp.join(data_path, record_id), 'r') as f:
                data_json = json.load(f)
                data = [(item, 1) for item in data_json['pos_segments']]
                data += [(item, 0) for item in data_json['neg_segments']]
                self.record_data[int(record_id)] = [(item, 1) for item in data_json['pos_segments']]
                self.record_data[int(record_id)] += [(item, 0) for item in data_json['neg_segments']]
                self.data += [(item[0], item[1], int(record_id)) for item in self.record_data[int(record_id)]]
        
        print(f'Total number of {self.mode} pos segments {len(self.pos_data)}')
        print(f'Total number of {self.mode} neg segments {len(self.neg_data)}')
        #self.data = [(item, True) for item in self.pos_data] + [(item , False) for item in self.neg_data]
        random.shuffle(self.data)
        print(f'Total number of {self.mode} pos/neg augmentations {self.count_number_of_all_augments()}')

    def _get_augment_fn(self, data):
        if not self.augmentation:
            return self._without_augment
        elif data[1]:
            return self._random_sleep_augment
        return self._random_wake_augment

    def __getitem__(self, idx):
        # return [2, N], [2]
        self.data[idx]
        record_data = self.record_data[self.data[idx][2]]
        first_elem = self.data[idx]
        second_elem =  record_data[random.randint(0, len(record_data) - 1)]
        first_aug_fn = self._get_augment_fn(first_elem)
        second_aug_fn = self._get_augment_fn(second_elem)
        x = torch.cat((torch.Tensor((first_aug_fn(first_elem[0]))).unsqueeze(0), torch.Tensor(second_aug_fn(second_elem[0])).unsqueeze(0)), 0)
        label = torch.cat((torch.FloatTensor([first_elem[1]]), torch.FloatTensor([second_elem[1]])), 0)
        return x, label
