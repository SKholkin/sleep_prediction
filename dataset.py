from torch.utils.data import Dataset
import os
import os.path as osp
import json
import random

class SegmentsIBIDataset(Dataset):
    def __init__(self, data_path, final_seg_len=120) -> None:
        super().__init__()
        self.data = []
        self.final_seg_len = final_seg_len
        for i, record_id in enumerate(os.listdir(data_path)):
            with open(osp.join(data_path, record_id), 'r') as f:
                info = json.load(f)['segments']
                [self.data.append(item) for item in info]

        self.number_of_augmentations = 0
        for item in self.data:
            self.number_of_augmentations += min(len(item) - self.final_seg_len, [item[1] for item in item].count('sleep'))
        print(f'Total number of segments {len(self.data)}')
        print(f'Total number of augmentations {self.number_of_augmentations}')

    def _random_augment(self, seg):
        numb = random.randint(0, min(len(seg) - self.final_seg_len, [item[1] for item in seg].count('sleep')))
        augmented_seg = []
        for i in range(len(seg) - self.final_seg_len - numb, len(seg) - numb):
            augmented_seg.append(seg[i])
        return augmented_seg

    def __getitem__(self, idx):
        # maybe to tensor
        return [item[0] for item in self._random_augment(self.data[idx])]

    def __len__(self):
        return len(self.data)
