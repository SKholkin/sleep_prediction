import os
import os.path as osp
import sys
import json
import random
from copy import deepcopy

def make_dirs():
    if not os.path.exists(osp.join(data_path, 'train')):
        os.mkdir(osp.join(data_path, 'train'))
    
    if not os.path.exists(osp.join(data_path, 'val')):
        os.mkdir(osp.join(data_path, 'val'))

if __name__ == '__main__':
    data_path = str(sys.argv[1])
    records_path_list = os.listdir(data_path)
    val_percentage = 0.2
    val_records = random.sample(records_path_list, int(len(records_path_list) * val_percentage))
    train_records = deepcopy(records_path_list)
    make_dirs()
    pos_segments = []
    neg_segments = []
    for record_id in records_path_list:
        if not osp.isfile(osp.join(data_path, record_id)):
            continue
        with open(osp.join(data_path, record_id), 'r') as f:
            data_json = json.load(f)
            pos_segments += data_json['pos_segments']
            neg_segments += data_json['neg_segments']

    val_pos_segments = random.sample(pos_segments, int(len(pos_segments) * val_percentage))
    val_neg_segments = random.sample(neg_segments, int(len(neg_segments) * val_percentage))
    train_pos_segments = deepcopy(pos_segments)
    train_neg_segments = deepcopy(neg_segments)
    
    for seg in val_pos_segments:
        train_pos_segments.remove(seg)
        
    for seg in val_neg_segments:
        train_neg_segments.remove(seg)

    for i, segment in enumerate(train_pos_segments):
        with open(osp.join(data_path, 'train', 'pos' + str(i)), 'w+') as new_f:
                json.dump({'pos_segments': [segment], 'neg_segments': []}, new_f)
    for i, segment in enumerate(train_neg_segments):
        with open(osp.join(data_path, 'train', 'neg' + str(i)), 'w+') as new_f:
                json.dump({'pos_segments': [], 'neg_segments': [segment]}, new_f)
    for i, segment in enumerate(val_pos_segments):
        with open(osp.join(data_path, 'val', 'pos' + str(i)), 'w+') as new_f:
                json.dump({'pos_segments': [segment], 'neg_segments': []}, new_f)
    for i, segment in enumerate(val_neg_segments):
        with open(osp.join(data_path, 'val', 'neg' + str(i)), 'w+') as new_f:
                json.dump({'pos_segments': [], 'neg_segments': [segment]}, new_f)
