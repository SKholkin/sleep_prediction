import os
import os.path as osp
import sys
import json
import random
from copy import deepcopy

def rewrite_data(records, train_or_val):
    if not os.path.exists(osp.join(data_path, train_or_val)):
        os.mkdir(osp.join(data_path, train_or_val))
    pos_datapoints_per_record_id = {}
    neg_datapoints_per_record_id = {}
    for record_id in records:
        if not osp.isfile(osp.join(data_path, record_id)):
            continue
        with open(osp.join(data_path, record_id), 'r') as f:
            data_json = json.load(f)
            pos_datapoints_per_record_id[record_id] = len(data_json['pos_segments'])
            neg_datapoints_per_record_id[record_id] = len(data_json['neg_segments'])
            with open(osp.join(data_path, train_or_val, record_id), 'w+') as new_f:
                json.dump(data_json, new_f)
    print(f'{train_or_val} pos {sum(pos_datapoints_per_record_id.values())} neg {sum(neg_datapoints_per_record_id.values())} segments')

if __name__ == '__main__':
    data_path = str(sys.argv[1])
    pos_datapoints_per_record_id = {}
    neg_datapoints_per_record_id = {}
    records_path_list = os.listdir(data_path)
    val_percentage = 0.2
    val_records = random.sample(records_path_list, int(len(records_path_list) * 0.2))
    train_records = deepcopy(records_path_list)
    for rec in val_records:
        train_records.remove(rec)

    rewrite_data(train_records, 'train')
    rewrite_data(val_records, 'val')

    print(f'Number of train/val recors {len(train_records)}/{len(val_records)}')
