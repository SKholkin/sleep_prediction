import pandas as pd
import os.path as osp
import os
import json
import random
from functools import reduce
from tqdm import tqdm

def make_segment(record_dataframe, idx, record_len, corrected_allowed):
    seg = []
    corrected_count =  0
    for i in range(1, record_len + 1):
        if record_dataframe.iloc[idx - record_len + i]['corrected'] == 1:
            corrected_count += 1
            if corrected_count > corrected_allowed:
                break
        part_name = 'sleep' if 'sleep' in record_dataframe.iloc[idx - record_len + i]['Part'].lower() else 'wake'
        seg.append((record_dataframe.iloc[idx - record_len + i]['Interp RR-interval'], part_name))
    if len(seg) == record_len:
        return seg
    return None

def negative_segment_record(record_dataframe, record_len=120, corrected_allowed=0):
    counter_wake = 0
    is_sleep = False
    segments = []
    for idx, row in record_dataframe.transpose().iteritems():
        if 'wake' in row['Part'].lower():
            if is_sleep:
                is_sleep = False
            counter_wake += 1
        else:
            if not is_sleep:
                if counter_wake >= 120:
                    # make segments by 180 len
                    if counter_wake <= 180:
                        seg = make_segment(record_dataframe, idx, counter_wake, corrected_allowed)
                        if seg is not None:
                            segments.append(seg)
                    else:
                        n = counter_wake // 180
                        indent = (counter_wake % 180) // 2
                        for i in range(1, n + 1):
                            seg = make_segment(record_dataframe, idx + indent - counter_wake + i * 180, 180, corrected_allowed)
                            if seg is not None:
                                segments.append(seg)
                counter_wake = 0
            is_sleep = True
    if counter_wake >= 120:
        seg = []
        if counter_wake <= 180:
            seg = make_segment(record_dataframe, idx, counter_wake, corrected_allowed)
            if seg is not None:
                segments.append(seg)
        else:
            n = counter_wake // 180
            indent = (counter_wake % 180) // 2
            for i in range(n):
                seg = make_segment(record_dataframe, idx + indent - counter_wake + i * 180, 180, corrected_allowed)
                if seg is not None:
                    segments.append(seg)
    return segments


def positive_segment_record(record_dataframe, record_len=120, corrected_allowed=0):
    counter_wake = 0
    counter_sleep = 0
    sleep = False
    is_record_ready = False
    segments = []
    for idx, row in record_dataframe.transpose().iteritems():
        if 'wake' in row['Part'].lower():
            if sleep:
                if is_record_ready:
                    seg = make_segment(record_dataframe, idx - 1, counter_sleep + min(counter_wake, record_len), corrected_allowed)
                    if seg is not None:
                        segments.append(seg)
                counter_wake = 0
                counter_sleep = 0
                is_record_ready = False
                sleep = False
            counter_wake += 1

        if 'sleep' in row['Part'].lower():
            sleep = True
            if counter_wake > 0:
                counter_sleep += 1
            if counter_sleep >= record_len - counter_wake and counter_wake >= record_len / 2:
                is_record_ready = True
                if counter_sleep >= record_len / 2:
                    seg = make_segment(record_dataframe, idx, counter_sleep + min(counter_wake, record_len), corrected_allowed)
                    if seg is not None:
                        segments.append(seg)
                    counter_wake = 0
                    counter_sleep = 0
                    is_record_ready = False
                    sleep = False

    if is_record_ready:
        seg = make_segment(record_dataframe, record_dataframe['Part'].count() - 1, counter_sleep + min(counter_wake, record_len), corrected_allowed)
        if seg is not None:
            segments.append(seg)
    return segments

def main():
    dataset_name = 'IBI_dataset'
    record_file_name = 'ZEPHYR_RR_FILTERED_49.csv'
    record_len = 120
    corrected_allowed = 2
    segments_json = []
    write_path = f'records_corrected_debug_{corrected_allowed}'
    if not osp.exists(write_path):
        os.mkdir(write_path)
    for i, record_id in enumerate(tqdm(os.listdir(dataset_name))):
        record_dataframe = pd.read_csv(osp.join(dataset_name, record_id, record_file_name))
        pos_segments = positive_segment_record(record_dataframe, record_len, corrected_allowed)
        neg_segments = negative_segment_record(record_dataframe, record_len, corrected_allowed)
        neg_segments = random.choices(neg_segments, k=min(len(neg_segments), len(pos_segments)))
        segments_json.append({'pos_segments': pos_segments, 'neg_segments': neg_segments})
        with open(osp.join(write_path, record_id), 'w+') as f:
            json.dump(segments_json[-1], f)

if __name__ == '__main__':
    main()
