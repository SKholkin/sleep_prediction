import pandas as pd
import os.path as osp
import os
import json

def make_segment(record_dataframe, idx, record_len, corrected_allowed):
    seg = []
    corrected_count =  0
    for i in range(record_len):
        if record_dataframe.iloc[idx - record_len + i]['corrected'] == 1:
            corrected_count += 1
            if corrected_count > corrected_allowed:
                break
        part_name = 'sleep' if 'sleep' in record_dataframe.iloc[idx - record_len + i]['Part'].lower() else 'wake'
        seg.append((record_dataframe.iloc[idx - record_len + i]['Interp RR-interval'], part_name))
    if len(seg) == record_len:
        return seg
    return None

def segment_record(record_dataframe, record_len=120, corrected_allowed=0):
    counter_wake = 0
    counter_sleep = 0
    sleep = False
    is_record_ready = False
    segments = []
    for idx, row in record_dataframe.transpose().iteritems():

        if 'wake' in row['Part'].lower():
            if sleep:
                if is_record_ready:
                    seg = make_segment(record_dataframe, idx, counter_sleep + min(counter_wake, record_len), corrected_allowed)
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
        seg = make_segment(record_dataframe, record_dataframe['Part'].count(), counter_sleep + min(counter_wake, record_len), corrected_allowed)
        if seg is not None:
            segments.append(seg)
    return segments

def main():
    dataset_name = 'IBI_dataset'
    record_file_name = 'ZEPHYR_RR_FILTERED_49.csv'
    record_len = 120
    segments_total = 0
    write_path = 'records_corrected_2'
    if not osp.exists(write_path):
        os.mkdir(write_path)
    for i, record_id in enumerate(os.listdir(dataset_name)):
        record_dataframe = pd.read_csv(osp.join(dataset_name, record_id, record_file_name))
        segments = segment_record(record_dataframe, record_len, corrected_allowed=2)
        print(len(segments))
        segments_json = {'segments': segments}
        with open(osp.join(write_path, record_id), 'w+') as f:
            json.dump(segments_json, f)
        segments_total += len(segments)
    print(segments_total)
if __name__ == '__main__':
    main()
