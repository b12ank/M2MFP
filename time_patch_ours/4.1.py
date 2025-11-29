# ^_^
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

time_window_size_map = {15 * 60: '15m', 60 * 60: '1h', 6 * 3600: '6h'}


processed_time_point_data_path = r"E:\pycharmproject\raw_code\work1\bfse_processed_data"
combined_sn_feature_data_path = r"E:\pycharmproject\raw_code\work1\time_patch_processed"
ticket_path = r"E:\pycharmproject\raw_code\failure_ticket.csv"
train_data_path_time_patch = r"E:\pycharmproject\raw_code\work1\time_patch_train"
# train_data_path_time_patch = r"E:\pycharmproject\raw_code\work\time_patch_train\time_patch_1"

test_data_path_time_patch = r"E:\pycharmproject\raw_code\work1\time_patch_test"
train_data_path_time_point = r"E:\pycharmproject\raw_code\work1\time_point_train"
# train_data_path_time_point = r"E:\pycharmproject\raw_code\work\time_point_train\time_point_1"

test_data_path_time_point = r"E:\pycharmproject\raw_code\work1\time_point_test"

os.makedirs(train_data_path_time_patch, exist_ok=True)
os.makedirs(test_data_path_time_patch, exist_ok=True)
os.makedirs(train_data_path_time_point, exist_ok=True)
os.makedirs(test_data_path_time_point, exist_ok=True)

ticket = pd.read_csv(ticket_path)
pos_ticket = ticket[ticket['alarm_time'] <= 1717171200]  # 2024-06-01
# pos_ticket = ticket[ticket['alarm_time'] <= 1714492800]  # 2024-05-01

# ticket中sn_name对应报警时间的字典
pos_ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(pos_ticket['sn_name']), list(pos_ticket['alarm_time']))}
ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(ticket['sn_name']), list(ticket['alarm_time']))}

train_date_range = ("2024-01-01", "2024-06-01")
test_date_range = ("2024-08-01", "2024-10-01")


def datetime_to_timestamp(date: str) -> int:
    """
    Takes a date string in the format "YYYY-MM-DD" and returns the corresponding Unix timestamp.
    """

    return int(datetime.strptime(date, "%Y-%m-%d").timestamp())


def concat_in_chunks(chunks):
    chunks = [chunk for chunk in chunks if chunk is not None]
    if chunks:
        return pd.concat(chunks)
    return None


def parallel_concat(results, num_threads=4, chunk_size=200):
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    with Pool(num_threads) as pool:
        concatenated_chunks = pool.map(concat_in_chunks, chunks)

    return concat_in_chunks(concatenated_chunks)


def process_pos_file(args):
    sn_file = args[0]
    data_type = args[1]
    if pos_ticket_sn_map.get(sn_file[:-8]):
        end_time = pos_ticket_sn_map.get(sn_file[:-8])
        start_time = end_time - 30 * 24 * 3600
        if data_type == "time_patch":
            data = pd.read_feather(os.path.join(combined_sn_feature_data_path, sn_file))
            data = data[(data['LogTime'] <= end_time) & (data['LogTime'] >= start_time)]
            data = data.sort_values(by=['all_ce_log_num_15m'])[-20:]    # 只取all_ce_log_num_15m最大的20行
        elif data_type == "time_point":
            data = pd.read_feather(os.path.join(processed_time_point_data_path, sn_file))
            data = data[(data['LogTime'] <= end_time) & (data['LogTime'] >= start_time)]
            if len(data) > 80:
                data = data.sample(n=80, random_state=40)     # 对于time_point的训练集一块dimm的正样本只随机取20条
        data['label'] = 1

        index_list = [(sn_file[:-8], log_time) for log_time in data['LogTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def process_neg_file(args):
    sn_file = args[0]
    data_type = args[1]
    if not pos_ticket_sn_map.get(sn_file[:-8]):
        end_time = 1717171200 - 30 * 24 * 3600
        start_time = 1717171200 - 150 * 24 * 3600
        if data_type == "time_patch":
            data = pd.read_feather(os.path.join(combined_sn_feature_data_path, sn_file))
        elif data_type == "time_point":
            data = pd.read_feather(os.path.join(processed_time_point_data_path, sn_file))
        data = data[(data['LogTime'] <= end_time) & (data['LogTime'] >= start_time)]
        if data.empty:
            return None
        if data_type == "time_patch":
            # data = data.sort_values(by=['all_ce_log_num_15m'])[-20:]
            if len(data) > 40:
                data = data.sample(n=40, random_state=40)
        elif data_type == "time_point":
            if len(data) > 80:
                data = data.sample(n=80, random_state=40)     # 一块dimm的负样本只随机取20条
        data['label'] = 0

        index_list = [(sn_file[:-8], log_time) for log_time in data['LogTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def get_positive_train_data(data_type):
    if data_type == "time_patch":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 3
    elif data_type == "time_point":
        file_list = os.listdir(processed_time_point_data_path)
        chunk_size = 10
    file_list = [x for x in file_list if x.endswith('.feather')]
    file_list.sort()

    split_test_files = np.array_split(file_list, chunk_size)
    for chunk_index, file_chunk in enumerate(split_test_files):
        print(f"正在生成{data_type}正样本训练集{chunk_index + 1}/{chunk_size}")
        args_file_list = [(i, data_type) for i in file_chunk]
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_pos_file, args_file_list), total=len(file_chunk)))
        pos_data_all = parallel_concat(results)
        # print(f"{data_type}正样本：", pos_data_all.info)
        if pos_data_all is not None:
            if data_type == "time_patch":
                pos_data_all.to_feather(f'{train_data_path_time_patch}/positive_train1-6_{chunk_index+1}.feather')
            elif data_type == "time_point":
                pos_data_all.to_feather(f'{train_data_path_time_point}/positive_lt80_train1-6_{chunk_index+1}.feather')
        else:
            print(f"{data_type}正样本训练集{chunk_index + 1}/{chunk_size}无正样本")
    print("生成全部正样本训练集")


def get_negative_train_data(data_type):
    if data_type == "time_patch":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 3
    elif data_type == "time_point":
        file_list = os.listdir(processed_time_point_data_path)
        chunk_size = 10
    file_list = [x for x in file_list if x.endswith('.feather')]
    file_list.sort()

    split_test_files = np.array_split(file_list, chunk_size)
    for chunk_index, file_chunk in enumerate(split_test_files):
        print(f"正在生成{data_type}负样本训练集{chunk_index + 1}/{chunk_size}")
        args_file_list = [(i, data_type) for i in file_chunk]
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_neg_file, args_file_list), total=len(file_chunk)))
        neg_data_all = parallel_concat(results)
        # print(f"{data_type}负样本：", neg_data_all.info)
        if neg_data_all is not None:
            if data_type == "time_patch":
                neg_data_all.to_feather(f'{train_data_path_time_patch}/negative03-04_train1-5_{chunk_index+1}.feather')
            elif data_type == "time_point":
                neg_data_all.to_feather(f'{train_data_path_time_point}/negative_month1-4_lt80_train1-6_{chunk_index+1}.feather')
        else:
            print(f"{data_type}负样本训练集{chunk_index + 1}/{chunk_size}无负样本")
    print("生成全部负样本训练集")


def get_test_data(data_type):
    if data_type == "time_patch":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 12  # 测试集分六块输出
    elif data_type == "time_point":
        file_list = os.listdir(processed_time_point_data_path)
        chunk_size = 10
    file_list = [x for x in file_list if x.endswith('.feather')]
    file_list.sort()

    split_test_files = np.array_split(file_list, chunk_size)
    for chunk_index, file_chunk in enumerate(split_test_files):
        print(f"正在生成{data_type}测试集{chunk_index + 1}/{chunk_size}")
        test_data_all = []
        sample_count_all = 0

        for file in tqdm(file_chunk):
            if data_type == "time_patch":
                data_tmp = pd.read_feather(os.path.join(combined_sn_feature_data_path, file))
                data_tmp = data_tmp[data_tmp['LogTime'] > datetime_to_timestamp(test_date_range[0])]
                data_tmp = data_tmp[data_tmp['LogTime'] <= datetime_to_timestamp(test_date_range[1])]
                data_tmp = data_tmp.sort_values(by=['LogTime'])[-40:]     # 每条dimm控制根据logtime排序的最后40行
            elif data_type == "time_point":
                data_tmp = pd.read_feather(os.path.join(processed_time_point_data_path, file))
                data_tmp = data_tmp[data_tmp['LogTime'] > datetime_to_timestamp(test_date_range[0])]
                data_tmp = data_tmp[data_tmp['LogTime'] <= datetime_to_timestamp(test_date_range[1])]
                data_tmp = data_tmp.sort_values(by=['LogTime'])[-40:]        # 每条dimm控制根据logtime排序的最后40行
            if data_tmp.empty:
                continue

            index_list = [(file[:-8], log_time) for log_time in data_tmp['LogTime']]
            data_tmp.index = pd.MultiIndex.from_tuples(index_list)
            sample_count_all += len(data_tmp)
            test_data_all.append(data_tmp)

        test_data_all = parallel_concat(test_data_all)
        # test_data_all = concat_in_chunks(test_data_all)
        if test_data_all is not None:
            if data_type == "time_patch":
                test_data_all.to_feather(os.path.join(test_data_path_time_patch, f"a_test_data_{chunk_index+1}.feather"))
            elif data_type == "time_point":
                test_data_all.to_feather(os.path.join(test_data_path_time_point, f"test_lt40_data{chunk_index+1}.feather"))
        else:
            print(f"{data_type}测试集{chunk_index + 1}/{chunk_size}无测试样本")
    print("已经生成全部测试集合")


if __name__ == "__main__":
    # for data_type in ["time_patch", "time_point"]:
    #     get_positive_train_data(data_type)
    #     get_negative_train_data(data_type)
    #     get_test_data(data_type)
    get_positive_train_data("time_point")
    get_negative_train_data("time_point")
    # get_test_data("time_point")

