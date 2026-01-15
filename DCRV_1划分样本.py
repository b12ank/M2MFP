# ^_^
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


train_date_range = ("2024-01-01", "2024-06-01")
test_date_range = ("2024-06-01", "2024-08-01")     # 测试集时间


combined_sn_feature_data_path = r"E:\pycharmproject\raw_code\新工作\DCRV\DCRV_processed_data"
ticket_path = r"E:\pycharmproject\raw_code\failure_ticket.csv"
train_data_path = r"E:\pycharmproject\raw_code\新工作\DCRV\train"
test_data_path = rf"E:\pycharmproject\raw_code\新工作\DCRV\test_{test_date_range[0][5:7]}{test_date_range[1][5:7]}"         # 修改成test_date_range的范围


os.makedirs(train_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)


ticket = pd.read_csv(ticket_path)
pos_ticket = ticket[ticket['alarm_time'] <= 1717171200]  # 2024-06-01

# ticket中sn_name对应报警时间的字典
pos_ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(pos_ticket['sn_name']), list(pos_ticket['alarm_time']))}
ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(ticket['sn_name']), list(ticket['alarm_time']))}


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
        if data_type == "DCRV":
            data = pd.read_feather(os.path.join(combined_sn_feature_data_path, sn_file))
            if data.empty:
                return None
            data = data[(data['ReportTime'] <= end_time) & (data['ReportTime'] >= start_time)]
            if data.empty:
                return None
            data = data.sort_values(by=['ReportTime'])    # pos取全部
        data['label'] = 1

        index_list = [(sn_file[:-8], log_time) for log_time in data['ReportTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def process_neg_file(args):
    sn_file = args[0]
    data_type = args[1]
    if not pos_ticket_sn_map.get(sn_file[:-8]):
        end_time = 1717171200 - 30 * 24 * 3600
        start_time = 1717171200 - 150 * 24 * 3600
        if data_type == "DCRV":
            data = pd.read_feather(os.path.join(combined_sn_feature_data_path, sn_file))
        if data.empty:
            return None
        data = data[(data['ReportTime'] <= end_time) & (data['ReportTime'] >= start_time)]
        if data.empty:
            return None
        if data_type == "DCRV":
            # data = data.sort_values(by=['ReportTime'])[-80:]
            if len(data) > 40:
                data = pd.concat([data.sample(n=20, random_state=40).copy(), data.sort_values(by=['ReportTime'])[-20:].copy()], axis=0)
        data['label'] = 0

        index_list = [(sn_file[:-8], log_time) for log_time in data['ReportTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def get_positive_train_data(data_type):
    if data_type == "DCRV":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 3
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
            if data_type == "DCRV":
                pos_data_all.to_feather(f'{train_data_path}/positive_train1-6_{chunk_index+1}.feather')
        else:
            print(f"{data_type}正样本训练集{chunk_index + 1}/{chunk_size}无正样本")
    print("生成全部正样本训练集")


def get_negative_train_data(data_type):
    if data_type == "DCRV":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 3

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
            if data_type == "DCRV":
                neg_data_all.to_feather(f'{train_data_path}/negative03-04_train1-5_{chunk_index+1}.feather')
        else:
            print(f"{data_type}负样本训练集{chunk_index + 1}/{chunk_size}无负样本")
    print("生成全部负样本训练集")


def get_test_data(data_type):
    if data_type == "DCRV":
        file_list = os.listdir(combined_sn_feature_data_path)
        chunk_size = 10  # 测试集分10块输出
    file_list = [x for x in file_list if x.endswith('.feather')]
    file_list.sort()

    split_test_files = np.array_split(file_list, chunk_size)
    for chunk_index, file_chunk in enumerate(split_test_files):
        print(f"正在生成{data_type}测试集{chunk_index + 1}/{chunk_size}")
        test_data_all = []
        sample_count_all = 0

        for file in tqdm(file_chunk):
            if data_type == "DCRV":
                data_tmp = pd.read_feather(os.path.join(combined_sn_feature_data_path, file))
                if data_tmp.empty:
                    continue
                data_tmp = data_tmp[data_tmp['ReportTime'] > datetime_to_timestamp(test_date_range[0])]
                data_tmp = data_tmp[data_tmp['ReportTime'] <= datetime_to_timestamp(test_date_range[1])]
                data_tmp = data_tmp.sort_values(by=['ReportTime'])
            if data_tmp.empty:
                continue

            index_list = [(file[:-8], log_time) for log_time in data_tmp['ReportTime']]
            data_tmp.index = pd.MultiIndex.from_tuples(index_list)
            sample_count_all += len(data_tmp)
            test_data_all.append(data_tmp)

        test_data_all = parallel_concat(test_data_all)
        # test_data_all = concat_in_chunks(test_data_all)
        if test_data_all is not None:
            if data_type == "DCRV":
                test_data_all.to_feather(os.path.join(test_data_path, f"test_data_{chunk_index+1}.feather"))
        else:
            print(f"{data_type}测试集{chunk_index + 1}/{chunk_size}无测试样本")
    print("已经生成全部测试集合")


if __name__ == "__main__":
    get_positive_train_data("DCRV")
    get_negative_train_data("DCRV")
    get_test_data("DCRV")

    data_list = []  # 负样本列表
    data_list1 = []
    for file in tqdm(os.listdir(train_data_path)):
        df = pd.read_feather(os.path.join(train_data_path, file))
        if file.startswith("negative"):  # 负样本放到data_list
            data_list.append(df)
        else:  # 正样本放到data_list1
            data_list1.append(df)
    merged_df = pd.concat(data_list, ignore_index=False)
    merged_df1 = pd.concat(data_list1, ignore_index=False)
    train_merge_path = r"E:\pycharmproject\raw_code\新工作\DCRV\train_merge"
    os.makedirs(train_merge_path, exist_ok=True)

    merged_df.to_feather(os.path.join(train_merge_path, r"negative_train_neg40.feather"))
    merged_df1.to_feather(os.path.join(train_merge_path, r"positive_train_neg40.feather"))
