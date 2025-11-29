# ^_^
from pathlib import Path
import os
import pickle
import abc
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, List, Dict, NoReturn, Union, Any, Optional
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from memory_profiler import profile

DATASET_DIR = r"E:\pycharmproject\M2-MFP\data1"
WORK_DIR = r"E:\pycharmproject\raw_code\work"
PARITY_ROW_COUNT = 8
PARITY_COLUMN_COUNT = 4


def indices_of_ones(binary_string: str) -> List[int]:
    """
    Get the indices of the ones in a binary string.

    :param binary_string: binary string
    :return: list of indices of ones
    """

    return [index for index, char in enumerate(binary_string) if char == "1"]


def get_binary_string_features(
        binary_string: str
) -> Tuple[int, int, int, int, int]:
    """
    Get the binary string features, including features of a one-dimensional binary string.

    :param binary_string: binary string
    :return: tuple of binary string information, including:
        - bit_count: the number of valid bits in the binary string
        - bit_min_interval: the minimum interval between valid bits
        - bit_max_interval: the maximum interval between valid bits
        - bit_max_consecutive_length: the maximum length of consecutive valid bits
        - bit_consecutive_length: the cumulative continuous length of consecutive valid bits
    """

    bit_count = binary_string.count("1")
    bit_min_interval = len(binary_string)
    bit_max_interval = 0
    bit_max_consecutive_length = 0
    bit_consecutive_length = 0

    indices = indices_of_ones(binary_string)

    if len(indices) > 0:
        bit_max_interval = indices[-1] - indices[0]
        bit_max_consecutive_length = max([len(i) for i in binary_string.split("0")])
        bit_consecutive_length = binary_string.count("1") - sum(
            1 for i in binary_string.split("0") if len(i) > 0
        )
        if len(indices) > 1:
            bit_min_interval = min(np.diff(indices))

    return (
        bit_count,
        bit_min_interval,
        bit_max_interval,
        bit_max_consecutive_length,
        bit_consecutive_length,
    )


def process_file(args) -> List[int]:
    """
    Process the file to get the unique error log parity.

    :param args: (data_path, file_name)的元组   元组做输入用于多进程 imap_unordered和imap只能传递单个参数
    :return: list of unique error log parity
    """
    data_path, file = args
    data = pd.read_feather(os.path.join(data_path, file))
    return data.RetryRdErrLogParity.dropna().astype(np.int64).unique().tolist()


def get_binary_string_map(
        binary_string_length: int,
) -> Dict[str, Tuple[int, int, int, int, int]]:
    """
    Convert the results of the function get_binary_string_info into a dictionary for convenient reuse.

    :param binary_string_length: the length of the binary string
    :return: dictionary of binary string information
    """

    binary_string_map = dict()

    for i in range(pow(2, binary_string_length)):
        binary_string = bin(i)[2:].zfill(binary_string_length)
        binary_string_info = get_binary_string_features(binary_string)
        binary_string_map[binary_string] = binary_string_info

    return binary_string_map


def get_row_features(binary_string_array: List[str]) -> List[int]:
    """
    Get the row features of the binary string array.

    :param binary_string_array: list of binary strings
    :return: list of row features, including:
        - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
        - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
        - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
    """
    binary_string_map_row = get_binary_string_map(PARITY_COLUMN_COUNT)
    f_binary_string_array = [
        binary_string_map_row[row] for row in binary_string_array
    ]
    max_pooling_f_binary_string_array = [
        max(column) for column in list(zip(*f_binary_string_array))
    ]
    sum_pooling_f_binary_string_array = [
        sum(column) for column in list(zip(*f_binary_string_array))
    ]
    rowwise_or_aggregate = 0
    for row in binary_string_array:
        rowwise_or_aggregate |= int(row, 2)
    rowwise_or_aggregate = bin(rowwise_or_aggregate)[2:].zfill(
        PARITY_COLUMN_COUNT
    )
    f_max_pooling_binary_string_array = list(
        binary_string_map_row[rowwise_or_aggregate]
    )

    return (
            max_pooling_f_binary_string_array
            + sum_pooling_f_binary_string_array
            + f_max_pooling_binary_string_array
    )


def get_column_features(binary_string_array: List[str]) -> List[int]:
    """
    Get the column features of the binary string array.

    :param binary_string_array: list of binary strings
    :return: list of column features, including:
        - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
        - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
        - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
    """
    binary_string_map_column = get_binary_string_map(PARITY_ROW_COUNT)
    f_binary_string_array = [
        binary_string_map_column[column] for column in binary_string_array
    ]
    max_pooling_f_binary_string_array = [
        max(row) for row in list(zip(*f_binary_string_array))
    ]
    sum_pooling_f_binary_string_array = [
        sum(row) for row in list(zip(*f_binary_string_array))
    ]
    columnwise_or_aggregate = 0
    for column in binary_string_array:
        columnwise_or_aggregate |= int(column, 2)
    columnwise_or_aggregate = bin(columnwise_or_aggregate)[2:].zfill(
        PARITY_ROW_COUNT
    )
    f_max_pooling_binary_string_array = list(
        binary_string_map_column[columnwise_or_aggregate]
    )

    return (
            max_pooling_f_binary_string_array
            + sum_pooling_f_binary_string_array
            + f_max_pooling_binary_string_array
    )


def get_parity_features(err_log_parity: int) -> List[int]:
    """
    Get the parity features for the error log parity.

    :param err_log_parity: error log parity
    :return: list of parity features, including:
        - row features: row features of the binary string array
        - column features: column features of the binary string array
    """

    binary_err_log_parity = bin(err_log_parity)[2:].zfill(
        PARITY_ROW_COUNT * PARITY_COLUMN_COUNT
    )

    binary_row_array = [
        binary_err_log_parity[i: i + PARITY_COLUMN_COUNT]
        for i in range(
            0,
            PARITY_ROW_COUNT * PARITY_COLUMN_COUNT,
            PARITY_COLUMN_COUNT,
        )
    ]
    binary_column_array = [
        binary_err_log_parity[i:: PARITY_COLUMN_COUNT]
        for i in range(PARITY_COLUMN_COUNT)
    ]
    row_features = get_row_features(binary_row_array)
    column_features = get_column_features(binary_column_array)

    return row_features + column_features


def process_parity(parity: int) -> Tuple[int, List[int]]:
    """
    Process the parity to get the parity features.

    :param parity: error log parity
    :return: tuple of parity and parity features
    """
    return parity, get_parity_features(parity)


@profile
def get_parity_dict(data_path) -> Dict[int, List[int]]:
    """
    Get the parity dictionary.

    :return: dictionary of parity features
    """
    args_process_file = [(data_path, file_name) for file_name in os.listdir(data_path)]  # 元组列表用于多进程
    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap(process_file, args_process_file),
                total=len(os.listdir(data_path)),
            )
        )
    parity_set = set()
    for i in tqdm(results):
        parity_set.update(i)
    parity_set = sorted(list(parity_set))

    parity_dict = dict()
    with Pool() as pool:
        results = list(
            tqdm(pool.imap(process_parity, parity_set), total=len(parity_set))
        )
    for i, parity_features in results:
        parity_dict[i] = parity_features

    return parity_dict


if __name__ == "__main__":
    # data_path = r"E:\pycharmproject\data_all\folder0\type_A"
    # parity_dict = get_parity_dict(data_path)
    # with open(os.path.join(r"E:\pycharmproject\data_all\folder0\type_A", "parity_features.pkl"), "wb") as pickle_file:
    #     pickle.dump(parity_dict, pickle_file)

    for idx, i in enumerate(os.listdir(r"E:\pycharmproject\data_all")[:2]):
        data_path = rf"E:\pycharmproject\data_all\{i}\type_A"
        parity_dict = get_parity_dict(data_path)
        with open(os.path.join(r"E:\pycharmproject\phase2_data\parity_dict_list1", f"parity_features{idx}.pkl"), "wb") as pickle_file:
            pickle.dump(parity_dict, pickle_file)
        print(f"生成{i}文件夹下的字典parity_features{idx}.pkl")
        del data_path, parity_dict
        gc.collect()
