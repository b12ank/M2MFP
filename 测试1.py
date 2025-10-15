# ^_^
import os
import abc
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, List, Dict, NoReturn, Union, Any, Optional

import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

# Setup dataset and working directories, and display input data information.
# DATASET_DIR = "/kaggle/input//kaggle/input/log-data-raw"
# WORK_DIR = "/kaggle/working"

DATASET_DIR = r"E:\pycharmproject\M2-MFP\data1"
WORK_DIR = r"E:\pycharmproject\M2-MFP\data1\work"

# DATASET_DIR = r"E:\pycharmproject\sn_data_raw"
# WORK_DIR = r"E:\pycharmproject\M2-MFP\data1\work_5000"


# Create necessary directories if they don't exist
os.makedirs(os.path.join(WORK_DIR, "bfse_processed_data"), exist_ok=True)  # bsfe处理后的数据（加入30维bit特征） && 直接用于生成time_point训练集和测试集
os.makedirs(os.path.join(WORK_DIR, "time_patch_processed"), exist_ok=True)  # time_patch聚合时间窗口后的数据
os.makedirs(os.path.join(WORK_DIR, "time_patch_train"), exist_ok=True)  # 训练集，包含正负样本
os.makedirs(os.path.join(WORK_DIR, "time_patch_test"), exist_ok=True)  # 测试集（有聚合时间窗口后的多维特征）
os.makedirs(os.path.join(WORK_DIR, "time_point_train"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "time_point_test"), exist_ok=True)

# 定义时间常量（单位：秒）
ONE_MINUTE = 60  # 一分钟的秒数
ONE_HOUR = 3600  # 一小时的秒数（60秒 * 60分钟）
ONE_DAY = 86400  # 一天的秒数（60秒 * 60分钟 * 24小时）

TIME_WINDOW_SIZE_MAP = {15 * 60: "15m", 60 * 60: "1h", 6 * 3600: "6h"}
TIME_RELATED_LIST = [15 * 60, 60 * 60, 6 * 3600]
FEATURE_EXTRACTION_INTERVAL = 15 * 60


@dataclass
class Config(object):
    """
    配置类, 用于存储和管理程序的配置信息
    包括时间窗口大小、路径设置、日期范围、特征提取间隔等
    """

    # 缺失值填充的默认值
    IMPUTE_VALUE: int = field(default=-1, init=False)

    # 是否使用多进程
    USE_MULTI_PROCESS: bool = field(default=False, init=False)  # init=False 表示该字段不会包含在自动生成的 __init__ 方法中。

    # 如果使用多进程, 并行时 worker 的数量
    WORKER_NUM: int = field(default=4, init=False)

    # 数据路径配置, 分别是原始数据集路径、生成的特征路径、处理后训练集特征路径、处理后测试集特征路径、维修单路径
    raw_data_path: str = "To be filled"
    bsfe_feature_path: str = "To be filled"
    time_patch_processed_path: str = "To be filled"
    time_patch_train_data_path: str = "To be filled"  # time_patch训练集存储路径
    time_patch_test_data_path: str = "To be filled"  # time_patch测试集存储路径
    time_point_train_data_path: str = "To be filled"  # time_point训练集存储路径
    time_point_test_data_path: str = "To be filled"  # time_point测试集存储路径
    ticket_path: str = "To be filled"

    # 日期范围配置
    train_date_range: tuple = ("2024-01-01", "2024-06-01")  # 1704038400-1717171200
    test_data_range: tuple = ("2024-06-01", "2024-08-01")


def get_use_high_order_bit_level_features_names() -> List[str]:
    """
    Get the high order bit-level feature names. High order bit-level feature are the features that are calculated based
    on DQ-Beat Matrix.
    无输入
    :return: A list of high order bit-level feature names.    输出30个特征名组成的列表
    """
    high_order_bit_level_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for F_function in [
                "bit_count",
                "bit_min_interval",
                "bit_max_interval",
                "bit_max_consecutive_length",
                "bit_consecutive_length",
            ]:
                high_order_bit_level_feature_names.append(
                    f"dq_beat_{row_column}wise_{G_function}_{F_function}"
                )
    return high_order_bit_level_feature_names


class BSFE_DQBeatMatrix:
    """
    Calculate the DQ-Beat Matrix features for the bit-level CE info.
    """
    # The shape of DQ-Beat Matrix, for DDR4 memory, can be represented as an 8-row by 4-column binary matrix
    DQBeatMatrix_ROW_COUNT = 8
    DQBeatMatrix_COLUMN_COUNT = 4

    def __init__(self, config: Config):
        self.binary_string_map_row = self.get_binary_string_map(
            self.DQBeatMatrix_COLUMN_COUNT
        )
        self.binary_string_map_column = self.get_binary_string_map(
            self.DQBeatMatrix_ROW_COUNT
        )
        self.config = config
        self.bsfe_dict = None  # 延迟初始化

    def get_binary_string_features(
            self, binary_string: str
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

        indices = self.indices_of_ones(binary_string)

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
        )  # 给一个字符串，返回一个五维特征

    def get_binary_string_map(
            self,
            binary_string_length: int,
    ) -> Dict[str, Tuple[int, int, int, int, int]]:
        """
        Convert the results of the function get_binary_string_info into a dictionary for convenient reuse.

        :param binary_string_length: the length of the binary string
        :return: dictionary of binary string information
        """

        binary_string_map = dict()

        for i in range(pow(2, binary_string_length)):  # 循环2的binary_string_length平方次
            binary_string = bin(i)[2:].zfill(binary_string_length)
            binary_string_info = self.get_binary_string_features(binary_string)  # 一维bsfe 转化为5个特征
            binary_string_map[binary_string] = binary_string_info
            # 设binary_string_length取8，计算所有8位二进制字符串bsfe后对应的五个特征
        return binary_string_map  # 返回一个字典，键是二进制字符串，值是一维bsfe后对应的5个特征

    @staticmethod
    def indices_of_ones(binary_string: str) -> List[int]:
        """
        Get the indices of the ones in a binary string.

        :param binary_string: binary string
        :return: list of indices of ones
        """
        # 获得一个二进制字符串中值为1的下标列表
        return [index for index, char in enumerate(binary_string) if char == "1"]

    def get_row_features(self, binary_string_array: List[str]) -> List[int]:
        # 一个矩阵有多个行，所以输入的binary_string_array是一个字符串列表
        """
        Get the row features of the binary string array. We define `f_` as 1d_BSFE.

        :param binary_string_array: list of binary strings
        :return: list of row features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        # Reduction-then-Aggregation
        f_binary_string_array = [
            self.binary_string_map_row[row] for row in binary_string_array
        ]  # reduction
        max_pooling_f_binary_string_array = [
            max(column) for column in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(column) for column in list(zip(*f_binary_string_array))
        ]

        # Aggregation-then-Reduction
        rowwise_or_aggregate = 0
        for row in binary_string_array:
            rowwise_or_aggregate |= int(row, 2)
        rowwise_or_aggregate = bin(rowwise_or_aggregate)[2:].zfill(
            self.DQBeatMatrix_COLUMN_COUNT
        )
        f_max_pooling_binary_string_array = list(
            self.binary_string_map_row[rowwise_or_aggregate]
        )

        return (
                max_pooling_f_binary_string_array
                + sum_pooling_f_binary_string_array
                + f_max_pooling_binary_string_array
        )

    def get_column_features(self, binary_string_array: List[str]) -> List[int]:
        """
        Get the column features of the binary string array. We define `f_` as 1d_BSFE.

        :param binary_string_array: list of binary strings
        :return: list of column features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        # Reduction-then-Aggregation
        f_binary_string_array = [
            self.binary_string_map_column[column] for column in binary_string_array
        ]
        max_pooling_f_binary_string_array = [
            max(row) for row in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(row) for row in list(zip(*f_binary_string_array))
        ]

        # Aggregation-then-Reduction
        columnwise_or_aggregate = 0
        for column in binary_string_array:
            columnwise_or_aggregate |= int(column, 2)
        columnwise_or_aggregate = bin(columnwise_or_aggregate)[2:].zfill(
            self.DQBeatMatrix_ROW_COUNT
        )
        f_max_pooling_binary_string_array = list(
            self.binary_string_map_column[columnwise_or_aggregate]
        )

        return (
                max_pooling_f_binary_string_array
                + sum_pooling_f_binary_string_array
                + f_max_pooling_binary_string_array
        )

    def get_high_order_bit_level_features(
            self, err_log_dq_beat_matrix: int
    ) -> List[int]:
        """
        Get the DQ-Beat Matrix features for the bit-level CE.

        :param err_log_dq_beat_matrix: error log DQ-Beat Matrix
        :return: list of DQ-Beat Matrix features, including:
            - row features: row features of the binary string array
            - column features: column features of the binary string array
        """

        # [2:]: 切片操作，去掉前两个字符 "0b"
        # .zfill(32): 用 0 在左侧填充，使字符串长度为 32 位
        binary_err_log_dq_beat_matrix = bin(err_log_dq_beat_matrix)[2:].zfill(
            self.DQBeatMatrix_ROW_COUNT * self.DQBeatMatrix_COLUMN_COUNT
        )

        binary_row_array = [
            binary_err_log_dq_beat_matrix[i: i + self.DQBeatMatrix_COLUMN_COUNT]
            for i in range(
                0,
                self.DQBeatMatrix_ROW_COUNT * self.DQBeatMatrix_COLUMN_COUNT,
                self.DQBeatMatrix_COLUMN_COUNT,
            )
        ]
        # 列表表示矩阵行：['0000', '0000', '0000', '0000', '0000', '0000', '0000', '0001']
        binary_column_array = [
            binary_err_log_dq_beat_matrix[i:: self.DQBeatMatrix_COLUMN_COUNT]
            for i in range(self.DQBeatMatrix_COLUMN_COUNT)
        ]
        row_features = self.get_row_features(binary_row_array)  # 获得这个矩阵的15维行特征
        column_features = self.get_column_features(binary_column_array)

        return row_features + column_features

    def process_file(self, file: str) -> List[int]:
        """
        Process the file to get the unique CE.

        :param file: file name
        :return: list of unique error log DQ-Beat Matrix
        """

        data = pd.read_feather(os.path.join(self.config.raw_data_path, file))
        return data.RetryRdErrLogParity.dropna().astype(np.int64).unique().tolist()  # 返回一个unique的parity列表

    def process_dq_beat_matrix(self, dq_beat_matrix: int) -> Tuple[int, List[int]]:
        """
        Process the dq_beat_matrix to get the high-order bit-level features.

        :param dq_beat_matrix: error log dq_beat_matrix
        :return: tuple of dq_beat_matrix and dq_beat_matrix features
        """
        return dq_beat_matrix, self.get_high_order_bit_level_features(dq_beat_matrix)

    parity_list = []

    def get_high_order_bit_level_features_dict(self) -> Dict[int, List[int]]:
        """
        Get the dq_beat_matrix dictionary.

        :return: dictionary of high-order bit-level features
        """
        print(f"读取{len(os.listdir(self.config.raw_data_path))}条原始数据")
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_file, os.listdir(self.config.raw_data_path)),
                    total=len(os.listdir(self.config.raw_data_path)),
                )
            )  # results是所有文件parity列表的列表
        dq_beat_matrix_set = set()  # 用集合获取unique的parity 并转化成升序排列的列表
        for i in results:
            dq_beat_matrix_set.update(i)
        dq_beat_matrix_set = sorted(list(dq_beat_matrix_set))

        _high_order_bit_level_features_dict = dict()

        print(f"开始生成{len(dq_beat_matrix_set)}个不同parity对应构造特征字典：")
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_dq_beat_matrix, dq_beat_matrix_set),
                    total=len(dq_beat_matrix_set),
                )
            )
        for i, _high_order_bit_level_features in results:
            _high_order_bit_level_features_dict[i] = _high_order_bit_level_features
        print("已获取parity对应特征字典")
        return _high_order_bit_level_features_dict  # 获得一个字典，键是parity，值是30维的特征

    '''新加入初步bsfe处理目录下所有文件的函数'''

    def bsfe_process_single_sn(self, sn_file):
        df = pd.read_feather(os.path.join(self.config.raw_data_path, sn_file))
        raw_df = df.sort_values(by="LogTime").reset_index(drop=True)
        # 提取需要的列并初始化 processed_df
        processed_df = raw_df[
            [
                "LogTime",
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",
                "RetryRdErrLogParity",
            ]
        ].copy()
        # deviceID 可能存在缺失值, 填充缺失值
        processed_df["deviceID"] = (
            processed_df["deviceID"].fillna(-1).astype(int)
        )

        # 将 error_type 转换为独热编码
        processed_df["error_type_is_READ_CE"] = (
                raw_df["error_type_full_name"] == "CE.READ"
        ).astype(int)
        processed_df["error_type_is_SCRUB_CE"] = (
                raw_df["error_type_full_name"] == "CE.SCRUB"
        ).astype(int)

        columns_to_process = get_use_high_order_bit_level_features_names()
        for feature_name in columns_to_process:
            processed_df[feature_name] = 0

        err_log_parity_array = (
            processed_df["RetryRdErrLogParity"]
            .fillna(0)
            .replace("", 0)
            .astype(np.int64)  # 转换为 np.int64, 此处如果为 int 会溢出，缺失值NaN和空字符串都填充0
            .values  # series 转numpy
        )

        for idx, parity_value in enumerate(err_log_parity_array):
            feature_value_list = self.bsfe_dict[parity_value]
            for num, feature in enumerate(columns_to_process):
                processed_df.at[idx, feature] = feature_value_list[num]
                # processed_df是经bsfe后加入30维特征的数据集， 共39列特征
        # feather.write_dataframe(processed_df, os.path.join(self.config.bsfe_feature_path, sn_file))
        processed_df.to_feather(os.path.join(self.config.bsfe_feature_path, sn_file))

    def bsfe_process_all_sn(self):
        sn_files_list = os.listdir(self.config.raw_data_path)[:5000]  # 在这里控制读取数据数量 [:5000]

        if self.bsfe_dict is None:
            self.bsfe_dict = self.get_high_order_bit_level_features_dict()  # 获取parity对应bsfe特征列表的字典，在这里才真正给实例属性bsfe_dict赋值

        if config.USE_MULTI_PROCESS is True:
            print(f"对{len(sn_files_list)}个文件进行bsfe处理(多进程)")
            with Pool(processes=4) as pool:  # 创建包含4个工作进程的进程池
                list(tqdm(pool.imap_unordered(self.bsfe_process_single_sn, sn_files_list), total=len(sn_files_list)))
        else:
            print(f"对{len(sn_files_list)}个文件进行bsfe处理(未使用多进程)")
            for sn_file in tqdm(sn_files_list, total=len(sn_files_list)):
                self.bsfe_process_single_sn(sn_file)


class Time_Patch_Process:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def get_bsfe_dict():
        bsfe = BSFE_DQBeatMatrix()
        return bsfe.get_high_order_bit_level_features_dict()

    def unique_num_filtered(self, input_array: np.ndarray) -> int:
        """
        Deduplicate the input array, remove elements with the value IMPUTE_VALUE, and count the remaining unique elements.
        去重
        :param input_array: Input array
        :return: Number of unique elements after filtering       输出去重后的元素个数
        """

        unique_array = np.unique(input_array)  # 取整个数组中所有不同元素组成新数组
        return len(unique_array) - int(self.config.IMPUTE_VALUE in unique_array)

    @staticmethod
    def get_max_sum_avg_values(
            input_values: pd.Series, valid_value_count: int
    ) -> Tuple[int, int, int]:
        """
        Get the max, sum, and average value of the input position.

        :param input_values: The input position.
        :param valid_value_count: The valid position count.
        """
        if valid_value_count == 0:  # 没有有效值（全零）->输出(0,0,0)
            return 0, 0, 0
        max_value = input_values.values.max()
        sum_value = input_values.values.sum()
        average_value = round(np.divide(sum_value, valid_value_count), 2)
        return max_value, sum_value, average_value

    @staticmethod
    def calculate_ce_storm_count(log_times: pd.Series) -> int:
        """
        Calculate the number of CE storms.
        See https://github.com/hwcloud-RAS/SmartHW/blob/main/competition_starterkit/baseline_en.py for more details.

        CE storm definition:
        - Adjacent CE logs: If the time interval between two CE logs' LogTime is < 60s, they are considered adjacent logs.
        - If the number of adjacent logs exceeds 10, it is counted as one CE storm (note: if the number of adjacent logs
        continues to grow beyond 10, it is still counted as one CE storm).

        :param log_times: List of log LogTimes
        :return: Number of CE storms
        """

        ce_storm_interval_seconds = 60  # CE 风暴的时间间隔阈值 60s内连续发生算一次相邻ce
        ce_storm_count_threshold = 10  # CE 风暴的数量阈值 连续10个相邻ce算一次ce风暴
        log_times = log_times.sort_values().reset_index(drop=True)
        ce_storm_count = 0
        consecutive_count = 0

        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count > ce_storm_count_threshold:
                ce_storm_count += 1
                consecutive_count = 0

        return ce_storm_count

    def get_aggregated_high_order_bit_level_features(
            self, window_df: pd.DataFrame,
    ) -> Dict[str, int]:
        """
        Get the aggregated high order bit-level features.

        :param window_df: Data within the time window
        :return: Dictionary of aggregated high order bit-level features
        """
        aggregated_high_order_bit_level_features = dict()

        aggregated_high_order_bit_level_features["error_bit_count"] = window_df[
            "dq_beat_rowwise_sum_pooling_F_bit_count"
        ].values.sum()

        # aggregated_high_order_bit_level_features["all_valid_err_log_count"] = window_df[
        #     "retry_log_is_valid"     # 可能是该条日志是否有效？
        # ].values.sum()          # ?????????????????????????????????
        '''修改代替上面代码'''
        aggregated_high_order_bit_level_features["all_valid_err_log_count"] = len(window_df)

        columns_to_process = get_use_high_order_bit_level_features_names()

        for col in columns_to_process:
            max_col, sum_col, avg_col = self.get_max_sum_avg_values(
                window_df[col],
                aggregated_high_order_bit_level_features["all_valid_err_log_count"],  # ??????
            )
            aggregated_high_order_bit_level_features[f"max_{col}"] = max_col
            aggregated_high_order_bit_level_features[f"sum_{col}"] = sum_col
            aggregated_high_order_bit_level_features[f"avg_{col}"] = avg_col  # 这个特征有啥用？

        dq_counts = dict()
        burst_counts = dict()
        for i in zip(
                window_df["dq_beat_rowwise_F_max_pooling_bit_count"].values,  # 几个DQ有错
                window_df["dq_beat_columnwise_F_max_pooling_bit_count"].values,  # 几个beat有错
        ):  # i是每行两列特征组成的元组
            dq_counts[i[0]] = dq_counts.get(i[0], 0) + 1  # 如果键i[0]存在于字典中，返回对应的值,如果键不存在，返回默认值0
            #  {0:,1:,2:,3:,4:}字典保存不同DQ错误数分别有几次
            burst_counts[i[1]] = burst_counts.get(i[1], 0) + 1

        for dq in [1, 2, 3, 4]:
            aggregated_high_order_bit_level_features[f"dq_count={dq}"] = dq_counts.get(
                dq, 0
            )
        for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
            aggregated_high_order_bit_level_features[
                f"burst_count={burst}"
            ] = burst_counts.get(burst, 0)

        aggregated_high_order_bit_level_features = {
            f"{key}": value
            for key, value in aggregated_high_order_bit_level_features.items()
        }  # 确保键是字符串?

        return aggregated_high_order_bit_level_features  # 返回字典{特征：值}   {"dq_count=1": 1,……}

    def get_spatio_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract spatial features including fault modes and counts of simultaneous row/column faults.
        See https://github.com/hwcloud-RAS/SmartHW/blob/main/competition_starterkit/baseline_en.py for more details.

        Fault mode definitions:
          - fault_mode_others: Other faults, where multiple devices exhibit faults.
          - fault_mode_device: Device faults, where multiple banks within the same device exhibit faults.
          - fault_mode_bank: Bank faults, where multiple rows within the same bank exhibit faults.
          - fault_mode_row: Row faults, where multiple cells in the same row exhibit faults.
          - fault_mode_column: Column faults, where multiple cells in the same column exhibit faults.
          - fault_mode_cell: Cell faults, where multiple cells with the same ID exhibit faults.
          - fault_row_num: Number of rows with simultaneous row faults.
          - fault_column_num: Number of columns with simultaneous column faults.

        :param window_df: Pandas DataFrame containing data within a specific time window.
        :return: A dictionary mapping spatial feature names to their integer values.
        """

        spatio_features = {
            "fault_mode_others": 0,
            "fault_mode_device": 0,
            "fault_mode_bank": 0,
            "fault_mode_row": 0,
            "fault_mode_column": 0,
            "fault_mode_cell": 0,
            "fault_row_num": 0,
            "fault_column_num": 0,
        }

        # Determine fault mode based on the number of faulty devices, banks, rows, columns, and cells
        if self.unique_num_filtered(window_df["deviceID"].values) > 1:  # 故障device数量>1，others错误模式设1
            spatio_features["fault_mode_others"] = 1
        elif self.unique_num_filtered(window_df["BankId"].values) > 1:
            spatio_features["fault_mode_device"] = 1
        elif (
                self.unique_num_filtered(window_df["ColumnId"].values) > 1
                and self.unique_num_filtered(window_df["RowId"].values) > 1
        ):
            spatio_features["fault_mode_bank"] = 1
        elif self.unique_num_filtered(window_df["ColumnId"].values) > 1:
            spatio_features["fault_mode_row"] = 1
        elif self.unique_num_filtered(window_df["RowId"].values) > 1:
            spatio_features["fault_mode_column"] = 1
        elif self.unique_num_filtered(window_df["CellId"].values) == 1:
            spatio_features["fault_mode_cell"] = 1

        # Record column address information for the same row
        row_pos_dict = {}
        # Record row address information for the same column
        col_pos_dict = {}

        for device_id, bank_id, row_id, column_id in zip(
                window_df["deviceID"].values,
                window_df["BankId"].values,
                window_df["RowId"].values,
                window_df["ColumnId"].values,
        ):
            current_row = "_".join([str(pos) for pos in [device_id, bank_id, row_id]])
            current_col = "_".join([str(pos) for pos in [device_id, bank_id, column_id]])
            row_pos_dict.setdefault(current_row, [])  # 字典初始化，如果键存在，返回对应的值；如果键不存在，设置key: default并返回default
            col_pos_dict.setdefault(current_col, [])
            row_pos_dict[current_row].append(column_id)  # 字典记录某一行哪几列有错误{"1_2_1":[10,20]} :1号device的2号bank的第一行在第10列，20列有错误
            col_pos_dict[current_col].append(row_id)

        for row in row_pos_dict:
            if self.unique_num_filtered(np.array(row_pos_dict[row])) > 1:
                spatio_features["fault_row_num"] += 1  # 计算行错误数量
        for col in col_pos_dict:
            if self.unique_num_filtered(np.array(col_pos_dict[col])) > 1:
                spatio_features["fault_column_num"] += 1

        return spatio_features

    def process_single_sn(self, sn_file) -> NoReturn:
        """
        Process a single SN file to extract and aggregate features based on time windows,
        and then write the combined feature data to a feather file.

        :param sn_file: The filename of the SN file to process.
        """

        new_df = pd.read_feather(
            os.path.join(self.config.bsfe_feature_path, sn_file)
        )

        new_df["time_index"] = new_df["LogTime"] // FEATURE_EXTRACTION_INTERVAL  # 因为有余数的原因，time_index相同的ce事件属于同一个15min时间窗口
        new_df["CellId"] = (
                new_df["RowId"].astype(str) + "_" + new_df["ColumnId"].astype(str)
        )  # 加两列特征

        # 找到同一时间窗口内最晚发生的那条ce事件的时间戳作为时间窗口结束时间
        grouped = new_df.groupby("time_index")["LogTime"].max()  # series格式
        window_end_time_list = grouped.tolist()

        combined_dict_list = []
        for end_time in window_end_time_list:
            combined_dict = {}
            window_df = new_df[
                (new_df["LogTime"] <= end_time)
                & (new_df["LogTime"] > end_time - 6 * 3600 - 15 * 60)
                ]  # 取六小时十五分钟内的数据？ --->统一去重，减少去重次数？
            combined_dict["ReportTime"] = window_df["LogTime"].max()  # ？  去重前的最晚时间
            window_df = window_df.drop_duplicates(
                subset=["deviceID", "BankId", "RowId", "ColumnId", "RetryRdErrLogParity"],
                keep="first",
            )  # 去重相同subset的行
            combined_dict["LogTime"] = window_df["LogTime"].max()  # 去重后的最晚时间可能会变，因为去重保留组合的第一次出现
            for time_window_size in TIME_RELATED_LIST[::-1]:  # 针对不同时间间隔 6h,1h,15min
                end_logtime_of_filtered_window_df = window_df["LogTime"].max()
                window_df = window_df[
                    window_df["LogTime"]
                    >= end_logtime_of_filtered_window_df - time_window_size
                    ]

                temporal_features = dict()
                temporal_features["read_ce_log_num"] = window_df[
                    "error_type_is_READ_CE"
                ].values.sum()
                temporal_features["scrub_ce_log_num"] = window_df[
                    "error_type_is_SCRUB_CE"
                ].values.sum()
                temporal_features["all_ce_log_num"] = len(window_df)
                temporal_features["log_happen_frequency"] = (
                    time_window_size / temporal_features["all_ce_log_num"]
                    if temporal_features["all_ce_log_num"]
                    else 0
                )
                temporal_features["ce_storm_count"] = self.calculate_ce_storm_count(
                    window_df["LogTime"]
                )

                aggregated_high_order_bit_level_features = (
                    self.get_aggregated_high_order_bit_level_features(window_df)
                )
                spatio_features = self.get_spatio_features(window_df)

                combined_dict.update(  # 不同时间窗口分别聚合特征保存在字典中
                    {
                        f"{key}_{TIME_WINDOW_SIZE_MAP[time_window_size]}": value
                        for d in [
                        temporal_features,
                        spatio_features,
                        aggregated_high_order_bit_level_features,
                    ]
                        for key, value in d.items()  # key是特征名，value是值
                    }
                )
            combined_dict_list.append(combined_dict)  # 字典的列表
        combined_df = pd.DataFrame(combined_dict_list)  # 生成多行所有时间窗口聚合的特征    353维？？
        # feather.write_dataframe(
        #     combined_df, os.path.join(self.config.time_patch_processed_path, sn_file)
        # )
        combined_df.to_feather(os.path.join(self.config.time_patch_processed_path, sn_file))

    def process_all_sn(self):
        sn_files_list = os.listdir(self.config.bsfe_feature_path)  # 在这里控制读取数据数量 [:5000]
        if config.USE_MULTI_PROCESS is True:
            print(f"开始对bsfe_processed_data路径下的{len(sn_files_list)}个文件进行time_patch处理(多进程)")
            with Pool(processes=4) as pool:  # 创建包含4个工作进程的进程池
                list(tqdm(pool.imap_unordered(self.process_single_sn, sn_files_list), total=len(sn_files_list)))
        else:
            print(f"开始对bsfe_processed_data路径下的{len(sn_files_list)}个文件进行time_patch处理(未使用多进程)")
            for sn_file in sn_files_list:
                self.process_single_sn(sn_file)


class DataGenerator(metaclass=abc.ABCMeta):
    """
    数据生成器基类, 用于生成训练和测试数据
    """

    # 数据分块大小, 用于分批处理数据
    CHUNK_SIZE = 200

    def __init__(self, config: Config, data_type: str = "time_patch"):
        """
        初始化数据生成器

        :param config: 配置类实例, 包含路径、日期范围等信息
        :param data_type: 数据集类型，'time_patch'或者'time_point'
        """

        self.config = config
        self.data_type = data_type

        if self.data_type == "time_patch":
            self.feature_path = self.config.time_patch_processed_path
            self.train_data_path = self.config.time_patch_train_data_path
            self.test_data_path = self.config.time_patch_test_data_path
        elif data_type == "time_point":
            self.feature_path = self.config.bsfe_feature_path
            self.train_data_path = self.config.time_point_train_data_path
            self.test_data_path = self.config.time_point_test_data_path

        self.ticket_path = self.config.ticket_path

        # 将日期范围转换为时间戳
        self.train_start_date = self._datetime_to_timestamp(
            self.config.train_date_range[0]
        )
        self.train_end_date = self._datetime_to_timestamp(
            self.config.train_date_range[1]
        )
        self.test_start_date = self._datetime_to_timestamp(
            self.config.test_data_range[0]
        )
        self.test_end_date = self._datetime_to_timestamp(self.config.test_data_range[1])

        ticket = pd.read_csv(self.ticket_path)
        ticket = ticket[ticket["alarm_time"] <= self.train_end_date]
        self.ticket = ticket
        self.ticket_sn_map = {
            sn: sn_t
            for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
        }

        os.makedirs(self.train_data_path, exist_ok=True)
        os.makedirs(self.test_data_path, exist_ok=True)

    @staticmethod
    def concat_in_chunks(chunks: List) -> Union[pd.DataFrame, None]:
        """
        将 chunks 中的 DataFrame 进行拼接

        :param chunks: DataFrame 列表
        :return: 拼接后的 DataFrame, 如果 chunks 为空则返回 None
        """

        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            return pd.concat(chunks)
        return None

    def parallel_concat(
            self, results: List, chunk_size: int = CHUNK_SIZE
    ) -> Union[pd.DataFrame, None]:
        """
        并行化的拼接操作, 可以视为 concat_in_chunks 的并行化版本

        :param results: 需要拼接的结果列表
        :param chunk_size: 每个 chunk 的大小
        :return: 拼接后的 DataFrame
        """

        chunks = [
            results[i: i + chunk_size] for i in range(0, len(results), chunk_size)
        ]

        # 使用多进程并行拼接
        worker_num = self.config.WORKER_NUM
        with Pool(worker_num) as pool:
            concatenated_chunks = pool.map(self.concat_in_chunks, chunks)

        return self.concat_in_chunks(concatenated_chunks)

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        """
        将 %Y-%m-%d 格式的日期转换为时间戳

        :param date: 日期字符串
        :return: 时间戳
        """

        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def _get_data(self) -> pd.DataFrame:
        """
        获取 feature_path 下的所有数据, 并进行处理

        :return: 处理后的数据
        """

        file_list = os.listdir(self.feature_path)
        file_list = [x for x in file_list if x.endswith(".feather")]
        file_list.sort()
        # print(f"找到 {len(file_list)} 个特征文件")  # 添加调试信息
        #
        # if not file_list:
        #     print("警告：没有找到任何特征文件！")
        # a = file_list[0]
        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._process_file, file_list),
                        total=len(file_list),
                        desc="Processing files",
                    )
                )
            data_all = self.parallel_concat(results)
        else:
            data_all = []
            data_chunk = []
            for i in tqdm(range(len(file_list)), desc="Processing files"):
                data = self._process_file(file_list[i])
                if data is not None:
                    data_chunk.append(data)
                if len(data_chunk) >= self.CHUNK_SIZE:
                    data_all.append(self.concat_in_chunks(data_chunk))
                    data_chunk = []
            if data_chunk:
                data_all.append(pd.concat(data_chunk))

            data_all = pd.concat(data_all)

        return data_all

    @abc.abstractmethod
    def _process_file(self, sn_file):
        """
        处理单个文件, 子类需要实现该方法

        :param sn_file: 文件名
        """

        raise NotImplementedError("Subclasses should implement this method")

    @abc.abstractmethod
    def generate_and_save_data(self):
        """
        生成并保存数据, 子类需要实现该方法
        """

        raise NotImplementedError("Subclasses should implement this method")


class PositiveDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取正样本数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if self.ticket_sn_map.get(sn_name):
            # 设正样本的时间范围为维修单时间的前 30 天
            end_time = self.ticket_sn_map.get(sn_name)
            start_time = end_time - 30 * ONE_DAY

            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            data["label"] = 1

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # 如果 SN 名称不在维修单中, 则返回 None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存正样本数据
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "positive_train.feather")
        )


class NegativeDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取负样本数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if not self.ticket_sn_map.get(sn_name):
            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))

            # 设负样本的时间范围为某段连续的 30 天
            # !!!!!!!!!!!!!!!!!! 这里只取4月1日到5月1日的数据作为负样本的取样时间，只有落在这一部分时间里的数据才生成负样本
            end_time = self.train_end_date - 30 * ONE_DAY
            start_time = self.train_end_date - 60 * ONE_DAY

            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            if data.empty:
                return None
            data["label"] = 0

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # 如果 SN 名称在维修单中, 则返回 None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存负样本数据
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "negative_train.feather")
        )


class TestDataGenerator(DataGenerator):
    @staticmethod
    def _split_dataframe(df: pd.DataFrame, chunk_size: int = 2000000):
        """
        将 DataFrame 按照 chunk_size 进行切分

        :param df: 拆分前的 DataFrame
        :param chunk_size: chunk 大小
        :return: 切分后的 DataFrame, 每次返回一个 chunk
        """

        for start in range(0, len(df), chunk_size):
            yield df[start: start + chunk_size]

    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取测试数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]

        # 读取特征文件, 并过滤出测试时间范围内的数据
        data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
        data = data[data["LogTime"] >= self.test_start_date]
        data = data[data["LogTime"] <= self.test_end_date]
        if data.empty:
            return None

        index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存测试数据
        """

        data_all = self._get_data()
        for index, chunk in enumerate(self._split_dataframe(data_all)):
            feather.write_dataframe(
                chunk, os.path.join(self.test_data_path, f"res_{index}.feather")
            )


class MFPmodel(object):
    """
    Memory Failure Prediction 模型类
    用lightGBM模型，用time_point数据集进行训练
    """

    def __init__(self, config: Config):
        """
        初始化模型类

        :param config: 配置类实例, 包含训练和测试数据的路径等信息
        """

        self.train_data_path = config.time_patch_train_data_path
        self.test_data_path = config.time_patch_test_data_path
        self.model_params = {
            "learning_rate": 0.02,
            "n_estimators": 500,
            "max_depth": 8,
            "num_leaves": 20,
            "min_child_samples": 20,
            "verbose": 1,
        }
        self.model = LGBMClassifier(**self.model_params)

    def load_train_data(self) -> NoReturn:
        """
        加载训练数据
        """

        self.train_pos = feather.read_dataframe(
            os.path.join(self.train_data_path, "positive_train.feather")
        )
        self.train_neg = feather.read_dataframe(
            os.path.join(self.train_data_path, "negative_train.feather")
        )

    def train(self) -> NoReturn:
        """
        训练模型
        """

        train_all = pd.concat([self.train_pos, self.train_neg])
        train_all.drop("LogTime", axis=1, inplace=True)
        train_all = train_all.sort_index(axis=1)

        self.model.fit(train_all.drop(columns=["label"]), train_all["label"])

    def predict_proba(self) -> Dict[str, List]:
        """
        预测测试数据每个样本被预测为正类的概率, 并返回结果

        :return: 每个样本被预测为正类的概率, 结果是一个 dict, key 为 sn_name, value 为预测结果列表
        """
        result = {}
        for file in os.listdir(self.test_data_path):
            test_df = feather.read_dataframe(os.path.join(self.test_data_path, file))
            test_df["sn_name"] = test_df.index.get_level_values(0)
            test_df["log_time"] = test_df.index.get_level_values(1)

            test_df = test_df[self.model.feature_name_]
            predict_result = self.model.predict_proba(test_df)

            index_list = list(test_df.index)
            for i in tqdm(range(len(index_list))):
                p_s = predict_result[i][1]

                # 过滤低概率样本, 降低预测结果占用的内存
                if p_s < 0.1:
                    continue

                sn = index_list[i][0]
                sn_t = datetime.fromtimestamp(index_list[i][1])
                result.setdefault(sn, [])
                result[sn].append((sn_t, p_s))
        return result

    def predict(self, threshold: int = 0.5) -> Dict[str, List]:
        """
        获得特定阈值下的预测结果

        :param threshold: 阈值, 默认为 0.5
        :return: 按照阈值筛选后的预测结果, 结果是一个字典, key 为 sn_name, value 为时间戳列表
        """

        # 获取预测概率结果
        result = self.predict_proba()

        # 将预测结果按照阈值进行筛选
        result = {
            sn: [int(sn_t.timestamp()) for sn_t, p_s in pred_list if p_s >= threshold]
            for sn, pred_list in result.items()
        }

        # 过滤空预测结果, 并将预测结果按照时间进行排序
        result = {
            sn: sorted(pred_list) for sn, pred_list in result.items() if pred_list
        }

        return result


class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    """
    Custom decision tree classifier that uses a custom splitting criterion based on SN sets.
    """

    def __init__(self, all_sns_set, pos_sn_set, features, *args, **kwargs):
        """
        Initialize the CustomDecisionTreeClassifier. Accepts any parameters supported by the parent
        DecisionTreeClassifier.
        """
        super().__init__(*args, **kwargs)
        self.all_sns_set = all_sns_set
        self.pos_sn_set = pos_sn_set
        self.features = features

    @staticmethod
    def _gini(y: List[int]) -> float:
        """
        Calculate the Gini impurity for a set of labels.

        :param y: Labels.
        :return: Gini impurity.
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    @staticmethod
    def _weighted_gini(y_left: List[int], y_right: List[int]) -> float:
        """
        Calculate the weighted Gini impurity for a binary split.

        :param y_left: Labels of the left split.
        :param y_right: Labels of the right split.
        :return: Weighted Gini impurity.
        """
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        return (n_left / n_total) * CustomDecisionTreeClassifier._gini(y_left) + (
                n_right / n_total
        ) * CustomDecisionTreeClassifier._gini(y_right)

    def _split_criterion(self, X: np.ndarray) -> Tuple[Optional[int], Optional[Any]]:
        """
        Determine the best split criterion for the data X. Iterates over all features (except the last column which is
        used for serial numbers) and all unique threshold values to find the split that minimizes the weighted Gini
        impurity.

        :param X: Data array where the last column contains serial numbers.
        :return: A tuple (feature_index, threshold) representing the best split.
                 Returns (None, None) if no valid split is found.
        """
        best_gini = float("inf")
        best_split = (None, None)
        n_samples, n_features = X.shape
        n_features -= 1  # The last column is assumed to be the serial number

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] == threshold
                left_sns_set = set(X[left_mask][:, -1])
                right_sns_set = self.all_sns_set - left_sns_set
                left_mask = np.array([x in left_sns_set for x in X[:, -1]])
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = [1 if i in self.pos_sn_set else 0 for i in left_sns_set]
                y_right = [1 if i in self.pos_sn_set else 0 for i in right_sns_set]

                gini = CustomDecisionTreeClassifier._weighted_gini(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, threshold)

        return best_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomDecisionTreeClassifier":
        """
        Train the model.

        :param X: Training data.
        :param y: Training labels.
        :return: self
        """
        self.tree_ = self._build_tree(X, y)
        return self

    def _build_tree(
            self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively build the decision tree.

        Stopping conditions:
          1. If all samples belong to the same class.
          2. If the maximum depth is reached.
          3. If no valid split is found.

        :param X: Data array for the current node.
        :param y: Labels corresponding to X.
        :param depth: Current depth of the tree.
        :return: A dictionary representing the tree node.
        """
        # Stopping condition 1: If all samples belong to the same class
        if len(np.unique(y)) == 1:
            label = len(set(X[:, -1]) & self.pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & self.pos_sn_set),
                1: len(set(X[:, -1]) & self.pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Stopping condition 2: If maximum depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            if len(y) == 0:
                return {"label": 0, "class_counts": {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & self.pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & self.pos_sn_set),
                1: len(set(X[:, -1]) & self.pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Find the best split
        feature, threshold = self._split_criterion(X)

        # Stopping condition 3: If no valid split is found
        if feature is None:
            if len(y) == 0:
                return {"label": 0, "class_counts": {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & self.pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & self.pos_sn_set),
                1: len(set(X[:, -1]) & self.pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Recursively build left and right subtrees
        left_mask = X[:, feature] == threshold
        left_sns_set = set(X[left_mask][:, -1])
        right_sns_set = self.all_sns_set - left_sns_set
        left_mask = np.array([x in left_sns_set for x in X[:, -1]])
        right_mask = ~left_mask
        left_mask = X[:, feature] == threshold

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given samples.

        :param X: Data array of samples.
        :return: An array of predicted labels.
        """
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree: Dict[str, Any]) -> int:
        """
        Predict the label for a single sample.

        :param x: A single sample.
        :param tree: The current node of the decision tree.
        :return: The predicted label.
        """
        if "label" in tree:  # Leaf node
            return tree["label"]
        feature, threshold = tree["feature"], tree["threshold"]
        if (
                x[feature] == threshold
        ):  # If the sample's feature value equals the threshold
            return self._predict_one(x, tree["left"])
        else:  # If the sample's feature value is not equal to the threshold
            return self._predict_one(x, tree["right"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability estimates for each sample.

        :param X: Data array of samples.
        :return: An array of probability distributions for each sample.
        """
        return np.array([self._predict_proba_one(x, self.tree_) for x in X])

    def _predict_proba_one(self, x: np.ndarray, tree: Dict[str, Any]) -> List[float]:
        """
        Return the probability distribution for a single sample.

        :param x: A single sample.
        :param tree: The current node of the decision tree.
        :return: A list with the probability for each class.
        """
        if "class_counts" in tree:  # Leaf node
            class_counts = tree["class_counts"]
            total_samples = sum(
                class_counts.values()
            )  # Total number of samples in the leaf
            proba = [
                class_counts[0] / total_samples,
                class_counts[1] / total_samples,
            ]  # Compute probabilities for each class
            return proba
        feature, threshold = tree["feature"], tree["threshold"]
        if x[feature] == threshold:
            return self._predict_proba_one(x, tree["left"])
        else:
            return self._predict_proba_one(x, tree["right"])

    def get_tree(self) -> Dict[str, Any]:
        """
        Get the decision tree with feature names instead of feature indices.

        :return: A dictionary representing the decision tree.
        """

        def replace_feature_names(node: Dict[str, Any]) -> Dict[str, Any]:
            if "label" in node:
                return node
            feature_name = self.features[node["feature"]]
            left_tree = replace_feature_names(node["left"])
            right_tree = replace_feature_names(node["right"])
            return {
                "feature": feature_name,
                "threshold": node["threshold"],
                "left": left_tree,
                "right": right_tree,
            }

        return replace_feature_names(self.tree_)

    def extract_rules(
            self,
            node: Optional[Dict[str, Any]] = None,
            current_rule: Optional[List[Tuple[str, str, Any]]] = None,
    ) -> Any:
        """
        Extract decision rules from the tree.

        :param node: Current node in the decision tree. If None, start from the root.
        :param current_rule: The list of conditions accumulated so far.
        :return: A list of decision rules (each rule is a list of conditions).
        """
        if node is None:
            node = self.tree_
        if current_rule is None:
            current_rule = []

        if "label" in node:
            if node["label"] == 1:
                return [current_rule]
            else:
                return []

        rules = []
        feature = self.features[node["feature"]]
        threshold = node["threshold"]

        left_rule = current_rule + [(feature, "==", threshold)]
        right_rule = current_rule + [(feature, "!=", threshold)]

        rules += self.extract_rules(node["left"], left_rule)
        rules += self.extract_rules(node["right"], right_rule)

        return rules


def predict(test_data_path: str, model: Any) -> Dict[Any, List[Tuple[datetime, float]]]:
    """
    Predict anomaly scores for test data files using the provided model.

    :param test_data_path: Path to the directory containing test data files.
    :param model: A trained model with a predict method.
    :return: A dictionary mapping each serial number (sn) to a list of tuples,
             each containing a timestamp and the predicted score.
    """
    result_all: Dict[Any, List[Tuple[datetime, float]]] = {}
    for test_file in sorted(
            os.listdir(test_data_path),
            key=lambda filename: int(filename.split("_")[-1].split(".")[0]),
    ):
        test_df = pd.read_feather(f"{test_data_path}/{test_file}")
        test_df["sn_name"] = test_df.index.get_level_values(0)
        X_test = test_df.values
        predict_result = model.predict(X_test)

        index_list = list(test_df.index)
        for i in tqdm(range(len(index_list)), test_file):
            p_s = predict_result[i]
            if p_s < 0.1:
                continue
            sn = index_list[i][0]
            # sn_t = datetime.fromtimestamp(index_list[i][1])    # 正常时间：年月日
            sn_t = index_list[i][1]       # 时间戳形式

            if sn not in result_all:
                result_all[sn] = [(sn_t, p_s)]
            else:
                result_all[sn].append((sn_t, p_s))

    return result_all


def submit(result, type: str):
    """
    :param result: 预测结果字典，key 为 sn_name, value 为时间戳列表
    :param type: "time_patch"或"time_point"
    :return:
    """
    submission = []    # 将预测结果转换为提交格式
    for sn in result:  # 遍历每个 SN 的预测结果
        for timestamp in result[sn]:  # 遍历每个时间戳
            submission.append([sn, timestamp])  # 添加 SN 名称、预测时间戳

    submission = pd.DataFrame(
        submission, columns=["sn_name", "prediction_timestamp"]
    )             # 将提交数据转换为 DataFrame 并保存为 CSV 文件
    submission.to_csv(os.path.join(WORK_DIR, f"submission_{type}.csv"), index=False, encoding="utf-8")


if __name__ == "__main__":
    print("Dataset directory listing:", os.listdir(DATASET_DIR))
    print("Type A file count:", len(os.listdir(os.path.join(DATASET_DIR, "type_A"))))
    print("Type B file count:", len(os.listdir(os.path.join(DATASET_DIR, "type_B"))))
    config = Config(
        raw_data_path=os.path.join(DATASET_DIR, "type_A"),
        bsfe_feature_path=os.path.join(WORK_DIR, "bfse_processed_data"),
        time_patch_processed_path=os.path.join(WORK_DIR, "time_patch_processed"),
        time_point_train_data_path=os.path.join(WORK_DIR, "time_point_train"),
        time_point_test_data_path=os.path.join(WORK_DIR, "time_point_test"),
        time_patch_train_data_path=os.path.join(WORK_DIR, "time_patch_train"),
        time_patch_test_data_path=os.path.join(WORK_DIR, "time_patch_test"),
        ticket_path=os.path.join(DATASET_DIR, "ticket.csv")
    )

    # 1.生成BSFE处理数据
    bsfe = BSFE_DQBeatMatrix(config)
    bsfe.bsfe_process_all_sn()  # 获取bfse_processed_data

    # 2.生成time_patch处理数据
    time_patch = Time_Patch_Process(config)
    time_patch.process_all_sn()

    # 3.对处理后两个数据集分别划分训练测试集
    data_types = ["time_patch", "time_point"]
    for data_type in data_types:
        print(f"生成 {data_type} 数据的训练集和测试集...")

        # 创建对应类型的数据生成器
        positive_gen = PositiveDataGenerator(config, data_type=data_type)
        negative_gen = NegativeDataGenerator(config, data_type=data_type)
        test_gen = TestDataGenerator(config, data_type=data_type)

        # 生成数据
        positive_gen.generate_and_save_data()
        negative_gen.generate_and_save_data()
        test_gen.generate_and_save_data()

        print(f"{data_type} 数据生成完成！")
    print("所有数据生成完成！")

    # 4.对time_patch数据用lightgbm预测
    print("lightgbm开始")
    model = MFPmodel(config)        # 初始化模型类 MFPmodel，加载训练数据并训练模型
    model.load_train_data()  # 加载训练数据
    model.train()  # 训练模型
    result = model.predict()  # 使用训练好的模型进行预测

    submit(result, "time_patch")
    print("已生成time_patch提交文件")

    # 5.对time_point数据用CustomDecisionTreeClassifier进行预测
    pos = pd.read_feather(os.path.join(config.time_point_train_data_path, "positive_train.feather"))
    neg = pd.read_feather(os.path.join(config.time_point_train_data_path, "negative_train.feather"))
    train_all = pd.concat([pos, neg])
    train_all["sn_name"] = train_all.index.get_level_values(0)

    # 决策时类的三个参数
    all_sns_set = set(train_all["sn_name"])
    pos_sn_set = set(train_all[train_all["label"] == 1]["sn_name"])
    features = get_use_high_order_bit_level_features_names()

    X = np.array(train_all.drop(["label"], axis=1))
    y = train_all["label"]
    model2 = CustomDecisionTreeClassifier(max_depth=4, all_sns_set=all_sns_set,
                                          pos_sn_set=pos_sn_set, features=features)
    model2.fit(X, y)
    result2 = predict(config.time_point_test_data_path, model2)
    submit(result2, "time_point")
    print("已生成time_point提交文件")
