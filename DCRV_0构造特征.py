import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from dataclasses import dataclass, field
from typing import Tuple
from tqdm import tqdm
from multiprocessing import Pool
import gc
import os

time_window_size_map = {1 * 24 * 3600: '1d', 7 * 24 * 3600: '7d', 14 * 24 * 3600: '14d', 28 * 24 * 3600: '28d'}
TIME_RELATED_LIST = [24 * 3600, 7 * 24 * 3600, 14 * 24 * 3600, 28 * 24 * 3600]  # 聚合窗口

stats_window_size_map = {1 * 24 * 3600: '1d', 3 * 24 * 3600: '3d', 7 * 24 * 3600: '7d'}
STATS_WINDOW_LIST = [24 * 3600, 3 * 24 * 3600, 7 * 24 * 3600]  # 聚合窗口

# parity有效校验编码
PARITY_VALID_MASK = 0x00000001
# 检验parity是否有UCE掩码
PARITY_UCE_MASK = 0x00000002


@dataclass
class Config:
    # 测试用的设置
    # raw_data_path: str = field(default=r"E:\pycharmproject\raw_code\新工作\测试样本", init=False)
    # output_data_path: str = field(default=r"E:\pycharmproject\raw_code\新工作\DCRV", init=False)

    raw_data_path: str = field(default=r"E:\pycharmproject\phase2_data\type_A\data_all", init=False)
    output_data_path: str = field(default=r"E:\pycharmproject\raw_code\新工作\DCRV\DCRV_processed_data", init=False)
    ticket_path = r"E:\pycharmproject\raw_code\failure_ticket.csv"


def unique_num_filtered(input_array: np.ndarray) -> int:
    """
    对输入的列表进行过滤,统计除了-1外的不同元素个数

    :param input_array: 输入的列表
    :return: 返回经过过滤后的列表元素个数
    """
    if input_array is None or len(input_array) == 0:
        return 0
    unique_array = np.unique(input_array)
    return len(unique_array) - int(-1 in unique_array)


def get_error_type_info(error_type_series):
    error_type_array = error_type_series.fillna("").values
    return pd.DataFrame({
        "error_type_is_CE": (error_type_array == "CE").astype(int),
        "error_type_is_READ_CE": (error_type_array == "CE.READ").astype(int),
        "error_type_is_SCRUB_CE": (error_type_array == "CE.SCRUB").astype(int)
    })


def get_retry_log_info(retry_rd_err_log_series):
    retry_rd_err_log_array = retry_rd_err_log_series.fillna(0).replace('', 0).astype(int).values
    retry_log_is_valid = (retry_rd_err_log_array & PARITY_VALID_MASK) > 0
    retry_log_is_uncorrectable_error = ((retry_rd_err_log_array & PARITY_UCE_MASK) > 0) & retry_log_is_valid
    return pd.DataFrame({
        "retry_log_is_valid": retry_log_is_valid.astype(int),
        "retry_log_is_uncorrectable_error": retry_log_is_uncorrectable_error.astype(int)
    })


def decode_parity_to_pins_beats(parity_int):
    """
    将 RetryRdErrLogParity (Int) 解码为具体的 Pin 和 Beat 信息。
    假设是 x4 DDR4，矩阵大小为 8 Beats x 4 DQs (Pins) = 32 bits。
    返回:
        active_pins (set): 出错的 DQ 索引集合 {0, 1, 2, 3}
        active_beats (set): 出错的 Beat 索引集合 {0..7}
        total_bit_errors (int): 总错误比特数
    """
    if pd.isna(parity_int) or parity_int == 0:
        return set(), set(), 0, 0, 0

    # 转为 32位 二进制字符串，不足补0
    bin_str = bin(int(parity_int))[2:].zfill(32)
    # 反转字符串以便从低位开始索引（视具体映射而定，这里采用通用假设）
    # 假设映射逻辑：每4位对应一个 Beat (0-3: Beat7, 4-7: Beat6..., 28-31:Beat0) 二进制字符串从右开始为第一个Beat

    active_pins = set()
    active_beats = set()
    total_bit_errors = 0

    # 遍历 32 位
    for i, bit in enumerate(bin_str):
        if bit == '1':
            total_bit_errors += 1
            # 简单的行列映射假设 (8行 x 4列)
            # 实际索引需要根据 log 手册，这里使用通用平铺映射作为特征代理
            beat_idx = 7 - (i // 4)
            pin_idx = i % 4
            active_beats.add(beat_idx)
            active_pins.add(pin_idx)

    # beat错误的最大最小间隔，文中有提到多突发错误更可能引起UCE
    sorted_beats = sorted(list(active_beats))
    bt_min_int = min([sorted_beats[i + 1] - sorted_beats[i] for i in range(len(sorted_beats) - 1)]) if len(sorted_beats) > 1 else 0
    bt_max_int = max([sorted_beats[i + 1] - sorted_beats[i] for i in range(len(sorted_beats) - 1)]) if len(sorted_beats) > 1 else 0

    # 输入：2 --> '00000000000000000000000000000010' --> 输出： {2},{0},1,0,0
    return active_pins, active_beats, total_bit_errors, bt_min_int, bt_max_int


def get_manufacturer_info(manufacturer_series):
    """
    输入：df["Manufacturer"]这一列series
    输出：(len(df["Manufacturer"]),4)形状的一个dataframe
    """
    manufacturer_array = manufacturer_series.fillna("").values
    return pd.DataFrame({
        "manufacturer_is_A": (manufacturer_array == "A").astype(int),
        "manufacturer_is_B": (manufacturer_array == "B").astype(int),
        "manufacturer_is_C": (manufacturer_array == "C").astype(int),
        "manufacturer_is_D": (manufacturer_array == "D").astype(int)
    })


def get_parity_info(parity_series):
    """

    Args:
        parity_series: df["RetryRdErrLogParity"]这一列series

    Returns: (len(df["Manufacturer"]), 5)形状的一个dataframe   列名：['错误DQ集合','错误Beat集合','错误bit数','错误DQ数','错误Beat数']

    """
    parity_infos = []
    for parity in parity_series:
        p, b, t, bt_min_int, bt_max_int = decode_parity_to_pins_beats(parity)
        parity_infos.append({
            'pins_set': p,
            'beats_set': b,
            'bit_count': t,
            'dq_count': len(p),
            'beat_count': len(b),
            'bt_min_int': bt_min_int,  # 最小错误节拍间隔 0-7
            'bt_max_int': bt_max_int  # 最大错误节拍间隔 0-7
        })
    parity_df = pd.DataFrame(parity_infos)
    return parity_df


def get_max_sum_mean(input_position: pd.Series, valid_position_count: int) -> Tuple[int, int, int]:
    """
    获取观测窗内 input_position 的最大值与平均值
    """
    max_value = input_position.max()
    if valid_position_count == 0:
        return 0, 0, 0
    sum_value = input_position.values.sum()
    mean_value = round(np.divide(sum_value, valid_position_count), 2)
    return max_value, sum_value, mean_value


def get_spatio_features(window_df):
    """
    获取聚合特征（无stats的特征） 1,7,14,28
    Args:
        window_df:  要含CellId,bit_count 特征列

    Returns:
        spatio_features: 字典

    """
    spatio_features = {
        # "device_cnt": unique_num_filtered(window_df["deviceId"].values),
        "bank_cnt": unique_num_filtered(window_df["BankId"].values),
        # "bank_group_cnt": unique_num_filtered(window_df["BankgroupId"].values),
        "ce_cnt_in_cell_max": 0,  # 论文特征18-29      单个错误cell中最大ce数
        "ce_cnt_in_cell_min": 0,  # 论文特征18-29      单个错误cell中最小ce数
        "ce_cnt_in_cell_sum": 0,  # 论文特征18-29      所有错误cell的ce数和
        "cell_in_col_max": 0,  # 论文特征76   所有错误列中cell错误最多的列有多少个不同错误cell
        "cell_in_row_max": 0,  # 77-80       所有错误行中cell错误最多的行有多少个不同错误cell
        "cell_in_row_min": 0,  # 81-84       所有错误行中cell错误最少的行有多少个不同错误cell
        "cell_in_row_sum": 0,  # 85-86       所有错误行中有多少不同错误cell
        "cell_cnt_sum": unique_num_filtered(window_df["CellId"].values),  # 87-90 错误cell数
        "col_cnt_sum": unique_num_filtered(window_df["ColumnId"].values),  # 95-97 含错误列数
        "col_single_bit": unique_num_filtered(window_df[window_df["bit_count"] == 1]["ColumnId"].values),  # 101  单bit错误列数
        "error_cnt_by_col_max": 0,  # 128-141   一列最多有几个ce错误
        "error_cnt_by_col_min": 0,
        "error_cnt_by_row_max": 0,
        "error_cnt_by_row_min": 0,
        "hard_error_cell_cnt_sum": 0,  # 142-145   硬错误cell数
        "hard_error_cnt_by_cell_max": 0,  # 146-154   和ce_cnt_in_cell特征不同的地方在于统计对象是筛选后的硬错误cell
        "hard_error_cnt_by_cell_min": 0,
        "hard_error_cnt_by_cell_sum": 0,
        "col_block_dq_break_cnt": 0,  # 92    多dq错误列计数
        "row_block_dq_break_cnt": 0,  # 214-217    多dq错误行计数
        "row_burst_max_diff": window_df[window_df['bt_max_int'] == window_df['bt_max_int'].max()].groupby(["RowId"]).ngroups,  # 218-221   有最大beat间隔的行数
        "row_cnt_sum": unique_num_filtered(window_df["RowId"].values),  # 226-229 含错误行数
        "row_max_multi_bits": 0,  # 235-238   发生最多次多bit ce的cell发生多少次ce
        "row_max_single_bit": 0,  # 239-242   发生最多次单bit ce的cell发生多少次ce
        "row_max_two_bits": 0,  # 249-252   发生最多次两bit ce的cell发生多少次ce
        "row_single_bit_sum": 0,  # 253  单bit ce总次数
        "row_two_bits_sum": 0,  # 258-260  2bit ce总次数
    }

    # "ce_cnt_in_cell"构造
    ce_cnt_in_cell_series = window_df.groupby(['CellId']).size()
    if not ce_cnt_in_cell_series.empty:
        spatio_features["ce_cnt_in_cell_max"] = ce_cnt_in_cell_series.max()
        spatio_features["ce_cnt_in_cell_min"] = ce_cnt_in_cell_series.min()
        spatio_features["ce_cnt_in_cell_sum"] = ce_cnt_in_cell_series.sum()
    # "cell_in_col"构造
    cell_in_col_series = window_df.groupby(['BankId', 'ColumnId'])['RowId'].apply(list)
    spatio_features["cell_in_col_max"] = max(
        unique_num_filtered(row_ids)
        for row_ids in cell_in_col_series
    ) if not cell_in_col_series.empty else 0
    # "cell_in_row"构造
    cell_in_row_series = window_df.groupby(['BankId', 'RowId'])["ColumnId"].apply(list)
    if not cell_in_row_series.empty:
        spatio_features["cell_in_row_max"] = max(
            unique_num_filtered(col_ids)
            for col_ids in cell_in_row_series
        )
        spatio_features["cell_in_row_min"] = min(
            unique_num_filtered(col_ids)
            for col_ids in cell_in_row_series
        )
        spatio_features["cell_in_row_sum"] = sum(
            unique_num_filtered(col_ids)
            for col_ids in cell_in_row_series
        )
    # "error_cnt_by_col"构造
    error_cnt_by_col_series = window_df.groupby(['BankId', 'ColumnId']).size()
    if not error_cnt_by_col_series.empty:
        spatio_features["error_cnt_by_row_max"] = error_cnt_by_col_series.max()
        spatio_features["error_cnt_by_row_min"] = error_cnt_by_col_series.min()
        # "error_cnt_by_row"构造
    error_cnt_by_row_series = window_df.groupby(['BankId', 'RowId']).size()
    if not error_cnt_by_row_series.empty:
        spatio_features["error_cnt_by_row_max"] = error_cnt_by_row_series.max()
        spatio_features["error_cnt_by_row_min"] = error_cnt_by_row_series.min()
    # 硬错误“hard_error_cell_cnt_sum”
    cell_counts = window_df.groupby(['deviceID', 'BankId', 'RowId', 'ColumnId']).size()  # series: 索引是['deviceID', 'BankId', 'RowId', 'ColumnId'],值是这个索引的行数量
    hard_fault_cells = cell_counts[cell_counts > 1]  # 筛选出报错次数 > 1的Cell数量
    spatio_features["hard_error_cell_cnt_sum"] = len(hard_fault_cells) if not cell_counts.empty else 0
    # "hard_error_cnt_by_cell"
    if not hard_fault_cells.empty:
        spatio_features["hard_error_cnt_by_cell_max"] = hard_fault_cells.max()
        spatio_features["hard_error_cnt_by_cell_min"] = hard_fault_cells.min()
        spatio_features["hard_error_cnt_by_cell_sum"] = hard_fault_cells.sum()
    # "row_block_dq_break_cnt"和 "col_block_dq_break_cnt"
    dq_break_df = window_df[window_df["dq_count"] > 1]
    if not dq_break_df.empty:
        spatio_features["col_block_dq_break_cnt"] = dq_break_df.groupby(['ColumnId']).ngroups
        spatio_features["row_block_dq_break_cnt"] = dq_break_df.groupby(['RowId']).ngroups
    # "row_max_multi_bits"
    multi_bits_df = window_df[window_df["bit_count"] > 1]
    if not multi_bits_df.empty:
        # 索引是 (Bank,Row, Col)，值是Count
        cell_multi_bits_counts = multi_bits_df.groupby(['BankId', 'RowId', 'ColumnId']).size()
        # 找到每一行中，报错最多的那个Column(cell)的ce次数
        # 按 (Bank, Row) 分组，取 max
        row_max_multi_bits_counts = cell_multi_bits_counts.groupby(['BankId', 'RowId']).max()
        if not row_max_multi_bits_counts.empty:
            spatio_features["row_max_multi_bits"] = row_max_multi_bits_counts.max()
    # "row_max_single_bit"， "row_single_bit_sum"
    single_bit_df = window_df[window_df["bit_count"] == 1]
    if not single_bit_df.empty:
        spatio_features["row_single_bit_sum"] = len(single_bit_df)
        # 索引是 (Bank,Row, Col)，值是Count
        cell_single_bit_counts = single_bit_df.groupby(['BankId', 'RowId', 'ColumnId']).size()
        # 找到每一行中，报错最多的那个Column(cell)的ce次数
        # 按 (Bank, Row) 分组，取 max
        row_max_single_bit_counts = cell_single_bit_counts.groupby(['BankId', 'RowId']).max()
        if not row_max_single_bit_counts.empty:
            spatio_features["row_max_single_bit"] = row_max_single_bit_counts.max()
    # "row_max_two_bits"
    two_bits_df = window_df[window_df["bit_count"] == 2]
    if not two_bits_df.empty:
        spatio_features["row_two_bits_sum"] = len(two_bits_df)
        # 索引是 (Bank,Row, Col)，值是Count
        cell_two_bits_counts = two_bits_df.groupby(['BankId', 'RowId', 'ColumnId']).size()
        # 找到每一行中，报错最多的那个Column(cell)的ce次数
        # 按 (Bank, Row) 分组，取 max
        row_max_two_bits_counts = cell_two_bits_counts.groupby(['BankId', 'RowId']).max()
        if not row_max_two_bits_counts.empty:
            spatio_features["row_max_two_bits"] = row_max_two_bits_counts.max()

    return spatio_features


def get_stats_one_hour_features(window_df):
    stats_features_dict = {
        "bank_cnt": unique_num_filtered(window_df["BankId"].values),    # 不同bank计数
        "bank_group_cnt": unique_num_filtered(window_df["BankgroupId"].values),     # 不同bankgroup计数
        "ce_cnt": len(window_df),          # ce计数
        "cell_cnt": unique_num_filtered(window_df["CellId"].values),    # 发生过ce的cell计数
        "multi_bit_cell_cnt": 0,       # 发生过多bit ce 的cell计数
        "repeat_cell_cnt": 0,      # 发生过多次ce的 cell计数 (hard_error)
        "repeat_row_cell_cnt": 0,      # 发生过硬错误的行数
        "repeat_row_cnt": 0,      # 发生过多次ce的行计数
        "single_bit_cell_cnt": 0,      # 发生过多次ce的行计数
        "single_event": 0,      # 单bit ce 日志数量
    }
    # "multi_bit_cell_cnt"
    multi_bits_df = window_df[window_df["bit_count"] > 1]
    if not multi_bits_df.empty:
        stats_features_dict["multi_bit_cell_cnt"] = unique_num_filtered(multi_bits_df["CellId"].values)
    # "repeat_cell_cnt"
    cell_cnt_series = window_df.groupby(["CellId"]).size()
    hard_error_cell = cell_cnt_series[cell_cnt_series > 1]
    stats_features_dict["repeat_cell_cnt"] = len(hard_error_cell)
    # "repeat_row_cell_cnt"
    repeat_df = window_df[window_df["CellId"].isin(hard_error_cell.index)]
    stats_features_dict["repeat_row_cell_cnt"] = unique_num_filtered(repeat_df["RowId"].values)
    # "repeat_row_cnt"
    row_cnt_series = window_df.groupby(["RowId"]).size()
    stats_features_dict["repeat_row_cnt"] = len(row_cnt_series[row_cnt_series > 1])
    # "single_bit_cell_cnt"
    single_bit_df = window_df[window_df["bit_count"] == 1]
    if not single_bit_df.empty:
        stats_features_dict["single_bit_cell_cnt"] = unique_num_filtered(single_bit_df["CellId"].values)
    # "single_event"
        stats_features_dict["single_event"] = len(single_bit_df)

    return stats_features_dict


def stats_func(x, f_name, selected_keys=None):
    """

    Args:
        x:    要生成统计特征的series列
        f_name:     要生成统计特征的特征名
        selected_keys:      用到的统计函数列表

    Returns:   该统计特征所选的统计函数字典

    """
    if len(x) == 0:
        vals = {
            f"{f_name}_mean": 0, f"{f_name}_std": 0, f"{f_name}_skew": 0,
            f"{f_name}_kurtosis": 0, f"{f_name}_entropy": 0, f"{f_name}_dist": 0,
            f"{f_name}_max": 0, f"{f_name}_sum": 0, f"{f_name}_diff": 0
        }
        if selected_keys:
            return {f"{f_name}_{k}": vals[f"{f_name}_{k}"] for k in selected_keys}
        return vals

    # 2. 预先计算 Diff 序列，并处理第一个位置的 NaN
    x_diff = x.diff().fillna(0)  # 关键：fillna(0) 解决 diff 产生的 NaN
    # 3. 安全计算各个统计量
    val_std = np.std(x, ddof=0)

    # Skew & Kurtosis (偏度/峰度)
    # 只有当标准差 > 0 且 长度 >= 3 时计算才有意义，否则填 0
    if val_std > 1e-9 and len(x) >= 3:
        val_skew = skew(x)
        val_kurt = kurtosis(x)
    else:
        val_skew = 0
        val_kurt = 0

    # Entropy (熵)
    # 只有当 sum(x) > 0 时才能计算分布熵，否则填 0
    val_sum = np.sum(x)
    if val_sum > 0:
        # scipy entropy 会自动归一化，但如果含负数需注意，这里假设是计数特征(非负)
        val_entropy = entropy(x)
    else:
        val_entropy = 0

    stats = {
        f"{f_name}_mean": np.mean(x),
        f"{f_name}_std": val_std,    # 样本标准差
        f"{f_name}_skew": val_skew,  # 改进：防止 NaN  偏度
        f"{f_name}_kurtosis": val_kurt,  # 改进：防止 NaN  峰度
        f"{f_name}_entropy": val_entropy,  # 改进：防止 sum=0 报错     熵   0没有波动，数值越大波动越大
        f"{f_name}_dist": np.sum(np.abs(x_diff)),  # 改进：使用处理过 NaN 的 diff
        f"{f_name}_max": np.max(x),  # 已有 len>0 检查，安全
        f"{f_name}_sum": val_sum,
        f"{f_name}_diff": np.mean(x_diff)  # 改进：使用处理过 NaN 的 diff，防止返回 NaN   取平均diff
    }
    if selected_keys is not None:
        selected_keys_with_prefix = [f"{f_name}_{key}" for key in selected_keys]
        return {k: stats[k] for k in selected_keys_with_prefix if k in stats}

    return stats


def get_stats_features(window_df):
    step = 60 * 60  # 先每小时取一次样本，用于构造统计特征   取样间隔1h
    window_df = window_df.copy()  # 添加这行
    window_df['time_index'] = window_df['LogTime'] // step
    grouped = window_df.groupby('time_index')['LogTime'].max()  # 每个窗口最大时间
    window_end_time_list = grouped.tolist()
    stats_base_features_dict_list = []
    for end_time in window_end_time_list:
        w_df = window_df[(window_df['LogTime'] <= end_time) & (window_df['LogTime'] > end_time - step)]  # stats特征最大7d窗口
        combined_dict = {"LogTime": w_df["LogTime"].max()}
        combined_dict.update(get_stats_one_hour_features(w_df))    # 这种采样方法w_df不会为空，至少有一条样本
        stats_base_features_dict_list.append(combined_dict)
    stats_base_feature_df = pd.DataFrame(stats_base_features_dict_list)

    stats_features_dict = {}
    # "bank_cnt_stats_dist"        1
    stats_features_dict.update(stats_func(stats_base_feature_df["bank_cnt"], "bank_cnt_stats", ["dist"]))
    # "bank_group_cnt_stats"      6-12
    stats_features_dict.update(stats_func(stats_base_feature_df["bank_group_cnt"], "bank_group_cnt_stats", ["diff", "dist", "skew"]))
    # "ce_cnt_stats"          30-56
    stats_features_dict.update(stats_func(stats_base_feature_df["ce_cnt"], "ce_cnt_stats"))
    # "cell_cnt_stats"       57-75
    stats_features_dict.update(stats_func(stats_base_feature_df["cell_cnt"], "cell_cnt_stats", ["diff", "dist", "entropy", "kurtosis", "mean", "skew", "std", "sum"]))
    # "multi_bit_cell_cnt_stats"      160-164
    stats_features_dict.update(stats_func(stats_base_feature_df["multi_bit_cell_cnt"], "multi_bit_cell_cnt_stats", ["kurtosis", "mean", "skew", "std"]))
    # "repeat_cell_cnt_stats"      189-208
    stats_features_dict.update(stats_func(stats_base_feature_df["repeat_cell_cnt"], "repeat_cell_cnt_stats", ["diff", "dist", "entropy", "kurtosis", "mean", "skew", "std", "sum"]))
    # "repeat_row_cell_cnt_stats"      209-212
    stats_features_dict.update(stats_func(stats_base_feature_df["repeat_row_cell_cnt"], "repeat_row_cell_cnt_stats", ["diff", "kurtosis", "skew"]))
    # "repeat_row_cnt"      213
    stats_features_dict.update(stats_func(stats_base_feature_df["repeat_row_cnt"], "repeat_row_cnt_stats", ["diff"]))
    # "single_bit_cell_cnt_stats"      261-266
    stats_features_dict.update(stats_func(stats_base_feature_df["single_bit_cell_cnt"], "single_bit_cell_cnt_stats", ["diff", "dist", "kurtosis", "skew"]))
    # "single_event_stats"      267-281
    stats_features_dict.update(stats_func(stats_base_feature_df["single_event"], "single_event_stats", ["diff", "dist", "kurtosis", "mean", "skew"]))

    return stats_features_dict


def process_single_file(file_path):
    """

    Args:
        file_path: 文件名路径(..\sn_1.feather)

    Returns:

    """
    try:
        filename = os.path.basename(file_path)
        sn_name = os.path.splitext(filename)[0]
        # raw_df = pd.read_feather(os.path.join(Config.raw_data_path, file_path))
        raw_df = pd.read_feather(file_path)
        manufacturer = raw_df["Manufacturer"].values[0]  # 用于后面生成静态特征 生产商  假设同一sn文件厂商都一样

        base_df = raw_df[["LogTime", "deviceID", "BankId", "BankgroupId", "RowId", "ColumnId"]].copy()
        base_df['deviceID'] = base_df['deviceID'].fillna(-1).astype(int)  # 处理缺失值，用-1填充
        parity_info_df = get_parity_info(raw_df["RetryRdErrLogParity"])
        retry_log_df = get_retry_log_info(raw_df["RetryRdErrLog"])
        error_type_df = get_error_type_info(raw_df["error_type_full_name"])

        new_df = pd.concat([base_df, parity_info_df, retry_log_df, error_type_df], axis=1)
        sample_step = 60 * 60 * 24  # 每小时取一次样
        new_df['time_index'] = new_df['LogTime'] // sample_step
        new_df['CellId'] = new_df['RowId'].astype(str) + '_' + new_df['ColumnId'].astype(str)  # 加入CellId列名
        grouped = new_df.groupby('time_index')['LogTime'].max()  # 每个窗口最大时间
        window_end_time_list = grouped.tolist()

        combined_dict_list = []
        stats_dict_list = []
        for end_time in window_end_time_list:  # 对一个文件的一次采样
            window_df = new_df[(new_df['LogTime'] <= end_time) & (new_df['LogTime'] > end_time - 28 * 24 * 3600 - 24 * 3600)]
            combined_dict = {
                "ReportTime": window_df["LogTime"].max(),
                "LogTime": window_df["LogTime"].max(),
                "Manufacturer": manufacturer
            }  # 初始化该样本字典
            for time_window_size in TIME_RELATED_LIST[::-1]:  # 此次采样回顾不同窗口（1d,7d,28d）构造时间特征 聚合窗口特征
                window_df = window_df[window_df['LogTime'] >= combined_dict["LogTime"] - time_window_size]
                spatio_features = get_spatio_features(window_df)
                combined_dict.update({f"{key}_{time_window_size_map[time_window_size]}": value for d in
                                      [spatio_features] for
                                      key, value in d.items()})
            combined_dict_list.append(combined_dict)
            stats_dict = {}
            for time_window_size in STATS_WINDOW_LIST[::-1]:
                stats_window_df = new_df[(new_df['LogTime'] <= end_time) & (new_df['LogTime'] > end_time - time_window_size)]
                stats_features = get_stats_features(stats_window_df)
                stats_dict.update({f"{key}_{stats_window_size_map[time_window_size]}": value for d in
                                   [stats_features] for
                                   key, value in d.items()})
            stats_dict_list.append(stats_dict)
        combined_df_tmp = pd.DataFrame(combined_dict_list)
        stats_df = pd.DataFrame(stats_dict_list)
        # stats_df.fillna(0, inplace=True)
        manufacturer_df = get_manufacturer_info(combined_df_tmp["Manufacturer"])
        combined_df = pd.concat([combined_df_tmp.drop(["Manufacturer"], axis=1), stats_df, manufacturer_df], axis=1)
        combined_df.to_feather(os.path.join(Config.output_data_path, f"{sn_name}.feather"))
        # combined_df.to_csv(os.path.join(r"E:\pycharmproject\raw_code\新工作\DCRV", f"{sn_name}.csv"), index=False)
    except Exception as e:
        print(f"Error: {e}")


def subprocess_single_sn(args):
    sn_file_list = args[0]
    thread = args[1]

    for sn_file in tqdm(sn_file_list, f"thread: {thread}"):
        process_single_file(sn_file)


if __name__ == "__main__":
    # 测试单个文件用
    # process_single_file(r"E:\pycharmproject\raw_code\新工作\测试样本\sn_14663.feather")

    os.makedirs(Config.output_data_path, exist_ok=True)
    exist_sn_file_list = os.listdir(Config.output_data_path)
    for folder in os.listdir(Config.raw_data_path):
        # folder0 : E:\pycharmproject\phase2_data\type_A\data_all\folder0\type_A
        data_dir = os.path.join(os.path.join(Config.raw_data_path, folder), "type_A")

        worker = 8
        threads = list(range(worker))

        sn_file_list = os.listdir(data_dir)

        file_list_sp = [[] for _ in range(worker)]
        for i, file_name in enumerate([i for i in sn_file_list if i not in exist_sn_file_list and i.endswith("feather")]):
            index = i % worker
            file_path = os.path.join(os.path.join(data_dir, file_name))
            file_list_sp[index].append(file_path)

        pool = Pool(worker)
        pool.imap_unordered(subprocess_single_sn, zip(file_list_sp, threads))
        pool.close()
        pool.join()
        print(f"{folder} 特征构造完成！")
        # del data_dir, worker, threads, sn_file_list
        # gc.collect()


