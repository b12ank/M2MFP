# ^_^
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import warnings
import gc

warnings.filterwarnings('ignore')

# ================= 配置路径 =================
SOURCE_DIR = r"E:\pycharmproject\phase2_data\type_A\data_all"
OUTPUT_DIR = r"E:/pycharmproject/raw_code/新工作/UPH全部样本/UPH全部特征构造"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ================= 辅助函数：CE Storm 计算 =================
def calculate_ce_storm_count(log_times_series):
    """
    计算窗口内的 CE Storm 次数。
    逻辑参考 M2-MFP: 间隔 <= 60s 且连续 > 10 次
    """
    if len(log_times_series) < 10:
        return 0

    # 确保排序
    times = log_times_series.sort_values().values
    ce_storm_count = 0
    consecutive_count = 0
    interval_threshold = 60  # 秒
    count_threshold = 10  # 次

    for i in range(1, len(times)):
        if times[i] - times[i - 1] <= interval_threshold:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # 触发一次 Storm 计数 (简单的触发逻辑，避免重复计数可以加标志位)
        # 这里采用宽松逻辑：每满足一次阈值算一次，或者一段连续算一次
        # 为简化，这里记录"达到阈值的次数"
        if consecutive_count == count_threshold:
            ce_storm_count += 1
            # consecutive_count = 0 # 如果想重置计数

    return ce_storm_count


# ================= 高级 Parity 解码 (增强版) =================
def decode_parity_advanced_v2(parity_int):
    """
    解码 Parity，提取位级特征，包括 Multi-pin beat 计数。
    """
    if pd.isna(parity_int) or parity_int == 0:
        return {
            'pins': [], 'beats': [],
            'dq_adj_count': 0, 'bt_min_int': -1, 'bt_max_int': -1, 'dq_min_int': -1,
            'multi_pin_beat_cnt': 0  # [新增] 单条CE中，含多Pin错误的Beat数量
        }

    bin_str = bin(int(parity_int))[2:].zfill(32)
    active_pins = set()
    active_beats = set()

    # 临时字典：记录每个 Beat 上坏了几个 Pin
    # key: beat_idx (0-7), value: pin_count
    beat_pin_counts = {}

    for i, bit in enumerate(bin_str):
        if bit == '1':
            beat_idx = i // 4
            pin_idx = i % 4
            active_beats.add(beat_idx)
            active_pins.add(pin_idx)

            beat_pin_counts[beat_idx] = beat_pin_counts.get(beat_idx, 0) + 1
            # {0:1,}   第0个beat上坏了一个pin  后续判别CE BtMulti *（多pin错误的beat数）用

    sorted_pins = sorted(list(active_pins))
    sorted_beats = sorted(list(active_beats))

    # 计算 DQ Adjacent
    dq_adj_count = 0       # 相邻错误pin数
    for p in sorted_pins:
        if (p + 1) in active_pins:
            dq_adj_count += 1

    # 计算 Intervals
    bt_min_int = min([sorted_beats[i + 1] - sorted_beats[i] for i in range(len(sorted_beats) - 1)]) if len(sorted_beats) > 1 else 0
    bt_max_int = max([sorted_beats[i + 1] - sorted_beats[i] for i in range(len(sorted_beats) - 1)]) if len(sorted_beats) > 1 else 0
    dq_min_int = min([sorted_pins[i + 1] - sorted_pins[i] for i in range(len(sorted_pins) - 1)]) if len(sorted_pins) > 1 else 0

    # [新增] 计算 Multi-pin Beats 数量 (Table III: CE_BtMulti)
    # 定义：一个 Beat 如果包含 >1 个错误 Pin，则该 Beat 算作 Multi-pin Beat
    multi_pin_beat_cnt = sum(1 for count in beat_pin_counts.values() if count > 1)

    return {
        'pins': list(active_pins),     # 出错pins索引
        'beats': list(active_beats),   # 出错beats索引
        'dq_adj_count': dq_adj_count,  # 相邻错误pin数
        'bt_min_int': bt_min_int,      # beat错最小间隔
        'bt_max_int': bt_max_int,      # beat错最大间隔
        'dq_min_int': dq_min_int,      # dq错最小间隔
        'multi_pin_beat_cnt': multi_pin_beat_cnt     # 多pin错误的beat数
    }


# ================= 核心处理逻辑 =================
def process_single_sn_full_features(file_path):
    try:
        if file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            df = pd.read_csv(file_path)

        filename = os.path.basename(file_path)
        sn_name = os.path.splitext(filename)[0]

        if 'LogTime' not in df.columns: return None
        df = df.sort_values('LogTime').reset_index(drop=True)

        # 1. 预处理：解码 Parity
        adv_infos = []
        for parity in df['RetryRdErrLogParity']:
            info = decode_parity_advanced_v2(parity)
            adv_infos.append({
                'dq_count': len(info['pins']),
                'beat_count': len(info['beats']),
                'dq_adj': info['dq_adj_count'],
                'bt_m': info['bt_min_int'],
                'bt_M': info['bt_max_int'],
                'dq_m': info['dq_min_int'],
                'multi_pin_beat_cnt': info['multi_pin_beat_cnt'],  # [新增]
                'pins_list': info['pins']
            })
        adv_df = pd.DataFrame(adv_infos)
        df = pd.concat([df, adv_df], axis=1)

        # 2. 窗口设置 (6小时)
        start_time = df['LogTime'].min()
        end_time = df['LogTime'].max()
        window_size = 24 * 3600
        step_size = 24*3600

        # === 历史状态变量 ===
        history_windows_with_ce = 0
        last_window_ce_count = -1
        history_same_count = 0

        features_list = []

        current_cursor = start_time
        while current_cursor <= end_time:
            window_end = current_cursor + window_size
            win_df = df[(df['LogTime'] >= current_cursor) & (df['LogTime'] < window_end)]

            if len(win_df) > 0:
                ce_count = len(win_df)

                # --- [Group 1] History (H) ---
                # HS_sameCnt: 与上一窗口报错数相同的累积次数
                is_same = 1 if ce_count == last_window_ce_count else 0
                h_feats = {
                    'HS_cnt': history_windows_with_ce,
                    'HS_sameCnt': history_same_count,  # [确认] 已包含
                    'HS_sameRate': history_same_count / history_windows_with_ce if history_windows_with_ce > 0 else 0
                }
                history_windows_with_ce += 1
                if is_same: history_same_count += 1
                last_window_ce_count = ce_count

                # --- [Group 2] Spatial ---
                row_counts = win_df['RowId'].value_counts()
                col_counts = win_df['ColumnId'].value_counts()
                spatial_feats = {
                    'r': row_counts.max(),
                    'c': col_counts.max(),
                    'y': len(row_counts),
                    'x': len(col_counts),
                    'rrange': win_df['RowId'].max() - win_df['RowId'].min(),
                    'crange': win_df['ColumnId'].max() - win_df['ColumnId'].min(),
                    'Mat': (win_df['RowId'].max() - win_df['RowId'].min() + 1) * (win_df['ColumnId'].max() - win_df['ColumnId'].min() + 1) if len(win_df) > 1 else 1
                }

                # --- [Group 3] Unique (U) ---
                win_df['pos_key'] = win_df['BankId'].astype(str) + '_' + win_df['RowId'].astype(str) + '_' + win_df['ColumnId'].astype(str)
                u_counts = win_df['pos_key'].value_counts()     # 返回一个series 索引是bankid_rowid_colid 值是数量
                u_feats = {
                    'Unique_cnt': len(u_counts),
                    'Unique_Max': u_counts.max(),
                    'Unique_Min': u_counts.min(),
                    'Unique_Avg': u_counts.mean(),
                    'Unique_Std': u_counts.std() if len(u_counts) > 1 else 0
                }

                # --- [Group 4] Pinx (DM Level) ---
                all_pins = []
                for p_list in win_df['pins_list']: all_pins.extend(p_list)
                pin_counts_map = {0: 0, 1: 0, 2: 0, 3: 0}
                for p in all_pins: pin_counts_map[p] = pin_counts_map.get(p, 0) + 1
                pin_vals = list(pin_counts_map.values())

                dm_pin_feats = {
                    'DM_PinCover': np.count_nonzero(pin_vals),
                    'DM_PinCnt': sum(pin_vals),
                    'DM_Pins_Max': max(pin_vals),
                    'DM_Pins_Min': min(pin_vals),
                    'DM_Pins_Avg': np.mean(pin_vals),
                    'DM_Pins_Std': np.std(pin_vals),
                    'DM_BtPattern': win_df['RetryRdErrLogParity'].nunique()
                }

                # --- [Group 5] Pinx (CE Level) - 补全缺失特征 ---
                ce_level_feats = {
                    'CE_PinAvg': win_df['dq_count'].mean(),
                    'CE_BtAvg': win_df['beat_count'].mean(),

                    'CE_BtCnt_Max': win_df['beat_count'].max(),
                    'CE_BtCnt_Min': win_df['beat_count'].min(),
                    'CE_BtCnt_Avg': win_df['beat_count'].mean(),
                    'CE_BtCnt_Std': win_df['beat_count'].std() if len(win_df) > 1 else 0,

                    # [新增] CE_BtMulti_*
                    # 统计窗口内所有 CE 的 multi_pin_beat_cnt 分布
                    'CE_BtMulti_Max': win_df['multi_pin_beat_cnt'].max(),
                    'CE_BtMulti_Min': win_df['multi_pin_beat_cnt'].min(),
                    'CE_BtMulti_Avg': win_df['multi_pin_beat_cnt'].mean(),
                    'CE_BtMulti_Std': win_df['multi_pin_beat_cnt'].std() if len(win_df) > 1 else 0,

                    'DQ_Adj_Sum': win_df['dq_adj'].sum(),
                    'Bt_m_Avg': win_df[win_df['bt_m'] > 0]['bt_m'].mean() if (win_df['bt_m'] > 0).any() else 0,
                    'Bt_M_Avg': win_df[win_df['bt_M'] > 0]['bt_M'].mean() if (win_df['bt_M'] > 0).any() else 0,
                    'DQ_m_Avg': win_df[win_df['dq_m'] > 0]['dq_m'].mean() if (win_df['dq_m'] > 0).any() else 0
                }

                # --- [Group 6] Risky / Events (补全 Storm, Overflow, CEs) ---
                risky_mask = (win_df['dq_count'] > 1) & (win_df['beat_count'] > 1)

                # [新增] CE_Overflow 模拟
                # 如果没有真实标签，我们假设短时间内巨量报错(>1000/窗口)为 Overflow
                # 或者如果有 'error_type' 列包含 'Overflow'，请取消注释下一行
                # is_overflow = win_df['error_type'].str.contains('Overflow').sum()
                is_overflow = 1 if ce_count > 1000 else 0

                # [新增] CE_Storm 计算
                storm_cnt = calculate_ce_storm_count(win_df['LogTime'])

                risk_feats = {
                    'CEs': ce_count,  # [新增] 对应 Table III "CEs"
                    'risky_CE': risky_mask.sum(),
                    'CE_Storm': storm_cnt,  # [新增] 真实的 Storm 计数
                    'CE_Overflow': is_overflow  # [新增] Overflow 指标
                }

                # --- 合并 ---
                feature_row = {
                    'sn_name': sn_name,
                    'ReportTime': win_df['LogTime'].max(),
                    **h_feats,
                    **spatial_feats,
                    **u_feats,
                    **dm_pin_feats,
                    **ce_level_feats,
                    **risk_feats
                }
                features_list.append(feature_row)

            current_cursor += step_size
        final_df = pd.DataFrame(features_list)
        final_df.to_feather(os.path.join(OUTPUT_DIR, f"{sn_name}.feather"))
        return pd.DataFrame(features_list)

    except Exception as e:
        print(f"Error: {e}")
        return None


def process_wrapper(file_path):
    return process_single_sn_full_features(file_path)


def subprocess_single_sn(args):
    sn_file_list = args[0]
    thread = args[1]

    for sn_file in tqdm(sn_file_list, f"thread: {thread}"):
        process_single_sn_full_features(sn_file)


def main():
    exist_sn_file_list = os.listdir(OUTPUT_DIR)
    for folder in os.listdir(SOURCE_DIR):
        # folder0 : E:\pycharmproject\phase2_data\type_A\data_all\folder0\type_A
        data_dir = os.path.join(os.path.join(SOURCE_DIR, folder), "type_A")

        worker = 4
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
        del data_dir, worker, threads, sn_file_list
        gc.collect()


if __name__ == '__main__':
    main()
    # df3 = process_single_sn_full_features(r"C:\Users\user\Desktop\test_UPh\sn_12068.feather")