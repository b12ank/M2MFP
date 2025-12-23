# ^_^
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm

UPH_train_path = r"E:\pycharmproject\raw_code\新工作\UPH全部样本\UPH_train"
data_list = []    # 负样本列表
data_list1 = []
for file in tqdm(os.listdir(UPH_train_path)):
    df = pd.read_feather(os.path.join(UPH_train_path, file))
    if file.startswith("negative"):   # 负样本放到data_list
        data_list.append(df)
    else:                             # 正样本放到data_list1
        data_list1.append(df)
merged_df = pd.concat(data_list, ignore_index=False)
merged_df1 = pd.concat(data_list1, ignore_index=False)
UPH_train_merge_path = r"E:\pycharmproject\raw_code\新工作\UPH全部样本\UPH_train_merge"
os.makedirs(UPH_train_merge_path, exist_ok=True)

merged_df.to_feather(os.path.join(UPH_train_merge_path, r"negative_train_neg40.feather"))
merged_df1.to_feather(os.path.join(UPH_train_merge_path, r"positive_train_neg40.feather"))
