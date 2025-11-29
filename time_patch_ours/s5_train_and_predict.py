import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime

import feather
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm


time_window_size_map = {15 * 60: '15m', 60 * 60: '1h', 6 * 3600: '6h'}


@dataclass
class Config:
    data_path: str = field(default=r"E:\pycharmproject\M2-MFP\data1\type_A", init=False)
    processed_time_point_data_path: str = field(default=r"E:\pycharmproject\raw_code\work\bfse_processed_data", init=False)
    combined_sn_feature_data_path: str = field(default=r"E:\pycharmproject\raw_code\work\time_patch_processed", init=False)
    ticket_path = r"E:\pycharmproject\M2-MFP\data1\ticket.csv"
    train_data_path = r"E:\pycharmproject\raw_code\work\time_patch_train"
    test_data_path = r"E:\pycharmproject\raw_code\work\time_patch_test"
    result_path = r"E:\pycharmproject\raw_code\work"


train_date_range = ("2024-01-01", "2024-05-01")
test_date_range = ("2024-05-01", "2024-06-01")


def train_model(train_data_path):
    train_pos = feather.read_dataframe(f'{train_data_path}/positive_train.feather')
    train_neg = feather.read_dataframe(f'{train_data_path}/negative_train.feather')
    train_neg['label'] = 0
    train_all = pd.concat([train_pos, train_neg])
    train_all.fillna(0, inplace=True)
    train_all = train_all.sample(frac=1, random_state=2024)
    print("Shape of training set:", train_all.shape)

    set_v5 = set([f"sn_{i}" for i in range(1, 65670)])
    train_all = train_all[train_all.index.get_level_values(0).isin(set_v5)]

    use_features = train_all.columns
    use_features = [i for i in use_features if i != "LogTime" and i != "ReportTime"]
    use_features = [i for i in use_features if "ce_log_num" not in i]

    train_all = train_all[use_features]
    train_all = train_all.sort_index(axis=1)

    train_all["sn_name"] = train_all.index.get_level_values(0)
    train_all.drop(columns=["sn_name"], inplace=True)

    LGB_MODEL_PARAMS = {"learning_rate": 0.02, "n_estimators": 500, "max_depth": 8,
                        'num_leaves': 20, 'min_child_samples': 20, 'verbose': 1,
                        'importance_type': 'gain'}
    model = lgb.LGBMClassifier(**LGB_MODEL_PARAMS)
    model.fit(train_all.drop(columns=['label']), train_all['label'])
    model_save_path = r"E:\pycharmproject\raw_code\work"
    with open(os.path.join(model_save_path, "time_patch_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"模型已经保存到{model_save_path}路径下")
    # feature_importance = model.booster_.feature_importance(importance_type='gain')
    # importance_df = pd.DataFrame({
    #     'Feature': model.feature_name_,
    #     'Importance (Gain)': feature_importance
    # })
    # importance_df = importance_df.sort_values(by='Importance (Gain)', ascending=False)
    # importance_df.to_csv('feature_importance_gain.csv', index=False)

    return model


def predict(test_data_path, model):
    result_all = {}
    for test_file in os.listdir(test_data_path):
        test_df = feather.read_dataframe(f'{test_data_path}/{test_file}')
        test_df["sn_name"] = test_df.index.get_level_values(0)
        test_df["log_time"] = test_df.index.get_level_values(1)

        test_df = test_df[model.feature_name_]

        predict_result = model.predict_proba(test_df)

        index_list = list(test_df.index)
        for i in tqdm(range(len(index_list)), test_file):
            p_s = predict_result[i][1]
            if p_s < 0.1:
                continue
            sn = index_list[i][0]
            sn_t = datetime.fromtimestamp(index_list[i][1])
            if sn not in result_all:
                result_all[sn] = [(sn_t, p_s)]
            else:
                result_all[sn].append((sn_t, p_s))
                # result_all[sn].append((sn_t, max(p_s, result_all[sn][-1][1])))   # 同一块dimm只保留最大概率
    return result_all


def train_and_predict():
    train_data_path = Config.train_data_path
    test_data_path = Config.test_data_path

    model = train_model(train_data_path)
    result_all = predict(test_data_path, model)

    # with open(os.path.join(Config.result_path, "time_patch_ours.pkl"), 'wb') as f:
    #     pickle.dump(result_all, f)
    submission = []
    for sn in result_all:
        for i in result_all[sn]:
            submission.append([sn, i[0], i[1]])
    submission = pd.DataFrame(submission, columns=["sn_name", "timestamp", "probability"])
    submission.to_csv(os.path.join(Config.result_path, "time_patch_submission.csv"), index=False)


os.makedirs(Config.result_path, exist_ok=True)
ticket = pd.read_csv(Config.ticket_path)
ticket = ticket[ticket['sn_type'] == "A"]

if __name__ == "__main__":
    train_and_predict()
