# ^_^
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from multiprocessing import Pool
from typing import Tuple, List, Dict, NoReturn, Union, Any, Optional
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

train_data_path = r"E:\pycharmproject\raw_code\work1\time_point_train"
test_data_path = r"E:\pycharmproject\raw_code\work1\time_point_test"
result_path = r"E:\pycharmproject\raw_code"
anomaly_parities = set()
first_layer_feature = None
second_layer_feature = None
max_depth = 4


WORK_DIR = r"E:\pycharmproject\raw_code\work1"
# Define functions for parallel concatenation of DataFrame chunks.
CHUNK_SIZE = 200


def get_use_parity_feature_names():
    parity_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["F_max_pooling", "max_pooling_F"]:     # 少了一个"sum_pooling_F"
            for F_function in ["bit_count", "bit_min_interval", "bit_max_interval",
                               "bit_max_consecutive_length", "bit_consecutive_length"]:
                parity_feature_names.append(f"parity_{row_column}wise_{G_function}_{F_function}")
    return parity_feature_names


features = get_use_parity_feature_names() + ['in_pos_parity_set',
                                             'retry_log_is_uncorrectable_error']


def all_digits_even(n):
    while n > 0:
        if (n % 16) % 2 != 0:
            return False
        n //= 16
    return True


class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, *args, **kwargs):
        first_layer_feature = kwargs.pop('first_layer_feature', None)
        second_layer_feature = kwargs.pop('second_layer_feature', None)
        super().__init__(*args, **kwargs)
        self.first_layer_feature = first_layer_feature
        self.second_layer_feature = second_layer_feature

    def _gini(self, y):
        """计算 Gini 指数"""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _weighted_gini(self, y_left, y_right):
        """计算加权 Gini 指数"""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        return (n_left / n_total) * self._gini(y_left) + (n_right / n_total) * self._gini(y_right)

    def _split_criterion(self, X, y, depth):
        """找到最佳分割特征和阈值"""
        if depth == 0 and self.first_layer_feature is not None:
            feature = features.index(self.first_layer_feature)
            thresholds = np.unique(X[:, feature])
        elif depth == 1 and self.second_layer_feature is not None:
            feature = features.index(self.second_layer_feature)
            thresholds = np.unique(X[:, feature])
        else:
            best_gini = float('inf')
            best_split = (None, None)
            n_samples, n_features = X.shape
            n_features -= 1

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_mask = X[:, feature] == threshold
                    left_sns_set = set(X[left_mask][:, -1])
                    right_sns_set = all_sns_set - left_sns_set
                    left_mask = np.array([x in left_sns_set for x in X[:, -1]])
                    right_mask = ~left_mask

                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue

                    y_left = [1 if i in pos_sn_set else 0 for i in left_sns_set]
                    y_right = [1 if i in pos_sn_set else 0 for i in right_sns_set]

                    gini = self._weighted_gini(y_left, y_right)

                    if gini < best_gini:
                        best_gini = gini
                        best_split = (feature, threshold)

            return best_split

        best_gini = float('inf')
        best_split = (None, None)

        for threshold in thresholds:
            left_mask = X[:, feature] == threshold
            left_sns_set = set(X[left_mask][:, -1])
            right_sns_set = all_sns_set - left_sns_set
            left_mask = np.array([x in left_sns_set for x in X[:, -1]])
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = [1 if i in pos_sn_set else 0 for i in left_sns_set]
            y_right = [1 if i in pos_sn_set else 0 for i in right_sns_set]

            gini = self._weighted_gini(y_left, y_right)

            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold)

        return best_split

    def fit(self, X, y):
        """训练模型"""
        self.tree_ = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件 1: 如果所有样本属于同一类别
        if len(np.unique(y)) == 1:
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 停止条件 2: 如果达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            if len(y) == 0:
                return {'label': 0, 'class_counts': {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 找到最佳分割
        feature, threshold = self._split_criterion(X, y, depth)

        # 停止条件 3: 如果无法找到有效分割
        if feature is None:
            if len(y) == 0:
                return {'label': 0, 'class_counts': {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 递归构建左右子树
        left_mask = X[:, feature] == threshold
        left_sns_set = set(X[left_mask][:, -1])
        right_sns_set = all_sns_set - left_sns_set
        left_mask = np.array([x in left_sns_set for x in X[:, -1]])
        right_mask = ~left_mask
        left_mask = X[:, feature] == threshold

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def predict(self, X):
        """预测"""
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree):
        """预测单个样本"""
        if 'label' in tree:  # 如果是叶子节点
            return tree['label']
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] == threshold:  # 如果样本的特征值等于阈值
            return self._predict_one(x, tree['left'])
        else:  # 如果样本的特征值大于阈值
            return self._predict_one(x, tree['right'])

    def predict_proba(self, X):
        """返回每个样本属于各个类别的概率"""
        return np.array([self._predict_proba_one(x, self.tree_) for x in X])

    def _predict_proba_one(self, x, tree):
        """返回单个样本的概率分布"""
        if 'class_counts' in tree:  # 如果是叶子节点
            class_counts = tree['class_counts']
            total_samples = sum(class_counts.values())  # 计算总样本数
            proba = [class_counts[0] / total_samples, class_counts[1] / total_samples]  # 计算概率
            return proba
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] == threshold:
            return self._predict_proba_one(x, tree['left'])
        else:
            return self._predict_proba_one(x, tree['right'])

    def get_tree(self):
        def replace_feature_names(node):
            if 'label' in node:
                return node
            feature_name = features[node['feature']]
            left_tree = replace_feature_names(node['left'])
            right_tree = replace_feature_names(node['right'])
            return {'feature': feature_name, 'threshold': node['threshold'], 'left': left_tree, 'right': right_tree}

        return replace_feature_names(self.tree_)

    def extract_rules(self, node=None, current_rule=None):
        if node is None:
            node = self.tree_
        if current_rule is None:
            current_rule = []

        if 'label' in node:
            if node['label'] == 1:
                return [current_rule]
            else:
                return []

        rules = []
        feature = features[node['feature']]
        threshold = node['threshold']

        left_rule = current_rule + [(feature, '==', threshold)]
        right_rule = current_rule + [(feature, '!=', threshold)]

        rules += self.extract_rules(node['left'], left_rule)
        rules += self.extract_rules(node['right'], right_rule)

        return rules


def predict(test_data_path, model):
    result_all = {}
    for test_file in os.listdir(test_data_path):
        test_df = pd.read_feather(f'{test_data_path}/{test_file}')     # test_df有logtime
        test_df.fillna(0, inplace=True)
        test_df['in_pos_parity_set'] = test_df['RetryRdErrLogParity'].apply(
            lambda x: 1 if int(x) in anomaly_parities else 0)
        test_df = test_df[features]     # 22维
        # test_df["sn_name"] = test_df.index.get_level_values(0)
        X_test = test_df.values
        predict_result = model.predict_proba(X_test)

        index_list = list(test_df.index)
        for i in tqdm(range(len(index_list)), test_file):
            p_s = predict_result[i][1]
            if p_s < 0.3:
                continue
            sn = index_list[i][0]
            # sn_t = datetime.fromtimestamp(index_list[i][1])    # 真实时间
            sn_t = index_list[i][1]          # 时间戳
            if sn not in result_all:
                result_all[sn] = [(sn_t, p_s)]
            else:
                result_all[sn].append((sn_t, p_s))
                # result_all[sn].append((sn_t, max(p_s, result_all[sn][-1][1])))
    return result_all


if __name__ == "__main__":
    train_pos = pd.read_feather(f'{train_data_path}/positive_train80.feather')
    train_neg = pd.read_feather(f'{train_data_path}/negative_train80.feather')
    train_all = pd.concat([train_pos, train_neg])
    train_all.fillna(0, inplace=True)
    train_all['in_pos_parity_set'] = train_all['RetryRdErrLogParity'].apply(
        lambda x: 1 if int(x) in anomaly_parities else 0)

    train_all = train_all[features + ['label']]
    train_all["sn_name"] = train_all.index.get_level_values(0)
    train_all.drop_duplicates(keep="first", inplace=True)
    train_all.drop(columns=['sn_name'], inplace=True)

    train_all["sn_name"] = train_all.index.get_level_values(0)
    X, y = train_all.drop(columns=['label']).values, train_all['label'].values
    all_sns_set = set(train_all['sn_name'])
    pos_sn_set = set(train_all[train_all['label'] == 1]['sn_name'])

    # model = CustomDecisionTreeClassifier(max_depth=max_depth,
    #                                      first_layer_feature=first_layer_feature,
    #                                      second_layer_feature=second_layer_feature)
    # model.fit(X, y)
    #
    # model_save_path = r"E:\pycharmproject\raw_code\work1"
    # with open(os.path.join(model_save_path, "time_point_train_lt80_model1-6.pkl"), "wb") as f:
    #     pickle.dump(model, f)
    # print(f"模型已经保存到{model_save_path}路径下")

    with open(r"E:\pycharmproject\raw_code\work1\time_point_train_lt80_model1-6.pkl", "rb") as f:
        model = pickle.load(f)

    result_all = predict(test_data_path, model)
    submission = []
    for sn in result_all:
        for i in result_all[sn]:
            submission.append([sn, i[0], i[1]])
    submission = pd.DataFrame(submission, columns=["sn_name", "prediction_timestamp", "probability"])
    submission.to_csv(os.path.join(result_path, "lt80_model1-6_time_point_submission0.3_month89.csv"), index=False)
