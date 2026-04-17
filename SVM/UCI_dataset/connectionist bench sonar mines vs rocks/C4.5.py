import math

import numpy as np
import pandas as pd


class node:
    def __init__(self):
        self.is_leaf = None
        self.pred_class = None
        self.split_feature = None
        self.split_feature_point = None
        self.left = None
        self.right = None


# 给定一个标签数组，计算熵
def calculate_entropy(labels: np.ndarray):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:  # 避免log2(0)的情况
            entropy -= prob * math.log2(prob)

    return entropy


# 用于计算在给定划分点下，特征数组的熵(用于计算信息增益比)
def calculate_split(features: np.ndarray, split_point: float):
    # 得到值小于划分值的掩码
    smaller_mask = features <= split_point
    # 得到值大于划分值的掩码
    bigger_mask = features > split_point
    prob_1 = np.sum(smaller_mask) / len(features)
    prob_2 = np.sum(bigger_mask) / len(features)
    entropy = 0.0
    if prob_1 > 0:
        entropy -= prob_1 * math.log2(prob_1)
    if prob_2 > 0:
        entropy -= prob_2 * math.log2(prob_2)
    return entropy


# 给定一个特征数组、切分点、标签数组，计算条件熵
def calculate_conditional_entropy(
    features: np.ndarray,  # 特征数组
    split_point: float,  # 切分点
    labels: np.ndarray,  # 标签数组
):
    total_num = len(labels)
    # 得到值小于划分值的掩码
    smaller_mask = features <= split_point
    # 得到值大于划分值的掩码
    bigger_mask = features > split_point
    # 得到值小于划分值部分的标签
    smaller_labels = labels[smaller_mask]
    # 得到值大于划分值部分的标签
    bigger_labels = labels[bigger_mask]
    # 得到两边的条件熵
    smaller_entropy = (
        len(smaller_labels) / total_num * calculate_entropy(smaller_labels)
    )
    bigger_entropy = len(bigger_labels) / total_num * calculate_entropy(bigger_labels)
    conditional_entropy = smaller_entropy + bigger_entropy
    return conditional_entropy


# 给定一个特征数组、标签数组、切分点，计算信息增益
def calculate_info_gain(
    features: np.ndarray,  # 特征数组
    split_point: float,  # 切分点
    labels: np.ndarray,  # 标签数组
):
    labels_entropy = calculate_entropy(labels)
    conditional_entropy = calculate_conditional_entropy(features, split_point, labels)
    information_gain = labels_entropy - conditional_entropy
    return information_gain


# 给定一个特征数组、标签数组、切分点，计算信息增益比
def calculate_info_gain_ratio(
    features: np.ndarray,  # 特征数组
    split_point: float,  # 切分点
    labels: np.ndarray,  # 标签数组
    epsilon: float = 1e-10,
):
    labels_entropy = calculate_entropy(labels)
    conditional_entropy = calculate_conditional_entropy(features, split_point, labels)
    information_gain = labels_entropy - conditional_entropy
    split_entropy = calculate_split(features, split_point)
    if split_entropy < epsilon or information_gain < epsilon:
        return 0.0
    return information_gain / split_entropy


# 在单个连续特征上寻找最优二分点，返回(最优切分点, 最优信息增益比)
def find_best_split_for_feature(
    feature_values: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-10,
):
    feature_values = np.asarray(feature_values)
    labels = np.asarray(labels)

    unique_values = np.unique(feature_values)
    # 若特征没有可二分的候选点，则视为无效划分
    if len(unique_values) < 2:
        return None

    candidate_split_points = (unique_values[:-1] + unique_values[1:]) / 2.0
    best_gain_ratio = -1.0
    best_split_point = None

    for split_point in candidate_split_points:
        current_gain_ratio = calculate_info_gain_ratio(
            feature_values, float(split_point), labels, epsilon
        )
        if current_gain_ratio > best_gain_ratio:
            best_gain_ratio = current_gain_ratio
            best_split_point = float(split_point)

    if best_split_point is None or best_gain_ratio < epsilon:
        return None

    return best_split_point, best_gain_ratio


# 在所有特征上选择全局最优划分属性，返回(特征名, 切分点, 信息增益比)
def choose_best_feature(
    X: pd.DataFrame,
    y: np.ndarray,
    epsilon: float = 1e-10,
):
    best_feature = None
    best_split_point = None
    best_gain_ratio = -1.0

    for feature_name in X.columns:
        feature_values = X[feature_name].to_numpy()
        result = find_best_split_for_feature(feature_values, y, epsilon)
        if result is None:
            continue

        split_point, gain_ratio = result
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = feature_name
            best_split_point = split_point

    if best_feature is None or best_split_point is None:
        return None

    return best_feature, best_split_point, best_gain_ratio


# 用于在得到树之后，对给定样本类别进行预测
def predict(current_node: node, sample: dict):
    if current_node.is_leaf is True:
        return current_node.pred_class
    else:
        if sample[current_node.split_feature] <= current_node.split_feature_point:
            return predict(current_node.left, sample)  # type: ignore
        else:
            return predict(current_node.right, sample)  # type: ignore


# 递归生成 C4.5 决策树
def build_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    epsilon: float = 1e-10,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    min_gain_ratio: float = 0.0,
    current_depth: int = 0,
):
    y = np.asarray(y)
    current_node = node()

    if len(y) == 0:
        raise ValueError("y 不能为空")

    # 当前节点默认类别使用多数类，作为停止时与预测异常时的兜底
    unique_labels, counts = np.unique(y, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]
    current_node.pred_class = majority_class

    # 停止条件1：纯节点
    if len(unique_labels) == 1:
        setattr(current_node, "is_leaf", True)
        return current_node

    # 停止条件2：达到最大深度
    if max_depth is not None and current_depth >= max_depth:
        setattr(current_node, "is_leaf", True)
        return current_node

    # 停止条件3：样本数小于最小继续划分样本数
    if len(y) < min_samples_split:
        setattr(current_node, "is_leaf", True)
        return current_node

    # 停止条件4：不存在有效划分
    best_split_result = choose_best_feature(X, y, epsilon)
    if best_split_result is None:
        setattr(current_node, "is_leaf", True)
        return current_node

    split_feature, split_point, best_gain_ratio = best_split_result

    # 停止条件5：最优划分增益率不足
    if best_gain_ratio < min_gain_ratio:
        setattr(current_node, "is_leaf", True)
        return current_node

    setattr(current_node, "is_leaf", False)
    setattr(current_node, "split_feature", split_feature)
    setattr(current_node, "split_feature_point", split_point)

    split_feature_values = X[split_feature].to_numpy()
    left_mask = split_feature_values <= split_point
    right_mask = split_feature_values > split_point

    # 安全兜底：若出现无效切分，转为叶子
    if (not np.any(left_mask)) or (not np.any(right_mask)):
        setattr(current_node, "is_leaf", True)
        setattr(current_node, "split_feature", None)
        setattr(current_node, "split_feature_point", None)
        return current_node

    # 停止条件6：划分后任一子节点样本数小于叶子最小样本数
    if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
        setattr(current_node, "is_leaf", True)
        setattr(current_node, "split_feature", None)
        setattr(current_node, "split_feature_point", None)
        return current_node

    X_left = X.loc[left_mask]
    y_left = y[left_mask]
    X_right = X.loc[right_mask]
    y_right = y[right_mask]

    setattr(
        current_node,
        "left",
        build_tree(
            X_left,
            y_left,
            epsilon,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_gain_ratio,
            current_depth + 1,
        ),
    )
    setattr(
        current_node,
        "right",
        build_tree(
            X_right,
            y_right,
            epsilon,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_gain_ratio,
            current_depth + 1,
        ),
    )
    return current_node
