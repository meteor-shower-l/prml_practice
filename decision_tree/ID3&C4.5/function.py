import numpy as np
import pandas as pd


# 在给定数据、目标属性的前提下，用于计算熵
def calculate_entropy(data: pd.DataFrame, target_character: str):
    total_num = len(data[target_character])
    group_num = data.groupby(target_character)[target_character].count()
    proportion = group_num / total_num
    singel_entropy = -proportion * np.log2(proportion)
    result = singel_entropy.sum()
    return result


# 在给定数据、目标属性、划分属性的前提下，计算划分属性下的条件熵
def calculate_conditional_entropy(
    data: pd.DataFrame, target_character: str, condition_character: str
):
    # 获取分组结果
    grouped = data.groupby(condition_character)
    total_num = len(data)
    conditional_entropy = 0
    for _, group in grouped:
        entropy = calculate_entropy(group, target_character)
        conditional_entropy += entropy * len(group) / total_num
    return conditional_entropy


# 计算信息增益
def calculate_infa_gain(
    f_data: pd.DataFrame, f_target_character: str, f_condition_character: str
):
    info_gain = calculate_entropy(
        data=f_data, target_character=f_target_character
    ) - calculate_conditional_entropy(
        data=f_data,
        target_character=f_target_character,
        condition_character=f_condition_character,
    )
    return info_gain


# 计算信息增益比
def calculate_infa_gain_ratio(
    f_data: pd.DataFrame, f_target_character: str, f_condition_character: str
):
    info_gain = calculate_entropy(
        data=f_data, target_character=f_target_character
    ) - calculate_conditional_entropy(
        data=f_data,
        target_character=f_target_character,
        condition_character=f_condition_character,
    )
    result = info_gain / calculate_entropy(
        data=f_data, target_character=f_condition_character
    )
    return result
