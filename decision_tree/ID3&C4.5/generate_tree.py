import function
import pandas as pd


class TreeNode:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.value = value
        self.children = {}


def generate_tree(
    f_data: pd.DataFrame,
    f_target_character: str,
    column_list: list,
    Epsilon: float,
    model=0,
):

    # 确定模式
    if model == 0:
        value_function = function.calculate_infa_gain
    else:
        value_function = function.calculate_infa_gain_ratio
    # 初始化column_list
    column_list = f_data.columns.to_list()
    # 终止条件
    # 如果数据集中所有目标值相同,则返回叶节点
    if len(f_data[f_target_character].unique()) == 1:
        return TreeNode(value=f_data[f_target_character].iloc[0])
    # 如果没有特征可用,则返回叶节点,值为多数类
    if not column_list:
        majority_class = f_data[f_target_character].mode()[0]
        return TreeNode(value=majority_class)

    # 计算信息增益/增益比
    gains = [
        value_function(f_data, f_target_character, col) for col in column_list
    ]
    # 若最大增益/增益比小于指定值,则返回叶节点,值为多数类
    if max(gains) < Epsilon:
        majority_class = f_data[f_target_character].mode()[0]
        return TreeNode(value=majority_class)
    else:
        best_feature = column_list[gains.index(max(gains))]
        node = TreeNode(feature=best_feature)
        for value in f_data[best_feature].unique():
            sub_data = f_data[f_data[best_feature] == value]
            new_columns = [
                column for column in column_list if column != best_feature
            ]
            node.children[value] = generate_tree(
                sub_data, f_target_character, new_columns, Epsilon, model
            )
