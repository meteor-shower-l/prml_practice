import pandas as pd


class TreeNode:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.value = value
        self.children = {}


def square_loss(data: pd.Series):
    ave = data.mean()
    loss = ((data - ave) ** 2).sum()
    return loss


def find_best_point(data: pd.DataFrame, character: str, target: str):
    sorted_data = data[[character, target]].sort_values(by=character)
    best_loss = float("inf")
    best_point = None
    for i in range(1, len(data[character])):
        left_target = sorted_data[target].iloc[:i]
        right_target = sorted_data[target].iloc[i:]
        loss = square_loss(left_target) + square_loss(right_target)
        if loss < best_loss:
            best_loss = loss
            best_point = sorted_data[character].iloc[i]
    return best_point, best_loss


def find_best_character(data: pd.DataFrame, target: str):
    character_list = [col for col in data.columns if col != target]
    point_list = []
    loss_list = []
    for i in range(len(character_list)):
        point, loss = find_best_point(data, character_list[i], target=target)
        point_list.append(point)
        loss_list.append(loss)
    index = loss_list.index(min(loss_list))
    character = character_list[index]
    point = point_list[index]
    return character, point


def generate_tree(
    data: pd.DataFrame,  # 原始数据
    current_layer: float,
    min_num: int = 0,  # 用于指定每个节点最少样本数
    max_layer: int = 1e8,  # 用于指定最大层数
    min_loss: float = 0,  # 用于指定loss阈值
):
    pass
