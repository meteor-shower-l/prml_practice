import pandas as pd


class TreeNode:
    def __init__(self, feature=None, split_value=None, value=None):
        self.feature = feature
        self.split_value = split_value
        self.value = value
        self.left = None
        self.right = None


def square_loss(data: pd.Series):
    ave = data.mean()
    loss = ((data - ave) ** 2).sum()
    return loss


def find_best_point(data: pd.DataFrame, character: str, target: str):
    sorted_data = data[[character, target]].sort_values(by=character)
    best_loss = float("inf")
    best_point = None
    for i in range(0, len(data[character])):
        if i + 1 <= len(data[character]):
            left_target = sorted_data[target].iloc[0 : i + 1]
            right_target = sorted_data[target].iloc[i + 1 :]
        else:
            left_target = sorted_data[target]
            right_target = pd.Series()
        loss = square_loss(left_target) + square_loss(right_target)
        if loss < best_loss:
            best_loss = loss
            best_point = sorted_data[character].iloc[i]
    return best_point, best_loss


def find_best_character(data: pd.DataFrame, target: str):
    best_char = None
    best_point = None
    best_loss = float("inf")
    for character in data.columns:
        if character == target:
            continue
        point, loss = find_best_point(data, character, target)
        if point is None:
            continue
        if loss < best_loss:
            best_loss = loss
            best_point = point
            best_char = character
    return best_char, best_point


def generate_tree(
    data: pd.DataFrame,  # 原始数据
    target: str,
    min_num: int = 1,  # 用于指定每个节点最少样本数
    accessiable_layer: int = 1e8,  # 用于记录层数
):
    if len(data) < min_num or accessiable_layer < 0:
        return TreeNode(value=data[target].mean())
    else:
        character, point = find_best_character(data, target)
        if character is None or point is None:
            return TreeNode(value=data[target].mean())
        data_1 = data[data[character] <= point]
        data_2 = data[data[character] > point]
        if len(data_1) == 0 or len(data_2) == 0:
            return TreeNode(value=data[target].mean())
        Node = TreeNode(feature=character, split_value=point)
        Node.left = generate_tree(
            data_1, target, min_num, accessiable_layer - 1
        )
        Node.right = generate_tree(
            data_2, target, min_num, accessiable_layer - 1
        )
        return Node


def tree_to_str(node):
    if node.value is not None:
        return f"预测: {node.value:.2f}"
    left = tree_to_str(node.left)
    right = tree_to_str(node.right)
    return f"{{{node.feature} ≤ {node.split_value}:{left}, {node.feature} > {node.split_value}:{right}}}"


data = {
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00],
}
df = pd.DataFrame(data)
tree = generate_tree(df, target="y")
print(tree_to_str(tree))
