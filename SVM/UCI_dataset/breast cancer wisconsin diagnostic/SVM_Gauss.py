from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed()


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data = data.iloc[:, 1:]  # 去掉第一列（ID列）
    y = data.iloc[:, 0]  # 第一列现在是标签
    X = data.iloc[:, 1:]  # 其余是特征
    return X, y


# 读取数据并进行数据预处理
current_path = Path(__file__).resolve().parent
data_dir = current_path / "wdbc.csv"
X, y = load_data(data_dir)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 将y的标签变为0,1
label_mapping = {"B": 0, "M": 1}
y_encoded = np.array([label_mapping[label] for label in y])
# 进行数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 定义并训练模型
svm_gauss = SVC(kernel="rbf", gamma="scale", C=1, probability=True, random_state=42)
svm_gauss.fit(X_train, y_train)

# 测试指标
y_pred = svm_gauss.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary", pos_label=1)
recall = recall_score(y_test, y_pred, average="binary", pos_label=1)
f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)

print(f"accuracy:{accuracy}")
print(f"precision:{precision}")
print(f"recall:{recall}")
print(f"f1:{f1}")
