# 代码完全由元宝生成
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 数据点
positive_points = np.array([[1, 2], [2, 3], [3, 3]])  # 正例
negative_points = np.array([[2, 1], [3, 2]])  # 负例
support_vectors = np.array([[1, 2], [3, 3], [3, 2]])  # 支持向量


# 分离超平面: -x1 + 2*x2 - 2 = 0
# 重写为: x2 = (x1 + 2) / 2
def decision_boundary(x1):
    return (x1 + 2) / 2


# 间隔边界:
# 上边界: -x1 + 2*x2 - 2 = 1  => x2 = (x1 + 3) / 2
# 下边界: -x1 + 2*x2 - 2 = -1 => x2 = (x1 + 1) / 2
def margin_upper(x1):
    return (x1 + 3) / 2


def margin_lower(x1):
    return (x1 + 1) / 2


# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")

# 设置坐标轴范围
x_min, x_max = 0, 4
y_min, y_max = 0, 4
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])

# 生成x值用于绘制直线
x_vals = np.linspace(x_min, x_max, 100)

# 绘制分离超平面（决策边界）
ax.plot(
    x_vals,
    decision_boundary(x_vals),
    "k-",
    linewidth=3,
    label="分离超平面: $-x_1+2x_2-2=0$",
)

# 绘制间隔边界
ax.plot(
    x_vals,
    margin_upper(x_vals),
    "b--",
    linewidth=2,
    alpha=0.7,
    label="间隔上边界: $-x_1+2x_2-2=1$",
)
ax.plot(
    x_vals,
    margin_lower(x_vals),
    "r--",
    linewidth=2,
    alpha=0.7,
    label="间隔下边界: $-x_1+2x_2-2=-1$",
)

# 绘制正例点（用三角形表示）
ax.scatter(
    positive_points[:, 0],
    positive_points[:, 1],
    c="red",
    s=200,
    marker="^",
    edgecolors="k",
    linewidths=1.5,
    label="正例 (+1)",
    zorder=5,
)

# 绘制负例点（用圆形表示）
ax.scatter(
    negative_points[:, 0],
    negative_points[:, 1],
    c="blue",
    s=200,
    marker="o",
    edgecolors="k",
    linewidths=1.5,
    label="负例 (-1)",
    zorder=5,
)

# 高亮显示支持向量
for i, sv in enumerate(support_vectors):
    # 给支持向量添加更大的标记
    ax.scatter(
        sv[0],
        sv[1],
        c="gold",
        s=300,
        marker="*",
        edgecolors="k",
        linewidths=2,
        zorder=6,
        alpha=0.8,
    )
    # 添加标签
    ax.text(
        sv[0] + 0.1,
        sv[1] + 0.1,
        f"SV{i + 1}",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
    )

# 填充间隔区域
x_fill = np.linspace(x_min, x_max, 100)
y_upper = margin_upper(x_fill)
y_lower = margin_lower(x_fill)
ax.fill_between(x_fill, y_lower, y_upper, color="gray", alpha=0.2, label="间隔区域")

# 标记正例区域和负例区域
x_sample = 0.5
y_sample_pos = decision_boundary(x_sample) + 0.5
y_sample_neg = decision_boundary(x_sample) - 0.5

ax.text(
    x_sample,
    y_sample_pos,
    "正例区域\n$f(x)>0$",
    fontsize=12,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2),
)
ax.text(
    x_sample,
    y_sample_neg,
    "负例区域\n$f(x)<0$",
    fontsize=12,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2),
)

# 添加坐标网格
ax.grid(True, alpha=0.3, linestyle="--")

# 设置坐标轴标签
ax.set_xlabel("特征 $x_1$", fontsize=14, fontweight="bold")
ax.set_ylabel("特征 $x_2$", fontsize=14, fontweight="bold")

# 设置标题
ax.set_title(
    "线性可分SVM可视化\n决策函数: $f(x) = -x_1 + 2x_2 - 2$",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# 添加图例
legend = ax.legend(loc="upper left", fontsize=12, framealpha=0.9)
legend.get_frame().set_facecolor("#f0f0f0")

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 打印详细信息
print("=" * 60)
print("SVM详细信息：")
print("=" * 60)
print("1. 数据点:")
print(f"   正例 (+1): {positive_points.tolist()}")
print(f"   负例 (-1): {negative_points.tolist()}")
print()
print("2. 支持向量:")
for i, sv in enumerate(support_vectors):
    f_x = -sv[0] + 2 * sv[1] - 2
    print(f"   SV{i + 1}: {sv}, f(x) = {f_x}")
print()
print("3. 分离超平面:")
print("   -x₁ + 2x₂ - 2 = 0")
print("   可重写为: x₂ = (x₁ + 2) / 2")
print()
print("4. 间隔边界:")
print("   上边界: -x₁ + 2x₂ - 2 = 1  => x₂ = (x₁ + 3) / 2")
print("   下边界: -x₁ + 2x₂ - 2 = -1 => x₂ = (x₁ + 1) / 2")
print("=" * 60)

# 验证支持向量
print("\n验证支持向量:")
for sv in support_vectors:
    f_val = -sv[0] + 2 * sv[1] - 2
    if abs(f_val) == 1:
        print(f"   {sv} 是支持向量，f(x) = {f_val}，在间隔边界上")
    else:
        print(f"   {sv} 不是支持向量，f(x) = {f_val}")
