import numpy as np
from scipy.optimize import minimize

# 输入格式:(m,n)的array,其中m为样本数，n为特征数


# 线性核
def Linear_kernel(x1: np.array, x2: np.array):
    return np.dot(x1, x2)


# 高斯核
def generate_Gaussian_kernel(sigma: float):
    def Gaussian_kernel(x1: np.array, x2: np.array):
        distance_sq = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-distance_sq / (2 * sigma**2))

    return Gaussian_kernel


# 求出用于表示待优化函数的矩阵
def calculate(x1_list: np.array, x2_list: np.array, kernel_function):
    n1 = x1_list.shape[0]
    n2 = x2_list.shape[0]
    total_shape = n1 + n2
    x_list = np.vstack((x1_list, x2_list))
    H = np.zeros((total_shape, total_shape))
    y = np.concatenate([np.ones(n1), -np.ones(n2)])
    for i in range(total_shape):
        for j in range(total_shape):
            H[i][j] = kernel_function(x_list[i], x_list[j]) * y[i] * y[j]
    c_vector = np.ones(total_shape)
    return H, c_vector


# 求解问题
def get_alphas(x1_list: np.array, x2_list: np.array, kernel_function, C=None):
    n1 = x1_list.shape[0]
    n2 = x2_list.shape[0]
    H, c_vector = calculate(x1_list, x2_list, kernel_function)

    # 得到目标函数
    def objective(alpha):
        return 0.5 * alpha @ H @ alpha - c_vector @ alpha

    # 等式约束
    def constraint(alpha):
        return np.sum(alpha[:n1]) - np.sum(alpha[n1:])

    # 不等式约束
    bounds = [(0, C) for _ in range(0, n1 + n2)]
    con = {"type": "eq", "fun": constraint}

    x0 = np.zeros(n1 + n2)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=con)
    print(f"最优值: f(α) = {result.fun}")
    print(f"是否成功: {result.success}")
    return result.x


# 得到的优化的α后，计算ω和b
def calculate_w_b(alpha, x1_list: np.array, x2_list: np.array, kernel_function):
    n1 = x1_list.shape[0]
    n2 = x2_list.shape[0]
    total_shape = n1 + n2
    x_list = np.vstack((x1_list, x2_list))
    y = np.concatenate([np.ones(n1), -np.ones(n2)])
    y.astype(np.float64)
    w = np.zeros_like(x1_list[0], dtype=np.float64)
    b = 1
    # 计算w
    for i in range(0, total_shape):
        w += alpha[i] * x_list[i] * y[i]
        b -= alpha[i] * kernel_function(x_list[0], x_list[i]) * y[i]
    return w, b


x1_list = np.array(
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [2.0, 2.5],
        [1.5, 2.5],
    ]
)
x2_list = np.array(
    [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0], [3.0, 2.5], [1.8, 1.8]]
)
kernel_function = Linear_kernel
alpha = np.round(get_alphas(x1_list, x2_list, kernel_function, C=1), 4)
w, b = calculate_w_b(
    alpha,
    x1_list,
    x2_list,
    kernel_function,
)
print(f"w:{w}")
print(f"b:{b}")
