import numpy as np
from scipy.optimize import minimize

# 将目标函数写为矩阵形式
H = np.array(
    [
        [5, 8, 9, -4, -7],
        [8, 13, 15, -7, -12],
        [9, 15, 18, -9, -15],
        [-4, -7, -9, 5, 8],
        [-7, -12, -15, 8, 13],
    ]
)
c = np.array([-1, -1, -1, -1, -1])


def objective(alpha):
    """目标函数：f(α) = 0.5αᵀHα + cᵀα"""
    return 0.5 * alpha @ H @ alpha + c @ alpha


def constraint(alpha):
    """等式约束：α1 + α2 + α3 - α4 - α5 = 0"""
    return alpha[0] + alpha[1] + alpha[2] - alpha[3] - alpha[4]


bounds = [(0, None) for _ in range(5)]
con = {"type": "eq", "fun": constraint}
x0 = np.array([0.2, 0.2, 0.2, 0.3, 0.3])
# 求解
result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=con)
print("优化结果:")
print(f"最优解: α = {result.x}")
print(f"最优值: f(α) = {result.fun}")
print(f"是否成功: {result.success}")
print(f"迭代次数: {result.nit}")

# 验证约束
print("\n约束验证:")
print(f"α1+α2+α3-α4-α5 = {constraint(result.x):.2e}")
print(f"所有变量>0: {np.all(result.x > 0)}")
