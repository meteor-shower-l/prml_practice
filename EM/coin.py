import numpy as np


def solve_3coin(labels: np.ndarray, pi: float, p: float, q: float, epsilon):
    total_num = len(labels)
    positive_num = np.sum(labels)
    nagetive_num = total_num - np.sum(labels)
    times = 0
    while True:
        times += 1
        # 保存原先值(用于稍后计算相对更新量)
        init_pi = pi
        init_p = p
        init_q = q
        # 更新参数
        miu_1 = (pi * p) / (pi * p + (1 - pi) * q)
        miu_2 = (pi * (1 - p)) / (pi * (1 - p) + (1 - pi) * (1 - q))
        temp = positive_num * miu_1 + nagetive_num * miu_2
        pi = temp / total_num
        p = np.sum(labels * miu_1) / temp
        q = np.sum(labels * (1 - miu_1)) / (total_num - temp)
        # 计算相对更新量
        pi_change = abs(init_pi - pi) * 2 / (init_pi + pi)
        p_change = abs(init_p - p) * 2 / (init_p + p)
        q_change = abs(init_q - q) * 2 / (init_q + q)
        print(f"""
完成第{times}轮迭代:
pi:{pi}
p:{p}
q:{q}
相对更新量:
pi:{(pi_change * 100):.2f}%
p:{(p_change * 100):.2f}%
q:{(q_change * 100):.2f}%""")
        if (
            pi_change < epsilon and p_change < epsilon and q_change < epsilon
        ) or times > 100:
            break

    return pi, p, q


labels = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
pi_0 = 0.46
p_0 = 0.55
q_0 = 0.67
epsilon = 0.01
solve_3coin(labels, pi_0, p_0, q_0, epsilon)
