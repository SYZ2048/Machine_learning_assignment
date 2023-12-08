import numpy as np


def perform_random_walk(steps):
    """执行指定步数的随机漫步，并返回终点位置"""
    walk = np.random.choice([-1, 1], size=steps)
    return np.sum(walk)


def analyze_random_walks(num_steps, num_walks):
    """分析指定步数和模拟次数的随机漫步"""
    endpoints = [perform_random_walk(num_steps) for _ in range(num_walks)]
    # 终点的平均位置
    avg_position = np.mean(endpoints)
    # 回到原点的概率
    return_to_origin_prob = np.sum(np.array(endpoints) == 0) / num_walks
    # 终点离原点的平均距离
    avg_distance = np.mean(np.abs(endpoints))
    return avg_position, return_to_origin_prob, avg_distance


# 设定漫步步数和模拟次数
steps_list = [50, 100, 200, 500, 1000]
num_walks = 10000

# 分析每个步数的随机漫步
results = {steps: analyze_random_walks(steps, num_walks) for steps in steps_list}
results
