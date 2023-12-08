from random import choice
import matplotlib.pyplot as plt
from scipy.special import comb

class RandomWalk_1d():
    def __init__(self, num_points=50):
        self.num_points = num_points  # number of walks
        self.value = [0]				# 漫步过程的每一个值
        self.end_point = 0				# 终点值

    def fill_walk(self):
        while len(self.value) <= self.num_points:
            step = choice([-1, 1])
            next_value = self.value[-1] + step
            self.value.append(next_value)
        self.end_point = next_value


def expected_absolute_distance(n):
    total = 0
    for k in range(0, n + 1, 2):
        prob = comb(n, (n + k) // 2) / (2 ** n)
        total += k * prob
    return total*2


steps_list = [50, 100, 200, 500, 1000]
num_walks = 1000
end_points = []
# step = 1000
for idx, step in enumerate(steps_list):     # 对于不同步长
    end_points.append([])
    for i in range(num_walks):
        rw = RandomWalk_1d(num_points=step)
        rw.fill_walk()
        end_points[idx].append(rw.end_point)    # 固定step。每次漫步的终点值存储为一个list

    abs_distance = [abs(number) for number in end_points[idx]]
    avg_distance = sum(abs_distance) / num_walks
    print(f"<step: {step}> Probability of returning to the origin: ", end_points[idx].count(0) / num_walks)
    print(f"<step: {step}> Expected distance to the origin: ", avg_distance)
    print("P = ", comb(step, step/2) * ((1/2) ** step))
    print("E = ", expected_absolute_distance(step))

    # Visualization
    # axis_x = range(0, num_walks)
    # ax = plt.subplot()
    # ax.set_title(f'{num_walks} times random walk step<{step}>')
    # ax.scatter(axis_x, end_points[idx])
    # ax.set_xlabel('num_walks')
    # ax.set_ylabel('Location')
    # plt.show()




# rw = RandomWalk_1d()
# rw.fill_walk()
# Visualization
# axis_x = range(0,len(rw.value))
# ax = plt.subplot()
# ax.set_title('Example for 1-d random walk')
# ax.plot(axis_x, rw.value)
# ax.set_xlabel('Time')
# ax.set_ylabel('Location')
# plt.show()
