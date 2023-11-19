from random import choice
import matplotlib.pyplot as plt

class RandomWalk_1d():
    def __init__(self,num_points = 50):
        self.num_points = num_points
        self.value = [0]
    def fill_walk(self):
        while len(self.value) <= self.num_points:
            step = choice([-1,1])
            next_value = self.value[-1] + step
            self.value.append(next_value)

rw = RandomWalk_1d()
rw.fill_walk()

# Visualization
axis_x = range(0,len(rw.value))
ax = plt.subplot()
ax.set_title('Example for 1-d random walk')
ax.plot(axis_x, rw.value)
ax.set_xlabel('Time')
ax.set_ylabel('Location')
plt.show()

