import numpy as np
import matplotlib.pyplot as plt
import math


# constants
HEIGHT = 10
WIDTH = 10
MAX_ITERATIONS = 100
INITIAL_RADIUS = max(WIDTH, HEIGHT) / 2
TIME_CONST = MAX_ITERATIONS / math.log(INITIAL_RADIUS)
INITIAL_LEARNING_RATE = 0.1

# initialize random test data
np.random.seed(101)
data = np.random.random((20, 3))

# initialize network
shape = (HEIGHT, WIDTH, 3)
network = np.random.random_sample(shape)

# returns euclidean distance between two vectors - a, b
def distance(a, b):
    d = np.subtract(a, b)
    d_2 = np.square(d)
    return np.sum(d_2)

def learning_rate(t):
    return math.exp(INITIAL_LEARNING_RATE, -1 * t / TIME_CONST)

def x(n):
    return n // WIDTH

def y(n):
    return n % WIDTH


x = np.array([1, 2, 3])
y = np.array([10, 14, 15])
d = np.subtract(x, y)




# print(data)
# plt.imshow(data)
# plt.imshow(network)
# plt.show()


