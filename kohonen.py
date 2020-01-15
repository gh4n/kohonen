import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import multiprocessing

class Kohonen:

    def __init__(self):
        # parse from config file -- for production deployment 
        # OR from command line -- for running experiments while developing
        self.HEIGHT = 100
        self.WIDTH = 100
        self.MAX_ITERATIONS 
        self.INITIAL_RADIUS = max(self.WIDTH, self.HEIGHT) / 2
        self.TIME_CONST = MAX_ITERATIONS / math.log10(self.INITIAL_RADIUS)
        self.INITIAL_LEARNING_RATE = 0.1
        self.WORKERS = 10
        self.network = None
        self.data = None

    def init_network(self, seed):
        np.random.seed(seed)
        shape = (self.HEIGHT, self.WIDTH, 3)
        return np.random.random_sample(shape)

    def gen_data(self, size, seed=None):
        np.random.seed(seed)
        return np.random.random((1, size, 3))

    def euclidean_dist(self, a, b):
        d = np.subtract(a, b)
        d_2 = np.square(d)
        return math.sqrt(np.sum(d_2))

    def distance(self, w_x, w_y, z_x, z_y):
        x_2 = (w_x - z_x) ** 2
        y_2 = (w_y - z_y) ** 2
        d_2 = x_2 + y_2
        return math.sqrt(d_2)
    
    def learing_rate(self, timestep):
        return INITIAL_LEARNING_RATE * math.exp(-1 * timestep / TIME_CONST)
    
    def radius(self, timestep):
        return INITIAL_RADIUS * math.exp(-1 * timestep / TIME_CONST)
    
    # return if node is located within the radius
    # around node (centre)
    def in_radius(self, x_c, y_c, x_n, y_n, r):
        return (x_n - x_c) ** 2 + (y_n - y_c) ** 2 < r ** 2
    
    # influence decay
    def influence(self, w_x, w_y, z_x, z_y, t):
        d = distance(w_x, w_y, z_x, z_y)
        r = radius(t)
        return math.exp((-1 * d ** 2) / (2 * r ** 2))
    
    # update weights
    def update(self, network, w_x, w_y, v_x, v_y, t):
        w = network[w_x][w_y]
        v = network[v_x][v_y]
        d = np.subtract(v, w)
        inf = influence(w_x, w_y, v_x, v_y, t)
        lr = learing_rate(t)
        scalar = lr * inf
        correction = np.multiply(scalar, d)
        return np.add(w, correction)

    def bmu(self, in_vec, network):
        min = float("inf")
        b = None
        for i in range(len(network)):
            for j in range(len(network[i])):
                dist = distance_weight(in_vec, network[i][j])
                if dist < min:
                    min = dist
                    b = (network[i][j], i, j)
                # print(f"euclidean dist: {distance_weight(in_vec, network[i][j])}")
        # print(bmu)
        return b
    
    def train(self, network, data):
        for t in range(1, self.MAX_ITERATIONS):
            ind = t % len(data[0])
            in_vec = data[0][ind]
            b = bmu(in_vec, network)
            print(f" invec {in_vec}, bmu {b}")
            
            r = radius(t)
            for i in range(len(network)):
                for j in range(len(network[i])):
                    if in_radius(b[1],b[2], i, j, r):
                        prev = network[i][j]
                        up = update(network, i, j, b[1], b[2], t)
                        print(f"prev {prev}, up {up}")
                        network[i][j] = up



r = 5
for i in range(len(network)):
    for j in range(len(network[i])):
        if in_radius(5, 5, i, j, r):
            network[i][j] = np.array([0, 0, 0])

plt.imshow(network)
plt.show()

            

print(data)
plt.imshow(data)
# plt.imshow(network)
plt.show()
#%%


# bmu = bmu(data[0], network)
# print(bmu)
# r = radius(1)
# in_r = []
# for i in range(len(network)):
#     for j in range(len(network[i])):
#         if in_radius(2, 5, i, j, r):
#             in_r.append((i, j))
            
#             print(f"UPDATED {update(bmu[0], network[i][j], bmu[1], bmu[2], i, j, 1)}")
            # network[i][j] = np.array([0, 0, 0])
#%%

# print(in_r)
# print(network)
train(network, data)
# print(network)
# print(data)
print(data)

plt.imshow(data)
plt.imshow(network)
plt.show()
#%%

#%%
rgb = lambda x: np.array([x[0]/255, x[1]/255, x[2]/255])


c = [np.array([255,102,102]), np.array([255,178,102]), np.array([178,255,102]), np.array([102,255,102]), np.array([102,255,178]), np.array([102,255,255
]), np.array([102,178,255]), np.array([102,102,255]), np.array([178,102,255]), np.array([255,102,255]), np.array([255, 102, 178])]

x = []
for i in c:
    y = rgb(i)
    x.append(y)
data = np.array([x])





# %%
