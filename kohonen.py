import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import multiprocessing




class Kohonen:

    def __init__(self, height, width, max_iterations, initial_learning_rate, workers, network=None, data=None, dataset_size=20, seed=42):
        self.HEIGHT = height
        self.WIDTH = width
        self.MAX_ITERATIONS = max_iterations
        self.INITIAL_LEARNING_RATE = initial_learning_rate
        self.WORKERS = workers

        self.network = network if network else self.init_network(seed)
        self.data = data if data else self.gen_data(dataset_size, seed)

        self.INITIAL_RADIUS = max(self.WIDTH, self.HEIGHT) / 2
        self.TIME_CONST = MAX_ITERATIONS / math.log10(self.INITIAL_RADIUS)

    def init_network(self, seed):
        """ Use a random seed for reproduceable experiments! """
        np.random.seed(seed)
        shape = (self.HEIGHT, self.WIDTH, 3)
        return np.random.random_sample(shape)

    def gen_data(self, size, seed):
        np.random.seed(seed)
        return np.random.random((1, size, 3))

    def euclidean_dist(self, a, b):
        d = np.subtract(a, b)
        d_2 = np.square(d)
        return math.sqrt(np.sum(d_2))

    def distance(self, node_x, node_y, bmu_x, bmu_y):
        x_2 = (node_x - bmu_x) ** 2
        y_2 = (node_y - bmu_y) ** 2
        d_2 = x_2 + y_2
        return math.sqrt(d_2)
    
    def get_learing_rate(self, timestep):
        return self.INITIAL_LEARNING_RATE * math.exp(-1 * timestep / self.TIME_CONST)
    
    def radius(self, timestep):
        return self.INITIAL_RADIUS * math.exp(-1 * timestep / self.TIME_CONST)
    
    def in_radius(self, centre_x, centre_y, node_x, node_y, r):
        """ return if node is located within the radius
            around `centre_x, centre_y` """
        return (node_x - centre_x) ** 2 + (node_y - centre_y) ** 2 < r ** 2
    
    def get_influence(self, node_x, node_y, bmu_x, bmu_y, timestep):
        d = self.distance(node_x, node_y, bmu_x, bmu_y)
        r = self.radius(timestep)
        return math.exp((-1 * d ** 2) / (2 * r ** 2))
    
    def update_weights(self, node_x, node_y, bmu_x, bmu_y, timestep):
        node = self.network[node_x][node_y]
        bmu = self.network[bmu_x][bmu_y]
        d = np.subtract(bmu, node)

        # calculate influence and learning rate
        influence = self.get_influence(node_x, node_y, bmu_x, bmu_y, timestep)
        learning_rate = self.get_learing_rate(timestep)
        scalar = influence * learning_rate
        correction = np.multiply(scalar, d)
        return np.add(node, correction)

    def get_bmu(self, in_vec):
        bmu = None
        min = float("inf")
        for col in range(len(self.network)):
            for row in range(len(self.network[col])):
                # use multiprocessing to parallelize euclidean distance calculation
                # pool = multiprocessing.Pool(self.WORKERS)
                dist = self.euclidean_dist(in_vec, self.network[col][row])
                if dist < min:
                    min = dist
                    bmu = (col, row)
        return bmu
    
    def train(self):
        """ Updates `network` for """
        for timestep in range(1, self.MAX_ITERATIONS):
            i = timestep % len(self.data[0])
            in_v = self.data[0][i]
            bmu = self.get_bmu(in_v)
            
            r = self.radius(timestep)
            for col in range(len(self.network)):
                for row in range(len(self.network[col])):
                    if self.in_radius(bmu[0], bmu[1], col, row, r):
                        self.network[col][row] = self.update_weights(col, row, bmu[0], bmu[1], timestep)

    def show(self):
        """ Graphical output of Kohonen network """
        plt.imshow(self.network)
        plt.show()
        return


if __name__ == "__main__":
    # Parse from config file -- for CI/CD and production deployment
    # Or from command line -- for running experiments while developing
    HEIGHT = 10
    WIDTH = 10
    MAX_ITERATIONS = 1000
    INITIAL_LEARNING_RATE = 0.1
    WORKERS = 10

    # We should be able to import a partially trained network and training data on disk
    DATA = None
    network = None
    k = Kohonen(HEIGHT, WIDTH, MAX_ITERATIONS, INITIAL_LEARNING_RATE, WORKERS)
    k.train()
    k.show()




# rgb = lambda x: np.array([x[0]/255, x[1]/255, x[2]/255])


# c = [np.array([255,102,102]), np.array([255,178,102]), np.array([178,255,102]), np.array([102,255,102]), np.array([102,255,178]), np.array([102,255,255
# ]), np.array([102,178,255]), np.array([102,102,255]), np.array([178,102,255]), np.array([255,102,255]), np.array([255, 102, 178])]

# x = []
# for i in c:
#     y = rgb(i)
#     x.append(y)
# data = np.array([x])
