import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import multiprocessing as mp
from pathlib import Path
import os


class Kohonen:

    def __init__(self, height, width, max_iterations, initial_learning_rate, workers, network=None, data=None, dataset_size=20, seed=43):
        self.HEIGHT = height
        self.WIDTH = width
        self.MAX_ITERATIONS = max_iterations
        self.INITIAL_LEARNING_RATE = initial_learning_rate
        self.WORKERS = workers

        self.network = network if network else self.init_network(seed ** 2)
        self.data = data if data else self.gen_data(dataset_size, seed)

        self.INITIAL_RADIUS = max(self.WIDTH, self.HEIGHT) / 2
        self.TIME_CONST = MAX_ITERATIONS / math.log10(self.INITIAL_RADIUS)

    def init_network(self, seed):
        """ Use a random seed for reproduceable experiments! """
        np.random.seed(seed)
        shape = (self.HEIGHT, self.WIDTH, 3)
        return np.random.random_sample(shape)

    def gen_data(self, size, seed):
        """ Generates dataset of `size` RGB values """
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
        """ Adds a correction to the weights of node at `node_x` and `node_y` using
            the `learning_rate`, `influence` and its distance from the BMU """
        node = self.network[node_x][node_y]
        bmu = self.network[bmu_x][bmu_y]
        d = np.subtract(bmu, node)

        # calculate influence and learning rate
        influence = self.get_influence(node_x, node_y, bmu_x, bmu_y, timestep)
        learning_rate = self.get_learing_rate(timestep)

        # adjust node weight
        scalar = influence * learning_rate
        correction = np.multiply(scalar, d)
        return np.add(node, correction)


    def _worker_bmu(self, id, queue, in_vec):
        """ Partitions network into n = `self.worker` chunks,
            each worker `id` finds the BMU in its chunk """
        # compute partition size
        size = self.WIDTH // self.WORKERS
        offset = self.WIDTH % self.WORKERS
        
        # set `start` and `end` indices
        start = id * size
        end = size * (id + 1) if id != (self.WORKERS - 1) else size * (id + 1) + offset
        
        bmu = None
        min_dist = float("inf")

        # iterate over each node in chunk, looking for closest eucliden dist, i.e local BMU
        for col in range(len(self.network)):
            for row in range(start, end):
                dist = self.euclidean_dist(in_vec, self.network[col][row])
                if dist < min_dist:
                    min_dist = dist
                    bmu = (dist, [col, row])

        # add local BMU to shared queue
        queue.put(bmu)

    def get_bmu(self, in_vec):
        """ Finds BMU with multiple processes """

        # initalizes a shared queue and distributes workers to `self._worker_bmu`
        queue = mp.Queue()
        procs = [mp.Process(target=self._worker_bmu, args=(id, queue, in_vec)) for id in range(self.WORKERS)]

        for proc in procs: 
            proc.start()
        
        for proc in procs:
            proc.join()

        # retrieves the BMU for each partition
        closest = [queue.get() for _ in procs]

        # return co-ords of global BMU
        return min(closest)[1]    
    
    def train(self):
        """ Loops `self.MAX_ITERATIONS` times to adjust the weights of the network """
        for timestep in range(1, self.MAX_ITERATIONS):
            i = timestep % len(self.data[0])

            # get input_vector from data, then find BMU
            in_vec = self.data[0][i]
            bmu = self.get_bmu(in_vec)
            
            r = self.radius(timestep)
            
            # iterate over all nodes and check if they're in the radius of the BMU
            for col in range(len(self.network)):
                for row in range(len(self.network[col])):
                    if self.in_radius(bmu[0], bmu[1], col, row, r):
                        # if so, the node's weights get updated
                        self.network[col][row] = self.update_weights(col, row, bmu[0], bmu[1], timestep)

    def show_network(self):
        """ Graphical output of Kohonen network """
        plt.imshow(self.network)
        plt.show()
    
    def show_data(self):
        """ Graphical output of training data """
        plt.imshow(self.data)
        plt.show()
    
    def save(self, save_fp):
        """ Utility function to save state data to directory `save_fp` """

        # Ensures directory exists and creates it if it doesn't
        Path(save_fp).mkdir(parents=True, exist_ok=True)

        with open(f"{save_fp}/network", 'w+') as f:
            f.write(str(self.network))
        
        with open(f"{save_fp}/data", 'w+') as f:
            f.write(str(self.data))

        matplotlib.image.imsave(f"{save_fp}/network.png", self.network)
        matplotlib.image.imsave(f"{save_fp}/data.png", self.data)


if __name__ == "__main__":
    # Parse from config file -- for CI/CD and production deployment
    # Or from command line -- for running experiments while developing
    HEIGHT = 100
    WIDTH = 100
    MAX_ITERATIONS = 1000
    INITIAL_LEARNING_RATE = 0.1
    WORKERS = 4

    # We should be able to import a partially trained network and training data on disk
    DATA = None
    network = None
    k = Kohonen(HEIGHT, WIDTH, MAX_ITERATIONS, INITIAL_LEARNING_RATE, WORKERS)
    k.train()
    k.show_network()
    k.save('res')


