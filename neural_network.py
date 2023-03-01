from random import shuffle

import numpy as np

class Neural_Network (object):

    def __init__ (self, neuron_structure):
        self.layers = len(neuron_structure)
        self.neuron_structure = neuron_structure
        self.biases = [np.random.randn(b, 1) for b in neuron_structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(neuron_structure[:-1], neuron_structure[1:])]

    def calc_network_activations(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a), b)
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size, eta):

        for epoch in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[x:x+mini_batch_size] for x in range(len(training_data)/mini_batch_size)]
            for this_batch in mini_batches:
                self.update_mini_batch(this_batch)

    def update_mini_batch(self, this_batch):
        pass

    @staticmethod
    def sigmoid(z):
        return 1/1+np.exp(-z)