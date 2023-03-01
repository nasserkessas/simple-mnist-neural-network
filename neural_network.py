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

    @staticmethod
    def sigmoid(z):
        return 1/1+np.exp(-z)