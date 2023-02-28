import numpy as np

class Neural_Network (object):

    def __init__ (self, neuron_structure):
        self.layers = len(neuron_structure)
        self.neuron_structure = neuron_structure
        self.biases = [np.random.randn(b, 1) for b in neuron_structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(neuron_structure[:-1], neuron_structure[1:])]
