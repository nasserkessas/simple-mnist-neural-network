from random import shuffle

import numpy as np

class Neural_Network (object):

    def __init__ (self, neuron_structure):
        self.layers = len(neuron_structure)
        self.neuron_structure = neuron_structure
        self.biases = [np.random.randn(b, 1) for b in neuron_structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(neuron_structure[:-1], neuron_structure[1:])]

    def calc_layer_activations(self, a):
        
        # Calculate activations for the next layer based on weights and biases of previous #
        for b, w in zip(self.biases, self.weights):
            
            # Calculate a' = σ(wa + b) #
            a = self.sigmoid(np.dot(w,a) + b)
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size, eta):

        for epoch in range(epochs):

            # Shuffle data #
            shuffle(training_data)

            # Construct mini-batches array using mini_batch_size #
            mini_batches = [training_data[x:x+mini_batch_size] for x in range(len(training_data)/mini_batch_size)]

            # Update mini batches #
            for this_batch in mini_batches:
                self.update_mini_batch(this_batch)

    def update_mini_batch(self, this_batch):
        # Do gradient descent and backprop here #
        pass

    @staticmethod
    def sigmoid(z):
        return 1 / 1 + np.exp(-z)