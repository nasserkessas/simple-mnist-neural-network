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
            
            # Calculate z = wa + b, a' = σ(z) #
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a, z
    

    def SGD(self, training_data, epochs, mini_batch_size, eta):

        for epoch in range(epochs):

            # Shuffle data #
            shuffle(training_data)

            # Construct mini-batches array using mini_batch_size #
            mini_batches = [training_data[x:x+mini_batch_size] for x in range(len(training_data)/mini_batch_size)]

            # Update mini batches #
            for this_batch in mini_batches:
                self.update_mini_batch(this_batch, eta)

    def update_mini_batch(self, this_batch, eta):

        # Initialise ∇b and ∇w with 0's #
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Loop though each training sample in this_batch #
        for x, y in this_batch:

            # Compute gradient of Cost function #
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            # Update ∇b and ∇w sum # 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases using equation v'=v−η∇C #
        self.weights = [this_weight-(eta/len(this_batch)*this_nabla_w) for this_weight, this_nabla_w in zip(self.weights, nabla_w)]
        self.biases = [this_bias-(eta/len(this_batch)*this_nabla_b) for this_bias, this_nabla_b in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        previous_activation, activation_vectors, z_vectors = self.feedForward(x)

        ###########################################################
        # Need to review this section (don't properly understand) #
        delta = self.cost_prime(previous_activation, y) * self.sigmoid_prime(z_vectors[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.array(activation_vectors[-2]).transpose())

        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.array(activation_vectors[-l-1]).transpose())

        ###########################################################
        
        return (nabla_b, nabla_w)
    
    def feedForward(self, input):
        activation_vectors = [input]
        z_vectors = []

        previous_activation = input
        for layer in self.layers:
            previous_activation, previous_z = self.calc_layer_activations(previous_activation)
            activation_vectors.append(previous_activation)
            z_vectors.append(previous_z)
        
        return previous_activation, activation_vectors, z_vectors


    @staticmethod
    def sigmoid(z):
        return 1 / 1 + np.exp(-z)

    @staticmethod
    def cost_prime(a, y):
        return a-y
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
