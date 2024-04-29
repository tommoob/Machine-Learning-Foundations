import random
import numpy as np
from tqdm import tqdm
import pickle
import os
from datetime import datetime


class MNISTHiddenLayer():
    
    
    def initialize_network(self, n_inputs=784, n_hidden=100, n_outputs=10, weight_range=[-0.1, 0.1]):
        network = list()
        hidden_layer = [{'weights': [random.uniform(weight_range[0], weight_range[1]) for _ in range(n_inputs + 1)]}
                    for _ in range(n_hidden)]
        network.append(hidden_layer)

        # Initialize weights for the output layer
        output_layer = [{'weights': [random.uniform(weight_range[0], weight_range[1]) * np.sqrt(2 / (n_inputs + n_hidden)) for _ in range(n_hidden + 1)]}
                        for _ in range(n_outputs)]
        network.append(output_layer)
        return network
    
    
    def activate_neuron(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
            
        return activation
        

    def transfer(self, activation):
        return np.tanh(activation)

    # Define the derivative of the tanh activation function
    def transfer_derivative(self, output):
        return 1 - np.tanh(output) ** 2
    
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate_neuron(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
       
    
    def backward_propagate_error(self, network, expected):
            for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                    for j in range(len(layer)):
                        error = 0.0
                        for neuron in network[i + 1]:
                            error += (neuron['weights'][j] * neuron['delta'])
                        errors.append(error)
                else:
                    for j in range(len(layer)):
                        neuron = layer[j]
                        errors.append(neuron['output'] - expected[j])
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])


    def update_weights(self, network, row, l_rate, clip_threshold):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    gradient = l_rate * neuron['delta'] * inputs[j]
                    if gradient > clip_threshold:
                        gradient = clip_threshold
                    elif gradient < -clip_threshold:
                        gradient = -clip_threshold
                    neuron['weights'][j] -= gradient
                neuron['weights'][-1] -= l_rate * neuron['delta']


    def train_network(self, network, train_imgs, train_labs, l_rate, n_epoch, n_outputs, batch_size, clip_threshold, model_save_path):
        
        num_samples = len(train_imgs)
        num_batches = num_samples // batch_size
        
        current_time = datetime.now().strftime("%H:%M:%S")
        for epoch in tqdm(range(n_epoch), desc=f"The time is {current_time}"):

            sum_error = 0
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size

                batch_imgs = train_imgs[batch_start:batch_end]
                batch_labs = train_labs[batch_start:batch_end]

                batch_error = 0
                for i, row in enumerate(batch_imgs):
                    outputs = self.forward_propagate(network, row)
                    expected = [0 for i in range(n_outputs)]
                    expected[int(batch_labs[i])] = 1
                    batch_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                    self.backward_propagate_error(network, expected)
                    self.update_weights(network, row, l_rate, clip_threshold)

                sum_error += batch_error
            current_time = datetime.now().strftime("%H:%M:%S")
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            if epoch % 5==0:
                checkpoint_save_path = os.path.join(os.path.dirname(model_save_path), f"checkpoint_{epoch}_" + os.path.basename(model_save_path))
                self.save_network_weights(network, checkpoint_save_path)
                
    def predict(self, network, imgs):
        outputs = self.forward_propagate(network, imgs)
        return np.array(outputs)
    
    def save_network_weights(self, network, filename):
        with open(filename, 'wb') as file:
            pickle.dump(network, file)

    # Load the network weights
    def load_network_weights(self, filename):
        with open(filename, 'rb') as file:
            network = pickle.load(file)
        return network  
