from __future__ import print_function
import torch
import os
from SNN import SNNetwork
import utils.filters as filters
from utils.training_utils import train, get_acc_and_loss
import time
import numpy as np
import tables
import math

''''
Code snippet to train an SNN on the MNIST dataset.
'''

t0 = time.time()

### Load Data DVS

# dataset = r'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs/mnist_dvs_25ms_26pxl_10_digits.hdf5'
dataset = r'/users/k1804053/FL-SNN/mnist_dvs_25ms_26pxl_10_digits.hdf5'

# The dataset is of size n_samples * n_visible_neurons * S'
input_train = torch.FloatTensor(tables.open_file(dataset).root.train.data[:])
output_train = torch.FloatTensor(tables.open_file(dataset).root.train.label[:])
training_data = torch.cat((input_train, output_train), dim=1)

input_test = torch.FloatTensor(tables.open_file(dataset).root.test.data[:])
output_test = torch.FloatTensor(tables.open_file(dataset).root.test.label[:])
test_data = torch.cat((input_test, output_test), dim=1)


### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
n_hidden_neurons_ = [128, 256, 512]


### Learning parameters
learning_rate = 0.05
epochs = 9000
epochs_test = 1000
kappa = 0.2  # learning signal and eligibility trace averaging factor
deltas = 1  # local updates period
r = 0.5  # Desired hidden neurons spiking rate
alpha = 1  # learning signal regularization coefficient
mu = 1.5  # compression factor for the raised cosine basis
num_ite = 5  # number of iterations

### Randomly select training samples
num_ite = 10

test_accs = [[] for _ in range(len(n_hidden_neurons_))]

### Run training
for i, n_hidden_neurons in enumerate(n_hidden_neurons_):
    print("Nh: %d" % n_hidden_neurons)
    n_neurons = n_input_neurons + n_output_neurons + n_hidden_neurons

    num_basis_feedforward = 8
    num_basis_feedback = 1
    feedforward_filter = filters.raised_cosine_pillow_08
    feedback_filter = filters.raised_cosine_pillow_08

    topology = torch.tensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_hidden_neurons +
                            [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)

    topology[[i for i in range(n_hidden_neurons + n_output_neurons)], [i + n_input_neurons for i in range(n_hidden_neurons + n_output_neurons)]] = 0

    for _ in range(num_ite):
        indices = np.random.choice(np.arange(input_train.shape[0]), [epochs], replace=False)

        training_sequence = training_data[indices, :, :]
        S_prime = training_sequence.shape[-1]
        S = epochs * S_prime

        # Create the network
        network = SNNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, topology,
                            n_basis_feedforward=num_basis_feedforward, feedforward_filter=feedforward_filter,
                            n_basis_feedback=num_basis_feedback, feedback_filter=feedback_filter,
                            tau_ff=10, tau_fb=10, mu=mu, weights_magnitude=0.01)

        # Train it
        train(network, training_sequence, learning_rate, kappa, deltas, r, alpha)
        print('Number of samples trained on: %d, time: %f' % (epochs, time.time() - t0))


        ### Test accuracy
        # The last 100 samples of each class are kept for test
        test_indices = np.hstack([np.arange(100*i, 100*i + int(epochs_test/n_output_neurons)) for i in range(n_output_neurons)])
        acc, loss = get_acc_and_loss(network, input_test[test_indices], output_test[test_indices])

        test_accs[i].append(acc)
        print('Final test accuracy: %f' % acc)

np.save(r'/users/k1804053/FL-SNN/results/acc_mnist_dvs_binary_10_digits_2.npy', test_accs)
