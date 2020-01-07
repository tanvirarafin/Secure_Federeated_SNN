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


### Load Data DVS

# dataset = r'C:/Users/K1804053/PycharmProjects/datasets/shd/shd_10ms_10_digits_eng.hdf5'
# dataset = r'/users/k1804053/FL-SNN/shd_10ms_10_digits_eng.hdf5'
dataset = r'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs/mnist_dvs_25ms_26pxl_2_digits_polarity.hdf5'

# The dataset is of size n_samples * n_visible_neurons * S'
input_train = torch.FloatTensor(tables.open_file(dataset).root.train.data[:])
output_train = torch.FloatTensor(tables.open_file(dataset).root.train.label[:])

digits = [1, 7]
# labels_train = np.argmax(np.sum(tables.open_file(dataset).root.train.label[:], axis=-1), axis=-1)
# train_indices = np.hstack([np.where(labels_train == i)[0] for i in digits])
#
# training_data = torch.cat((input_train[train_indices], output_train[train_indices]), dim=1)
training_data = torch.cat((input_train, output_train), dim=1)

input_test = torch.FloatTensor(tables.open_file(dataset).root.test.data[:])
output_test = torch.FloatTensor(tables.open_file(dataset).root.test.label[:])

# labels_test = np.argmax(np.sum(tables.open_file(dataset).root.test.label[:], axis=-1), axis=-1)
# test_indices = np.hstack([np.where(labels_test == i)[0] for i in digits])
test_indices = np.random.choice(np.arange(len(training_data)), [200], replace=False)

### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
n_hidden_neurons = 4

num_basis_feedforward = 8
num_basis_feedback = 1
feedforward_filter = filters.raised_cosine_pillow_08
feedback_filter = filters.raised_cosine_pillow_08

### Learning parameters
learning_rate = 0.05
# epochs = len(labels_train)
epochs = 200

kappa = 0.2  # learning signal and eligibility trace averaging factor
deltas = 1  # local updates period
r = 0.5  # Desired hidden neurons spiking rate
alpha = 1  # learning signal regularization coefficient
mu = 1.5  # compression factor for the raised cosine basis
num_ite = 1 # number of iterations

# test_acc = [[] for _ in range(len(n_hidden_neurons_))]

# for i, n_hidden_neurons in enumerate(n_hidden_neurons_):
#     print('Nh: %d' % n_hidden_neurons)

topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_hidden_neurons +
                        [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)

topology[[i for i in range(n_hidden_neurons + n_output_neurons)], [i + n_input_neurons for i in range(n_hidden_neurons + n_output_neurons)]] = 0


    # for _ in range(num_ite):
### Run training
indices = np.random.choice(np.arange(len(training_data)), [epochs], replace=False)

training_sequence = training_data[indices, :, :]
S_prime = training_sequence.shape[-1]
S = epochs * S_prime


# Create the network
network = SNNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, topology,
                    n_basis_feedforward=num_basis_feedforward, feedforward_filter=feedforward_filter,
                    n_basis_feedback=num_basis_feedback, feedback_filter=feedback_filter,
                    tau_ff=10, tau_fb=10, mu=mu, weights_magnitude=0.01)

# Train it
t0 = time.time()
train(network, training_sequence, learning_rate, kappa, deltas, r, alpha)
t1 = time.time()
print('Number of samples trained on: %d, time: %f' % (epochs, t1 - t0))


### Test accuracy
acc, loss = get_acc_and_loss(network, input_test[test_indices], output_test[test_indices])
# test_acc[i].append(acc)
print('Final test accuracy: %f, test time: %f' % (acc, time.time() - t1))

# np.save(r'/users/k1804053/FL-SNN/results/acc_shd_binary_10_digits_2.npy', test_acc)
