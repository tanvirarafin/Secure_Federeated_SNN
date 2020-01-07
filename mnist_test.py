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

dataset = 'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs/mnist_dvs_25ms_26pxl_2_digits.hdf5'

# The dataset is of size n_samples * n_visible_neurons * S'
# The first 1000 samples correspond to label '1', the next 1000 correspond to '7'
input_train = torch.FloatTensor(np.vstack((tables.open_file(dataset).root.data[:900], tables.open_file(dataset).root.data[1000:1900])))
output_train = torch.FloatTensor(np.vstack((tables.open_file(dataset).root.label[:900], tables.open_file(dataset).root.label[1000:1900])))

training_data = torch.cat((input_train, output_train), dim=1)


### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
n_hidden_neurons = 64


### Learning parameters
learning_rate = 0.005 / max(1, n_hidden_neurons)
epochs = 200
eta = 0.5  # balancedness of the dataset
epochs_test = 200
kappa = 0.2  # learning signal and eligibility trace averaging factor
deltas = 1  # local updates period
r = 0.3  # Desired hidden neurons spiking rate
alpha = 1  # learning signal regularization coefficient
mu = 1.5  # compression factor for the raised cosine basis
num_ite = 5  # number of iterations


### Dataset parameters
n_samples_train_per_class = 900
indices_0 = np.arange(0, 900)
indices_1 = np.arange(900, 1800)
n_main_class = math.floor(epochs * eta)
n_secondary_class = epochs - n_main_class

### Randomly select training samples
num_ite = 20

# test_accs = [[] for _ in range(len(n_hidden_neurons_))]

### Run training
# for i, n_hidden_neurons in enumerate(n_hidden_neurons_):
#     print("Nh: %d" % n_hidden_neurons)
n_neurons = n_input_neurons + n_output_neurons + n_hidden_neurons

num_basis_feedforward = 8
num_basis_feedback = 1
feedforward_filter = filters.raised_cosine_pillow_08
feedback_filter = filters.raised_cosine_pillow_08

topology = torch.tensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_hidden_neurons +
                        [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)

topology[[i for i in range(n_hidden_neurons + n_output_neurons)], [i + n_input_neurons for i in range(n_hidden_neurons + n_output_neurons)]] = 0

    # for _ in range(num_ite):
# indices = np.hstack((np.random.choice(indices_0, [n_main_class], replace=False), np.random.choice(indices_1, [n_secondary_class], replace=False)))
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
train(network, training_sequence, learning_rate, kappa, deltas, alpha, r)
print('Number of samples trained on: %d, time: %f' % (epochs, time.time() - t0))


### Test accuracy
# The last 100 samples of each class are kept for test
test_indices = np.hstack((np.arange(900, 1000)[:epochs_test], np.arange(1900, 2000)[:epochs_test]))
np.random.shuffle(test_indices)

acc, loss = get_acc_and_loss(network, dataset, test_indices)

# test_accs[i].append(acc)
print('Final test accuracy: %f' % acc)

# np.save(r'/users/k1804053/FL-SNN-distant/results/test_accs_binary.npy', test_accs)

