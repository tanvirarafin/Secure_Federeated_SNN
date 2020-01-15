from __future__ import print_function
import torch
import os
from SNN import SNNetwork
from utils.training_utils import train, get_acc_and_loss
import time
import numpy as np
import tables
import argparse
import utils
import pickle

''''
Code snippet to train an SNN.
'''


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--dataset')
    parser.add_argument('--num_ite', default=10, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--epochs_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--deltas', default=1, type=int)
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=1, type=float, help='KL regularization coefficient')
    parser.add_argument('--r', default=0.3, help='Desired spiking rate of hidden neurons')
    parser.add_argument('--topology_type', default='fully_connected', choices=['fully_connected', 'feedforward', 'sparse'], help='Desired spiking rate of hidden neurons')


    args = parser.parse_args()


local_data_path = r'path/to/datasets'
save_path = os.getcwd() + r'/results'

datasets = {'mnist_dvs_2': r'mnist_dvs_25ms_26pxl_2_digits.hdf5',
            'mnist_dvs_10': r'mnist_dvs_25ms_26pxl_10_digits.hdf5',
            }


dataset = local_data_path + datasets[args.dataset]


input_train = torch.FloatTensor(tables.open_file(dataset).root.train.data[:])
output_train = torch.FloatTensor(tables.open_file(dataset).root.train.label[:])

input_test = torch.FloatTensor(tables.open_file(dataset).root.test.data[:])
output_test = torch.FloatTensor(tables.open_file(dataset).root.test.label[:])


### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
n_hidden_neurons = 4

### Learning parameters
if args.epochs:
    epochs = args.epochs
else:
    epochs = input_train.shape[0]
if args.epochs_test:
    epochs_test = args.epochs_test
else:
    epochs_test = input_test.shape[0]

test_accs = []

learning_rate = args.lr / n_hidden_neurons
kappa = args.kappa
alpha = args.alpha
deltas = args.deltas
num_ite = args.num_ite
r = args.r

### Randomly select training samples
indices = np.random.choice(np.arange(input_train.shape[0]), [epochs], replace=True)

S_prime = input_train.shape[-1]

S = epochs * S_prime
for _ in range(num_ite):
    ### Run training
    # Train it
    t0 = time.time()

    # Create the network
    network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected'))

    # Train it
    train(network, input_train, output_train, indices, learning_rate, kappa, deltas, alpha, r)
    print('Number of samples trained on: %d, time: %f' % (epochs, time.time() - t0))


    ### Test accuracy
    test_indices = np.random.choice(np.arange(input_test.shape[0]), [epochs_test], replace=False)
    np.random.shuffle(test_indices)

    acc, loss = get_acc_and_loss(network, input_test[test_indices], output_test[test_indices])

    test_accs.append(acc)
    print('Final test accuracy: %f' % acc)

np.save(save_path + '/acc_' + args.dataset + '_fully_connected' + '.npy', test_accs)

