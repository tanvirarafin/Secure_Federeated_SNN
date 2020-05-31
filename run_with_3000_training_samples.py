from __future__ import print_function
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from SNN import SNNetwork
import time
import numpy as np
import tables
import utils
import pickle
import utils.filters as filters
from utils.training_utils import train, get_acc_and_loss
def run(rank, size):
 
    train_indices = torch.zeros([3,3000],dtype=torch.long)
    test_indices = torch.zeros([3,333],dtype=torch.long)

    local_data_path = '/home/cream/Desktop/arafin_experiments/SOCC/FL-SNN/data/'
    save_path = os.getcwd() + r'/results'

    datasets = {'mnist_dvs_10': r'mnist_dvs_25ms_26pxl_10_digits.hdf5'}
    dataset = local_data_path + datasets['mnist_dvs_10']

    input_train = torch.FloatTensor(tables.open_file(dataset).root.train.data[:])
    output_train = torch.FloatTensor(tables.open_file(dataset).root.train.label[:])

    input_test = torch.FloatTensor(tables.open_file(dataset).root.test.data[:])
    output_test = torch.FloatTensor(tables.open_file(dataset).root.test.label[:])
    ### Network parameters
    n_input_neurons = input_train.shape[1]
    n_output_neurons = output_train.shape[1]
    n_hidden_neurons = 16
    epochs = input_train.shape[0]
    epochs_test = input_test.shape[0]



    learning_rate = 0.005 / n_hidden_neurons
    kappa = 0.2
    alpha = 1
    deltas = 1
    num_ite = 1
    r = 0.3
    weights_magnitude=0.05
    task='supervised'
    mode='train', 
    tau_ff=10
    tau_fb=10
    mu=1.5, 
    n_basis_feedforward = 8
    feedforward_filter = filters.raised_cosine_pillow_08
    feedback_filter = filters.raised_cosine_pillow_08
    n_basis_feedback = 1
    topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0
    assert torch.sum(topology[:, :n_input_neurons]) == (n_input_neurons * (n_hidden_neurons + n_output_neurons))
    print(topology[:, n_input_neurons:])
    if rank  == 0:
        train_indicess=torch.tensor(np.random.choice(np.arange(input_train.shape[0]),[3,3000],replace=False),dtype=torch.long)
        test_indicess=torch.tensor(np.random.choice(np.arange(input_test.shape[0]),[3,333],replace=False),dtype=torch.long)
        dist.send(tensor=train_indicess,dst=1)
        dist.send(tensor=train_indicess,dst=2)
        dist.send(tensor=train_indicess,dst=3)
    else:
        dist.recv(tensor=train_indices,src=0)
    dist.barrier()
    
    if rank  == 0:
        dist.send(tensor=test_indicess,dst=1)
        dist.send(tensor=test_indicess,dst=2)
        dist.send(tensor=test_indicess,dst=3)
    else:
        dist.recv(tensor=test_indices,src=0)
    dist.barrier()
    if rank != 0:
        training_data = input_train[train_indices[rank-1,:]]
        training_label =output_train[train_indices[rank-1,:]]
        test_data = input_test[test_indices[rank-1,:]]
        test_label= output_test[test_indices[rank-1,:]]
        indices = np.random.choice(np.arange(training_data.shape[0]), [training_data.shape[0]], replace=True)
        S_prime = training_data.shape[-1]
        S = epochs * S_prime
        num_ite =1
        test_accs =[]
        for _ in range(num_ite):
            ### Run training
            # Train it
            t0 = time.time()
            # Create the network
            network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected'))

            # Train it
            train(network, training_data, training_label, indices, learning_rate, kappa, deltas, alpha, r)
            print('Number of samples trained on: %d, time: %f' % (epochs, time.time() - t0))
            ### Test accuracy
            test_indx = np.random.choice(np.arange(test_data.shape[0]), [test_data.shape[0]]
                                         , replace=False)
            np.random.shuffle(test_indx)

            acc, loss = get_acc_and_loss(network, test_data[test_indx], test_label[test_indx])

            test_accs.append(acc)
            print('Final test accuracy: %f' % acc)



def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] =  '29520'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)

if __name__=="__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

