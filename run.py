from __future__ import print_function
import os
import torch
import datetime
import torch.distributed as dist
from torch.multiprocessing import Process
from SNN import SNNetwork
import time
import numpy as np
import tables
import utils
import pickle
import utils.filters as filters
from utils.distributed_utils import  global_update
from utils.training_utils import local_feedback_and_update, feedforward_sampling, get_acc_and_loss, refractory_period

      
def run(rank, size):
    local_train_length = 3000
    local_test_length = 333
    train_indices = torch.zeros([3,local_train_length],dtype=torch.long)
    test_indices = torch.zeros([3, local_test_length],dtype=torch.long)

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
    n_hidden_neurons =4
    epochs = local_train_length
    epochs_test = local_test_length

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
    tau=10
    mu=1.5, 
    n_basis_feedforward = 8
    feedforward_filter = filters.raised_cosine_pillow_08
    feedback_filter = filters.raised_cosine_pillow_08
    n_basis_feedback = 1
    topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0
    assert torch.sum(topology[:, :n_input_neurons]) == (n_input_neurons * (n_hidden_neurons + n_output_neurons))
    print(topology[:, n_input_neurons:])
    # Create the network
    network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected'))
    
    # At the beginning, the master node:
    # - transmits its weights to the workers
    # - distributes the samples among workers
    if rank == 0:
        # Initializing an aggregation list for future weights collection
        weights_list = [[torch.zeros(network.feedforward_weights.shape, dtype=torch.float) for _ in range(size)],
                        [torch.zeros(network.feedback_weights.shape, dtype=torch.float) for _ in range(size)],
                        [torch.zeros(network.bias.shape, dtype=torch.float) for _ in range(size)],
                        [torch.zeros(1, dtype=torch.float) for _ in range(size)]]
    else:
        weights_list = []
        
    if rank  == 0:
        train_indicess=torch.tensor(np.random.choice(np.arange(input_train.shape[0]),[3,local_train_length],replace=False),dtype=torch.long)
        test_indicess=torch.tensor(np.random.choice(np.arange(input_test.shape[0]),[3,local_test_length],replace=False),dtype=torch.long)
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
        print("S is", S)
    dist.barrier()

    group = dist.group.WORLD
    # Master node sends its weights
    for parameter in network.get_parameters():
        dist.broadcast(network.get_parameters()[parameter], 0)
    if rank == 0:
        print('Node 0 has shared its model and training data is partitioned among workers')  
    # The nodes initialize their eligibility trace and learning signal
    eligibility_trace = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}
    et_temp = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}

    learning_signal = 0
    ls_temp = 0
    dist.barrier()
    num_ite =1
  
    test_accs =[]
    if rank != 0:
        test_indx = np.random.choice(np.arange(test_data.shape[0]), [test_data.shape[0]]
                                         , replace=False)
        np.random.shuffle(test_indx)

        _, loss = get_acc_and_loss(network, test_data[test_indx], test_label[test_indx])

        network.set_mode('train')
        local_training_sequence = torch.cat((training_data, training_label), dim=1)
    dist.barrier()
    ### First local step
    for i in range(num_ite):
        for s in range(deltas):
            if rank != 0:
                # Feedforward sampling step
                log_proba, learning_signal, eligibility_trace \
                    = feedforward_sampling(network, local_training_sequence[indices[0]], eligibility_trace, learning_signal, s, S_prime, alpha, r)
               
        if rank != 0:
            # First local update
            for parameter in eligibility_trace:
                eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= learning_signal
                network.get_parameters()[parameter] += eligibility_trace[parameter] * learning_rate

        # First global update
        if (s + 1) % (tau * deltas) == 0:
            dist.barrier()
            global_update(group, rank, network, weights_list)
            dist.barrier()

        S=input_train.shape[-1]*local_train_length
        ### Remainder of the steps
        for s in range(deltas, S):
            print(s)
            if rank != 0:
                if s % S_prime == 0:  # Reset internal state for each example
                    network.reset_internal_state()

                # lr decay
                if (s % S/5 == 0) & (learning_rate > 0.005):
                    learning_rate /= 2

                # Feedforward sampling
                log_proba, ls_temp, et_temp \
                    = feedforward_sampling(network, local_training_sequence[indices[0]], et_temp, ls_temp, s, S_prime, alpha, r)

                # Local feedback and global update
                learning_signal, ls_temp, eligibility_trace, et_temp \
                    = local_feedback_and_update(network, eligibility_trace, learning_signal, et_temp, ls_temp, learning_rate, kappa, s, deltas)

                ## Every few timesteps, record test losses
                if (s + 1) % 40 == 0:
                    _, loss = get_acc_and_loss(network, test_data[test_indx], test_label[test_indx])
              
                    network.set_mode('train')

            # Global update
            if (s + 1) % (tau*deltas) == 0:
                dist.barrier()
                global_update(group, rank, network, weights_list)
                dist.barrier()

        if rank == 0:
            global_test_indices = np.random.choice(np.arange(input_test.shape[0]), [epochs_test], replace=False)
            np.random.shuffle(global_test_indices)
            print (global_test_indices)
            global_acc,_ = get_acc_and_loss(network, input_test[global_test_indices], output_test[global_test_indices])   
            print('Final global test accuracy: %f' % global_acc)
           


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

