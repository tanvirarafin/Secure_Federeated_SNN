from __future__ import print_function
import datetime
import torch
import numpy as np
import os
import utils.filters as filters
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.distributed_utils import global_update_subset, feedforward_sampling_accum_gradients, init_training, init_processes
from utils.training_utils import local_feedback_and_update, get_acc_and_loss


""""

Runs FL-SNN using two devices for a fixed communication rate and with different values of tau as described in the paper. 

"""


# Distributed Synchronous SGD
def train(rank, num_nodes, net_params, train_params):
    # Setup training parameters
    rate = train_params['rate']
    eta = train_params['eta']
    kappa = train_params['kappa']
    deltas = train_params['deltas']
    epochs = train_params['epochs']
    epochs_test = train_params['epochs_test']
    dataset = train_params['dataset']
    learning_rate = train_params['learning_rate']
    alpha = train_params['alpha']
    r = train_params['r']
    num_ite = train_params['num_ite']
    save_path = net_params['save_path']

    # Create network groups for communication
    all_nodes = dist.new_group([0, 1, 2], timeout=datetime.timedelta(0, 360000))
    print("Training",flush=True)
    tau_list = [int((2**i)/rate) for i in range(4)]  # global update periods used for training
    test_accuracies = [[] for _ in range(len(tau_list))]  # used to store test accuracies
    test_indices = np.hstack((np.arange(900, 1000)[:epochs_test], np.arange(1900, 2000)[:epochs_test]))

    for i, tau in enumerate(tau_list):
        n_weights_to_send = int(tau * rate)

        for _ in range(num_ite):
            # Initialize main parameters for training
            network, local_training_sequence, weights_list, S_prime, S, eligibility_trace, et_temp, learning_signal, ls_temp \
                = init_training(rank, num_nodes, all_nodes, dataset, eta, epochs, net_params)

            # Gradients accumulator
            gradients_accum = torch.zeros(network.feedforward_weights.shape, dtype=torch.float)
            dist.barrier(all_nodes)

            ### First local step
            for s in range(deltas):
                if rank != 0:
                    # Feedforward sampling step
                    log_proba, gradients_accum, learning_signal, eligibility_trace \
                        = feedforward_sampling_accum_gradients(network, local_training_sequence, eligibility_trace, learning_signal, gradients_accum, s, S_prime, alpha, r)

            if rank != 0:
                print('rank 0')
                # First local update
                for parameter in eligibility_trace:
                    eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= learning_signal
                    network.get_parameters()[parameter] += eligibility_trace[parameter] * learning_rate

            if (s + 1) % (tau*deltas) == 0:  # First global update
                dist.barrier(all_nodes)
                global_update_subset(all_nodes, rank, network, weights_list, gradients_accum, n_weights_to_send)
                gradients_accum = torch.zeros(network.feedforward_weights.shape, dtype=torch.float)

                dist.barrier(all_nodes)
                print('Hi')


            ### Remainder of the steps
            for s in range(deltas, S):
                if rank != 0:
                    if s % S_prime == 0:  # Reset internal state for each example
                        network.reset_internal_state()

                    # lr decay
                    if (s % S/5 == 0) & (learning_rate > 0.005):
                        learning_rate /= 2

                    # Feedforward sampling
                    log_proba, gradients_accum, ls_temp, et_temp \
                        = feedforward_sampling_accum_gradients(network, local_training_sequence, et_temp, ls_temp, gradients_accum, s, S_prime, alpha, r)

                    # Local feedback and global update
                    learning_signal, ls_temp, eligibility_trace, et_temp \
                        = local_feedback_and_update(network, eligibility_trace, learning_signal, et_temp, ls_temp, learning_rate, kappa, s, deltas)

                if (s + 1) % (tau*deltas) == 0:
                    dist.barrier(all_nodes)
                    # Global update
                    global_update_subset(all_nodes, rank, network, weights_list, gradients_accum, n_weights_to_send)
                    gradients_accum = torch.zeros(network.feedforward_weights.shape, dtype=torch.float)

                    dist.barrier(all_nodes)

            if rank == 0:
                global_acc, test_loss = get_acc_and_loss(network, dataset, test_indices)
                test_accuracies[i].append(global_acc)
                print('Tau: %d, final accuracy: %f' % (tau, global_acc))

    if rank == 0:
        if save_path is None:
            save_path = os.getcwd()
        np.save(save_path + r'/test_accuracies_r_%f.npy' % rate, arr=np.array(test_accuracies))
        print('Training finished and accuracies saved to ' + save_path + r'/test_accuracies_r_%f.npy' % rate)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Train probabilistic SNNs in a distributed fashion using Pytorch')
    # Mandatory arguments
    parser.add_argument('--dist_url', type=str, help='URL to specify the initialization method of the process group')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes to run')
    parser.add_argument('--processes_per_node', default=1, type=int, help='Number of processes in the node')
    parser.add_argument('--dataset', help='Path to the dataset')

    # Pytorch arguments
    parser.add_argument('--backend', default='gloo', choices=['gloo', 'nccl', 'mpi', 'tcp'], help='Communication backend to use')

    # Training arguments
    parser.add_argument('--num_ite', default=10, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=200, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--epochs_test', default=200, type=int, help='Number of samples to test on')
    parser.add_argument('--save_path', default=None)

    # SNN arguments
    parser.add_argument('--n_hidden_neurons', default=0, type=int)
    parser.add_argument('--n_basis_feedforward', default=8, type=int)
    parser.add_argument('--n_basis_feedback', default=8, type=int)
    parser.add_argument('--tau_ff', default=10, type=int, help='Feedforward filter length')
    parser.add_argument('--tau_fb', default=10, type=int, help='Feedback filter length')
    parser.add_argument('--ff_filter', default='raised_cosine_pillow_08', help='Feedforward filter type')
    parser.add_argument('--fb_filter', default='raised_cosine_pillow_08', help='Feedback filter type')
    parser.add_argument('--mu', default=1.5, type=float, help='Filters width')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--eta', default=0.05, type=float, help='Defines the balance of classes per node')
    parser.add_argument('--kappa', default=0.05, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--deltas', default=1, type=int, help='Local update period')
    parser.add_argument('--alpha', default=1, type=float, help='KL regularization strength')
    parser.add_argument('--rate', default=0.25, type=float, help='Fixed communication rate')
    parser.add_argument('--r', default=0.3, type=float, help='Desired hidden neurons spiking rate')
    parser.add_argument('--weights_magnitude', default=0.01, type=float)
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    print(args)

    node_rank = args.local_rank + args.local_rank*(args.processes_per_node - 1)
    n_processes = args.processes_per_node
    assert (args.world_size % n_processes == 0), 'Each node must have the same number of processes'
    assert (node_rank + n_processes) <= args.world_size, 'There are more processes specified than world_size'

    n_input_neurons = 26**2
    n_output_neurons = 2
    n_hidden_neurons = args.n_hidden_neurons

    topology = torch.tensor([[1] * n_input_neurons + [0] * n_hidden_neurons + [1] * n_output_neurons] * n_hidden_neurons +
                            [[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_output_neurons)
    topology[-2, -1] = 1
    topology[-1, -2] = 1

    filters_dict = {'base_ff_filter': filters.base_feedforward_filter, 'base_fb_filter': filters.base_feedback_filter, 'cosine_basis': filters.cosine_basis,
                    'raised_cosine': filters.raised_cosine, 'raised_cosine_pillow_05': filters.raised_cosine_pillow_05, 'raised_cosine_pillow_08': filters.raised_cosine_pillow_08}

    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'n_output_neurons': n_output_neurons,
                          'topology': topology,
                          'n_basis_feedforward': args.n_basis_feedforward,
                          'feedforward_filter': filters_dict[args.ff_filter],
                          'n_basis_feedback': args.n_basis_feedback,
                          'feedback_filter': filters_dict[args.fb_filter],
                          'tau_ff': args.tau_ff,
                          'tau_fb': args.tau_fb,
                          'mu': args.mu,
                          'weights_magnitude': args.weights_magnitude,
                          'save_path': args.save_path
                          }

    training_parameters = {'dataset': args.dataset,
                           'learning_rate': args.lr,
                           'epochs': args.epochs,
                           'epochs_test': args.epochs_test,
                           'eta': args.eta,
                           'kappa': args.kappa,
                           'deltas': args.deltas,
                           'alpha': args.alpha,
                           'r': args.r,
                           'num_ite': args.num_ite,
                           'rate': args.rate
                           }

    processes = []
    for local_rank in range(n_processes):
        p = mp.Process(target=init_processes, args=(node_rank + local_rank, args.world_size, args.backend, args.dist_url, network_parameters, training_parameters, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
