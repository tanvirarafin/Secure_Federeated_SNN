import torch
import tables
import utils.filters as filters
import numpy as np


def make_topology(topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, density=1):
    if topology_type == 'feedforward':
        topology = torch.zeros([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[:, :n_input_neurons] = 1
        topology[n_hidden_neurons:, n_input_neurons:(n_input_neurons + n_hidden_neurons)] = 1
    elif topology_type == 'fully_connected':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)
    elif topology_type == 'sparse':
        indices = np.random.choice(n_hidden_neurons * n_hidden_neurons, [int(density * n_hidden_neurons**2)], replace=False)

        row = np.array([int(index / n_hidden_neurons) for index in indices])
        col = np.array([int(index % n_hidden_neurons) for index in indices]) + n_input_neurons

        topology = torch.zeros([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[[r for r in row], [c for c in col]] = 1
        topology[:, :n_input_neurons] = 1
        topology[-n_output_neurons:, :] = 1

    if topology_type != 'feedforward':
        topology[:n_hidden_neurons, -n_output_neurons:] = 1
        topology[-n_output_neurons:, -n_output_neurons:] = 1

    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    assert torch.sum(topology[:, :n_input_neurons]) == (n_input_neurons * (n_hidden_neurons + n_output_neurons))

    return topology


def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type, density=1, mode='train', weights_magnitude=0.05,
                            n_basis_ff=8, ff_filter=filters.raised_cosine_pillow_08, n_basis_fb=1, fb_filter=filters.raised_cosine_pillow_08,
                            tau_ff=10, tau_fb=10, mu=1.5, task='supervised'):

    topology = make_topology(topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, density)
    print(topology[:, n_input_neurons:])
    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'n_basis_feedforward': n_basis_ff,
                          'feedforward_filter': ff_filter,
                          'n_basis_feedback': n_basis_fb,
                          'feedback_filter': fb_filter,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'mu': mu,
                          'weights_magnitude': weights_magnitude,
                          'task': task,
                          'mode': mode
                          }

    return network_parameters


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons)], dtype=torch.float))


def get_acc_and_loss(network, input_sequence, output_sequence):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]

    S = S_prime * epochs
    outputs = torch.zeros([epochs, network.n_output_neurons, S_prime])
    loss = 0

    rec = torch.zeros([network.n_learnable_neurons, S_prime])

    for s in range(S):
        if s % S_prime == 0:
            refractory_period(network)

        log_proba = network(input_sequence[int(s / S_prime), :, s % S_prime])
        loss += torch.sum(log_proba).numpy()
        outputs[int(s / S_prime), :, s % S_prime] = network.spiking_history[network.output_neurons, -1]
        rec[:, s % S_prime] = network.spiking_history[network.learnable_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(output_sequence, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
    return acc, loss


def feedforward_sampling(network, example, et, ls, s, S_prime, alpha, r):
    """"
    Runs a feedforward sampling pass:
    - computes log probabilities
    - accumulates learning signal
    - accumulates eligibility trace
    """
    # Run forward pass
    log_proba = network(example[:, s % S_prime])

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_non_learnable_neurons])
    ls += torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]) / network.n_output_neurons \
          - alpha*torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-12 + proba_hidden / r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - r))) / network.n_hidden_neurons

    # Accumulate eligibility trace
    for parameter in et:
        et[parameter] += network.gradients[parameter]

    return log_proba, ls, et


def local_feedback_and_update(network, eligibility_trace, learning_signal, et_temp, ls_temp, learning_rate, kappa, s, deltas):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """
    # At local algorithmic timesteps, do a local update
    if (s + 1) % deltas == 0:
        # local feedback
        learning_signal = kappa * learning_signal + (1 - kappa) * ls_temp
        ls_temp = 0

        # Update parameter
        for parameter in eligibility_trace:
            eligibility_trace[parameter] = kappa * eligibility_trace[parameter] + (1 - kappa) * et_temp[parameter]

            et_temp[parameter] = 0

            network.get_parameters()[parameter][network.output_neurons - network.n_non_learnable_neurons] += \
                learning_rate * eligibility_trace[parameter][network.output_neurons - network.n_non_learnable_neurons]

            network.get_parameters()[parameter][network.hidden_neurons - network.n_non_learnable_neurons] \
                += learning_rate * learning_signal * eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons]

    return learning_signal, ls_temp, eligibility_trace, et_temp



def train(network,  input_train, output_train, indices, learning_rate, kappa, deltas, alpha, r):
    """"
    Train a network on the sequence passed as argument.
    """

    network.set_mode('train')

    eligibility_trace = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}
    et_temp = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}

    learning_signal = 0
    ls_temp = 0

    training_sequence = torch.cat((input_train, output_train), dim=1)

    S_prime = training_sequence.shape[-1]
    S = len(indices) * S_prime

    # Run the first Deltas feedforward sampling steps to initialize the learning signal and eligibility trace
    for s in range(deltas):
        # Feedforward sampling step
        log_proba, learning_signal, eligibility_trace = feedforward_sampling(network, training_sequence[indices[0]], eligibility_trace, learning_signal, s, S_prime, alpha, r)

    # First update
    for parameter in eligibility_trace:
        network.get_parameters()[parameter][network.output_neurons - network.n_non_learnable_neurons] += \
            learning_rate * eligibility_trace[parameter][network.output_neurons - network.n_non_learnable_neurons]

        network.get_parameters()[parameter][network.hidden_neurons - network.n_non_learnable_neurons] \
            += learning_rate * learning_signal * eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons]

    for s in range(deltas, S):
        # Reset network for each example
        if s % S_prime == 0:
            refractory_period(network)

        # Feedforward sampling step
        log_proba, ls_temp, et_temp = feedforward_sampling(network, training_sequence[indices[int(s / S_prime)]], et_temp, ls_temp, s, S_prime, alpha, r)

        # Local feedback and update
        learning_signal, ls_temp, eligibility_trace, et_temp \
            = local_feedback_and_update(network, eligibility_trace, learning_signal, et_temp, ls_temp, learning_rate, kappa, s, deltas)

        if s % int(S / 5) == 0:
            print('Step %d out of %d' % (s, S))
