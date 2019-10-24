import torch
import tables


def feedforward_sampling(network, training_sequence, et, ls, s, S_prime, alpha, r):
    """"
    Runs a feedforward sampling pass:
    - computes log probabilities
    - accumulates learning signal
    - accumulates eligibility trace
    """
    # Run forward pass
    log_proba = network(training_sequence[int(s / S_prime), :, s % S_prime])

    # Accumulate learning signal
    ls += torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]) / network.n_learnable_neurons \
          - alpha*torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-07 + torch.sigmoid(network.potential[network.hidden_neurons - network.n_non_learnable_neurons]) / r)
          + (1 - network.spiking_history[network.hidden_neurons, -1])
          * torch.log(1e-07 + (1. - torch.sigmoid(network.potential[network.hidden_neurons - network.n_non_learnable_neurons])) / (1 - r))) / network.n_learnable_neurons

    # Accumulate eligibility trace
    for parameter in et:
        et[parameter] += network.get_gradients()[parameter]

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
            eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= learning_signal
            network.get_parameters()[parameter] += eligibility_trace[parameter] * learning_rate

    return learning_signal, ls_temp, eligibility_trace, et_temp


def get_acc_and_loss(network, dataset, indices):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    input_sequence = torch.FloatTensor(tables.open_file(dataset).root.data[indices, :, :])
    output_sequence = torch.FloatTensor(tables.open_file(dataset).root.label[indices, :, :])

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]

    S = S_prime * epochs
    outputs = torch.zeros([epochs, network.n_output_neurons, S_prime])
    loss = 0

    for s in range(S):
        if s % S_prime == 0:
            network.reset_internal_state()

        log_proba = network(input_sequence[int(s / S_prime), :, s % S_prime])
        loss += torch.sum(log_proba).numpy()
        outputs[int(s / S_prime), :, s % S_prime] = network.spiking_history[network.output_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(output_sequence, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
    return acc, loss


def train(network, training_sequence, learning_rate, kappa, deltas, r, alpha):
    """"
    Train a network on the sequence passed as argument.
    """

    network.set_mode('train')

    eligibility_trace = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}
    et_temp = {'ff_weights': 0, 'fb_weights': 0, 'bias': 0}

    learning_signal = 0
    ls_temp = 0

    epochs = training_sequence.shape[0]
    S_prime = training_sequence.shape[-1]
    S = epochs * S_prime

    # Run the first Deltas feedforward sampling steps to initialize the learning signal and eligibility trace
    for s in range(deltas):
        # Feedforward sampling step
        log_proba, learning_signal, eligibility_trace = feedforward_sampling(network, training_sequence, eligibility_trace, learning_signal, s, S_prime, alpha, r)

    # First update
    for parameter in eligibility_trace:
        eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= learning_signal
        network.get_parameters()[parameter] += eligibility_trace[parameter] * learning_rate


    for s in range(deltas, S):
        # Reset network for each example
        if s % S_prime == 0:
            network.reset_internal_state()

        # Regularly decrease learning rate
        if s % (S / 5) == 0:
            learning_rate /= 2

        # Feedforward sampling step
        log_proba, ls_temp, et_temp = feedforward_sampling(network, training_sequence, et_temp, ls_temp, s, S_prime, alpha, r)

        # Local feedback and update
        learning_signal, ls_temp, eligibility_trace, et_temp \
            = local_feedback_and_update(network, eligibility_trace, learning_signal, et_temp, ls_temp, learning_rate, kappa, s, deltas)

        if s % int(S / 20) == 0:
            print('Step %d out of %d' % (s, S))

