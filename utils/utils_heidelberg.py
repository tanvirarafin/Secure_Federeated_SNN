import numpy as np
import tables


def load_shd(datafile, S_prime, min_pxl_value=48, max_pxl_value=80, window_length=10000):

    hdf5_file = tables.open_file(datafile, 'r')
    num_samples = len(hdf5_file.root.labels)

    for i in range(num_samples):
        # variables to parse
        timestamps = hdf5_file.root.spikes.times[i] * 10e6  # get times in mus
        addr = hdf5_file.root.spikes.units[i]

        # create windows to store eventa
        windows = list(range(window_length, int(max(timestamps)), window_length))
        window_ptr = 0
        ts_pointer = 0
        n_neurons = (max_pxl_value - min_pxl_value + 1)**2

        timestamps_grouped = [[] for _ in range(len(windows))]
        current_group = []

        while (ts_pointer < len(timestamps)) & (window_ptr < len(windows)):
            if timestamps[ts_pointer] <= windows[window_ptr]:
                current_group += [ts_pointer]
            else:
                timestamps_grouped[window_ptr] += current_group
                window_ptr += 1
                current_group = [ts_pointer]
            ts_pointer += 1

        spiking_neurons_per_ts = [[addr[ts] for ts in group] for group in timestamps_grouped]

        if S_prime <= len(windows):
            input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(S_prime)])
            input_signal = input_signal.T[None, :, :]
        else:
            input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(len(windows))])
            padding = np.zeros([S_prime - len(windows), n_neurons])

            input_signal = np.vstack((input_signal, padding))
            input_signal = input_signal.T[None, :, :]

    return input_signal.astype(bool)
