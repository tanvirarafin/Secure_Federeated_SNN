import tables
import os
import glob
from utils_dvs import load_dvs
from utils_heidelberg import load_shd
import re
import numpy as np
import math



def make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale):
    """"
    Preprocess the .aedat file and save the dataset as an .hdf5 file
    """

    dirs = [r'/' + dir_ for dir_ in os.listdir(path_to_data)]

    S_prime = math.ceil(T_max/window_length)

    pattern = [1, 0, 0, 0, 0]   # the pattern used as output for the considered digit

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_data = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), shape=(0, (max_pxl_value - min_pxl_value + 1)**2, S_prime))
    train_labels = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), S_prime))

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), shape=(0, (max_pxl_value - min_pxl_value + 1)**2, S_prime))
    test_labels = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), S_prime))


    for i, digit in enumerate(digits):
        for dir_ in dirs:
                if dir_.find(str(digit)) != -1:
                    for subdir, _, _ in os.walk(path_to_data + dir_):
                        if subdir.find(scale) != -1:
                            for j, file in enumerate(glob.glob(subdir + r'/*.aedat')):
                                if j < 0.9*len(glob.glob(subdir + r'/*.aedat')):
                                    print('train', file)
                                    train_data.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value, window_length=window_length))

                                    output_signal = np.array([[[0] * S_prime]*i
                                                              + [pattern * int(S_prime/len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                                              + [[0] * S_prime]*(len(digits) - 1 - i)], dtype=bool)
                                    train_labels.append(output_signal)
                                else:
                                    print('test', file)
                                    test_data.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value, window_length=window_length))

                                    output_signal = np.array([[[0] * S_prime] * i
                                                              + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                                              + [[0] * S_prime] * (len(digits) - 1 - i)], dtype=bool)
                                    test_labels.append(output_signal)

    hdf5_file.close()


def make_shd(path_to_train, path_to_test, path_to_hdf5, digits, window_length):

    data = tables.open_file(path_to_data, 'r')

    n_neurons = 700
    T_max = max([max(data.root.spikes.times[i]) for i in range(len(data.root.labels))])
    S_prime = math.ceil(T_max/window_length)

    pattern = [1, 0, 0, 0, 0]   # the pattern used as output for the considered digit

    hdf5_file = tables.open_file(path_to_hdf5, 'w')
    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    data = hdf5_file.create_array(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), obj=load_shd(data, S_prime, digits, window_length))
    labels = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), S_prime))


path_to_data = r'path/to/mnist-dvs-processed'

# digits to consider
digits = [i for i in range(10)]

# Pixel values to consider
max_pxl_value = 73
min_pxl_value = 48

T_max = int(2e6)  # maximum duration of an example in us
window_length = 25000

scale = 'scale4'

path_to_hdf5 = r'path/to/datasets/mnist-dvs/mnist_dvs_%dms_%dpxl_%d_digits.hdf5' \
               % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))


# make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale)





