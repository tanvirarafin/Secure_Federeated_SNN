import tables
import os
import glob
from utils_dvs import load_dvs
import re
import numpy as np
import math


""""
Preprocess the .aedat file and save the dataset as an .hdf5 file
"""


path = r'path/to/mnist-dvs-processed'
path = r'C:\Users\K1804053\Desktop\PhD\Federated SNN\processed'

dirs = [r'/' + dir_ for dir_ in os.listdir(path)]
allFiles = {key: [] for key in dirs}

T_max = int(2e6)  # maximum duration of an example in us
window_length = 25000
S_prime = math.ceil(T_max/window_length)
scale = 'scale4'
# Pixel values to consider
max_pxl_value = 73
min_pxl_value = 48

hdf5_file = tables.open_file(r'mnist_dvs_%dms_%dpxl.hdf5' % (int(window_length/1000), max_pxl_value - min_pxl_value + 1), 'w')
data = hdf5_file.create_earray(where=hdf5_file.root, name='data', atom=tables.BoolAtom(), shape=(0, (max_pxl_value - min_pxl_value + 1)**2, S_prime))
labels = hdf5_file.create_earray(where=hdf5_file.root, name='label', atom=tables.BoolAtom(), shape=(0, 2, S_prime))

digits = [i for i in range(10)]
pattern = [1, 0, 0, 0, 0]   # the pattern used as output for the considered digit

for i, digit in enumerate(digits):
    for dir_ in dirs:
            if dir_.find(str(digit)) != -1:
                for subdir, _, _ in os.walk(path + dir_):
                    if subdir.find(scale) != -1:
                        print(digit, subdir)
                        for file in glob.glob(subdir + r'/*.aedat'):
                            data.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value, window_length=window_length))

                            output_signal = np.array([[[0] * S_prime]*i
                                                      + [pattern * int(S_prime/len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                                      + [[0] * S_prime]*(len(digits) - 1 - i)], dtype=bool)
                            labels.append(output_signal)
hdf5_file.close()
