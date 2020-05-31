from __future__ import print_function
import datetime
import torch
import numpy as np
import utils.filters as filters
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from utils.training_utils import local_feedback_and_update, feedforward_sampling, get_acc_and_loss, refractory_period

import tables

def run(rank,size):
    pass

def init_process(rank, world_size,train_func):
    """"
    Initialize process group and launches training on the nodes
    """ 
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8844'
    backend ='gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print('Process %d started' % rank)
    train_func(rank, world_size)

if __name__ == "__main__":
    # setting the hyper parameters
    size = 4
    processes = []
    for local_rank in range(size):
        p = mp.Process(target=init_process, args=(local_rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
