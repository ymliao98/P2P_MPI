from fileinput import filename
import imp
import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
from tkinter.tix import Tree
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
import datasets, models
from training_utils import test

from mpi4py import MPI

import logging


#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='/data/docker/data')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/server/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

def main():
    logger.info("csize:{}".format(int(csize)))
    logger.info("server start (rank):{}".format(int(rank)))
    # init config
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.data_path = args.data_path
    common_config.weight_decay = args.weight_decay

    worker_num = int(csize)-1

    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    common_config.para_nums=init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(common_config.para_nums))
    logger.info("Model Size: {} MB".format(model_size))

    # create workers
    worker_list: List[Worker] = list()
    for worker_idx in range(worker_num):
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config),rank=worker_idx+1)
        )
    #到了这里，worker已经启动了

    # Create model instance
    train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.train_data_idxes = train_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(worker_list, 1, comm, action="init")

    # recoder: SummaryWriter = SummaryWriter()
    global_model.to(device)
    _, test_dataset = datasets.load_datasets(common_config.dataset_type,common_config.data_path)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    adjacency_matrix = np.zeros((worker_num, worker_num), dtype=np.int)

    # for worker_idx in range(worker_num):
    #     adjacency_matrix[worker_idx][worker_idx-1] = 1
    #     adjacency_matrix[worker_idx][(worker_idx+1)%worker_num] = 1

    for worker_idx1 in range(worker_num):
        for worker_idx2 in range(worker_num):
            if worker_idx1!=worker_idx2:
                adjacency_matrix[worker_idx1][worker_idx2] = 1

    topology=adjacency_matrix
    logging.info(topology)
    print(topology)
    update_W(topology,worker_list)
    for epoch_idx in range(1, 1+common_config.epoch):
        communication_parallel(worker_list, epoch_idx, comm, action="send_para")
        # print("send end")
        communication_parallel(worker_list, epoch_idx, comm, action="get_para")

        avg_acc = 0.0
        avg_test_loss = 0.0
        avg_train_loss = 0.0
        for worker in worker_list:
            acc, test_loss,train_loss = worker.config.worker_paras
            avg_acc += acc
            avg_test_loss += test_loss
            avg_train_loss += train_loss

        avg_acc /= worker_num
        avg_test_loss /= worker_num
        avg_train_loss /= worker_num
        print("Epoch: {}, average accuracy: {}, average test_loss: {}, average train_loss: {}\n".format(epoch_idx, avg_acc, avg_test_loss, avg_train_loss))
        logger.info("Epoch: {}, average accuracy: {}, average test_loss: {}, average train_loss: {}\n".format(epoch_idx, avg_acc, avg_test_loss, avg_train_loss))

        topology=update_T(worker_num)
        logging.info(topology)
        print(topology)
        update_W(topology,worker_list)
    # exit()
    # close socket

def communication_parallel(worker_list, epoch_idx, comm, action, data=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for worker in worker_list:
        if action == "init":
            task = asyncio.ensure_future(worker.send_init_config(comm, epoch_idx))
        elif action == "get_para":
            task = asyncio.ensure_future(worker.get_data(comm, epoch_idx))
        elif action == "send_para":
            data=worker.config.comm_neighbors
            task = asyncio.ensure_future(worker.send_data(data, comm, epoch_idx))
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

def update_T(worker_num):
    worker_selected=dict()
    for i in range(worker_num):
        worker_selected[i]=0
    topology = np.zeros((worker_num, worker_num), dtype=np.int)
    for i in range(worker_num):
        if worker_selected[i]<3:
            nei_list=random.sample(range(0,9),2)
            for nei in nei_list:
                if nei==i:
                    nei=(nei+1)%10
                topology[i][nei]=1
                topology[nei][i]=1

    return topology


def update_W(topology, worker_list):
    for worker in worker_list:
        for i in range(len(worker_list)):
            if topology[worker.rank-1][i]==1:
                worker.config.comm_neighbors.append(i+1)


def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num=10
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    elif dataset_type == "EMNIST":
        train_class_num=62
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    if dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num=100
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    return train_data_partition

if __name__ == "__main__":
    main()
