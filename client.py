import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from pulp import *
import random
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train, test
import datasets, models
from mpi4py import MPI
import logging

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank)% 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# init logger
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)


filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] + '_'+ str(int(rank)) +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# end logger

MASTER_RANK=0

async def get_init_config(comm, MASTER_RANK, config):
    logger.info("before init")
    config_received = await get_data(comm, MASTER_RANK, 1)
    logger.info("after init")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

def main():
    logger.info("client_rank:{}".format(rank))
    client_config = ClientConfig(
        common_config=CommonConfig()
    )

    logger.info("start")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(comm,MASTER_RANK,client_config))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.data_path = client_config.common_config.data_path
    common_config.para=client_config.para
    
    common_config.tag = 1
    # init config
    logger.info(str(common_config.__dict__))

    logger.info(str(len(client_config.train_data_idxes)))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type, common_config.data_path)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.train_data_idxes)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)
    local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    common_config.para=local_model

    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(comm, common_config, train_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        local_model=common_config.para
        logger.info("get begin")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []

        local_para=torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
        for i in range(len(common_config.comm_neighbors)):
            l=len(common_config.comm_neighbors)
            logger.info("nei:{}".format(common_config.comm_neighbors[i]))

            if common_config.comm_neighbors[i] > rank:
                task = asyncio.ensure_future(send_para(comm, local_para, common_config.comm_neighbors[i], common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(get_para(comm, common_config, common_config.comm_neighbors[i], common_config.tag))
                tasks.append(task)
            else:
                task = asyncio.ensure_future(get_para(comm, common_config, common_config.comm_neighbors[i], common_config.tag))
                tasks.append(task)
                # print("worker send")
                task = asyncio.ensure_future(send_para(comm, local_para, common_config.comm_neighbors[i], common_config.tag))
                tasks.append(task)
        logger.info(len(tasks))

        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        logger.info("get end")

        local_para = aggregate_model(local_para, common_config)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
        common_config.para=local_model

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training2(comm, common_config, test_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        if common_config.tag==common_config.epoch+1:
            break
    # exit()

async def local_training(comm, common_config, train_loader):
    comm_neighbors = await get_data(comm, MASTER_RANK, common_config.tag)
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    epoch_lr = common_config.lr
    
    local_steps = 20
    if common_config.tag > 1 and common_config.tag % 1 == 0:
        epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        common_config.lr = epoch_lr
    logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))
    if common_config.momentum<0:
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
    else:
        optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
    # local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()

    common_config.comm_neighbors = comm_neighbors
    common_config.para = local_model
    common_config.train_loss = train_loss

async def local_training2(comm, common_config, test_loader):
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model = common_config.para
    local_model.to(device)
    # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
    logger.info("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(common_config.tag, common_config.train_loss, test_loss, acc))
    logger.info("send para")

    data=(acc, test_loss, common_config.train_loss)
    # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
    await send_data(comm, data, MASTER_RANK, common_config.tag)
    logger.info("after send")
    # local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    # common_config.para=local_para
    # common_config.tag = common_config.tag+1
    # logger.info("get end")
    common_config.tag = common_config.tag+1

async def send_para(comm, data, rank, epoch_idx):
    # print("send_data")
    logger.info("send_data")
    # print("send rank {}, get rank {}".format(comm.Get_rank(), rank))
    logger.info("send rank {}, get rank {}".format(comm.Get_rank(), rank))
    # print("get rank: ", rank)
    await send_data(comm, data, rank, epoch_idx)

async def get_para(comm, common_config, rank, epoch_idx):
    # print("get_data")
    logger.info("get_data")
    logger.info("get rank {}, send rank {}".format(comm.Get_rank(), rank))
    common_config.neighbor_paras[rank] = await get_data(comm, rank, epoch_idx)

def aggregate_model(local_para, common_config):
    with torch.no_grad():
        weight=1.0/(len(common_config.comm_neighbors)+1)
        para_delta = torch.zeros_like(local_para)
        for neighbor_name in common_config.comm_neighbors:
            logger.info("idx: {},".format(neighbor_name))
            model_delta = common_config.neighbor_paras[neighbor_name] - local_para
            para_delta += weight * model_delta

        local_para += para_delta
    return local_para

if __name__ == '__main__':
    main()
