import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self, config, rank):
        #这个config就是后面的client_config
        self.config = config
        self.rank = rank

    async def send_data(self, data, comm, epoch_idx):
        await send_data(comm, data, self.rank, epoch_idx)    

    async def send_init_config(self, comm, epoch_idx):
        print("before send", self.rank, "tag:", epoch_idx)
        await send_data(comm, self.config, self.rank, epoch_idx)    

    async def get_data(self, comm, epoch_idx):
        self.config.worker_paras = await get_data(comm, self.rank, epoch_idx)

class CommonConfig:
    def __init__(self):
        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None
        self.para = None
        self.data_path = None
        self.neighbor_paras = dict()
        self.comm_neighbors = None
        self.train_loss = None
        self.tag=None
        #这里用来存worker的


class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        self.para = None
        self.train_data_idxes = None
        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=20
        self.compre_ratio=1
        self.train_time=0
        self.send_time=0
        self.worker_paras=None
        self.comm_neighbors=list()