import os
import sys
import struct
import socket
import pickle
from time import sleep
import time

async def send_data(comm, data, client_rank, tag_epoch):
    # print("MPI begin send")
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL) 
    # print("MPI begin send")
    # print("send rank: ", client_rank)
    comm.send(data, dest=client_rank, tag=tag_epoch)
    # print("MPI send")

async def get_data(comm, client_rank, tag_epoch):
    # print("MPI begin get")
    data = comm.recv(source=client_rank, tag=tag_epoch)
    data = pickle.loads(data)
    # print("MPI get")
    return data


# def send_data1(comm, data, client_rank, tag_epoch):
#     data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  
#     comm.send(data, dest=client_rank, tag=tag_epoch)
#     print("MPI send")

# def get_data1(comm, client_rank, tag_epoch):
#     data = comm.recv(source=client_rank, tag=tag_epoch)
#     data = pickle.loads(data)
#     print("MPI get")
#     return data