from __future__ import print_function

import os
import time
import shutil
import random

import numpy as np
from matplotlib.pyplot import imshow, savefig

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from progress.bar import Bar

from models import FeedForward
from utils import *


# BATCH
BATCH_SIZE    = 256
NUM_WORKERS   = 4

CUDA = True

def main():

    print('==> Preparing dataset')

    trainloader, validloader, testloader = load_MNIST(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)


def setup_model():
    model = AlexNet(num_classes=100)
    model = torch.nn.DataParallel(model)
    if CUDA:
        model = model.cuda
    return model

def loss2(model, outputs, targets):
    loss = nn.CrossEntropyLoss()
    coeff = 1e-4
    reg = 0
    for layer in model.parameters():
        reg += torch.norm(layer, 1)
    return loss(outputs, targets) + coeff * reg

def loss3(model, outputs, targets):
    loss = nn.CrossEntropyLoss()
    coeff = 1e-4
    layers = list(model.parameters())
    reg = torch.norm(layers[-1], 1)
    return loss(outputs, targets) + coeff * reg

def loss4(model, outputs, targets):
    pass

def bfs(model):
    model = copy.deepcopy(model)

    prev_active = []
    for layer in reversed(list(model.parameters())):
        for neuron in layer:
            pass
        prev_active = active


def incremental_learning(datasets, tau, sigma):

    model = setup_model()

    for t, dataset in enumerate(datasets):
        if t == 0:
            criterion = (lambda outputs, targets: loss1(model, output, targets))
            train(dataset, model, loss1)
        else:
            loss, model = selective_training(prev_model)
            if loss > tau:
                model = dynamic_network_expansion(model)

def selective_training(model):
    """Finds neurons that are relevant to the new
       task and retrains the network parameters
       associated with them.
    """
    # freeze all layers except the last one
    layers = list(model.parameters())
    for layer in layers[:-1]:
        layer.requires_grad = False

    # train the network and receive sparse
    # connections on the last layer
    train(dataset, model, loss3)

    # use breadth-first search on the network
    # to receive set of affected neurons
    subnetwork = bfs(model)

    # train only the weights of the acquired
    # subnetwork
    train(dataset, subnetwork, loss4)

def dynamic_network_expansion(weights, tau):
    """Expands the network capacity in a top-down
       manner, while eliminating any unnecessary
       neurons using group-sparsity regularization.
    """
    #TODO: implementation
    pass

def network_split_duplication(weights, sigma):
    """Calculates the drift for each unit to
       identify units that have drifted too much
       from their original values during training
       and duplicates them.
    """
    #TODO: implementation
    pass


if __name__ == '__main__':
    main()
