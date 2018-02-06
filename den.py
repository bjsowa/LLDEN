from __future__ import print_function

import os
import time
import shutil
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from progress.bar import Bar

from models import FeedForward
from utils import *

# PATHS
CHECKPOINT    = "./checkpoints/mnist-den"

# BATCH
BATCH_SIZE    = 256
NUM_WORKERS   = 4

# SGD
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0

# MISC
EPOCHS = 200
CUDA = True

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    print('==> Preparing dataset')

    trainloader, validloader, testloader = load_MNIST(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    print("==> Creating model")
    model = FeedForward(num_classes=10)

    if CUDA:
        model = model.cuda()
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    layers = list(model.module.classifier)
    criterion = MyLoss(1e-4, layers)
    optimizer = optim.SGD(model.parameters(), 
                lr=LEARNING_RATE, 
                momentum=MOMENTUM, 
                weight_decay=WEIGHT_DECAY
            )

    print("==> Learning")

    for epoch in range(EPOCHS):

        # adjust_learning_rate(optimizer, epoch + 1)

        print('\nEpoch: [%d | %d]' % (epoch + 1, EPOCHS))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer )
        test_loss, test_acc = train(validloader, model, criterion, test = True )

        # # save model
        # is_best = test_acc > best_acc
        # best_acc = max(test_acc, best_acc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'acc': test_acc,
        #     'optimizer': optimizer.state_dict()
        #     }, is_best)



class MyLoss(nn.Module):
    def __init__(self, coeff, layers):
        super(MyLoss, self).__init__()
        self.coeff = coeff
        self.layers = layers
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        reg = 0
        for layer in self.layers:
            for p in layer.parameters():
                reg += torch.norm(p,1)
        return self.loss(x,y) + self.coeff * reg


def train(batchloader, model, criterion, optimizer = None, test = False):
    
    # switch to train or evaluate mode
    if test:
        model.eval()
    else:
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if test:
        bar = Bar('Testing', max=len(batchloader))
    else:
        bar = Bar('Training', max=len(batchloader))

    for batch_idx, (inputs, targets) in enumerate(batchloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        #compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if not test:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(batchloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg)
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg)


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


def incremental_learning(model, datasets, tau, sigma):

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
