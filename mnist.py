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
from torch.autograd import Variable
from progress.bar import Bar

from models import FeedForward
from utils import *

# PATHS
CHECKPOINT    = "./checkpoints/mnist"

# BATCH
BATCH_SIZE    = 256
NUM_WORKERS   = 4

# SGD
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4

# Step Decay
LR_DROP       = 0.5
EPOCHS_DROP   = 20

# MISC
EPOCHS        = 200
CUDA          = True

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)

#CLASSES = [5,3,4,9,8,7,0,1,2,6]
CLASSES = range(10)

def main():

    if not os.path.isdir(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    trainloader, validloader, testloader = load_MNIST(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    print("==> Creating model")
    model = FeedForward(num_classes=len(CLASSES))

    if CUDA:
        model = model.cuda()
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    print('    Total params: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), 
                    lr=LEARNING_RATE, 
                    momentum=MOMENTUM, 
                    weight_decay=WEIGHT_DECAY
                )

    print("==> Learning")

    for epoch in range(EPOCHS):

        adjust_learning_rate(optimizer, epoch + 1)

        print('\nEpoch: [%d | %d]' % (epoch + 1, EPOCHS))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer )
        test_loss, test_acc = train(validloader, model, criterion, test = True )

        # save model
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': test_loss,
            'optimizer': optimizer.state_dict()
            }, is_best)

    print("==> Calculating AUROC")

    filepath_best = os.path.join(CHECKPOINT, "best.pt")
    checkpoint = torch.load(filepath_best)
    model.load_state_dict(checkpoint['state_dict'])

    if CUDA:
        model = model.module

    train(testloader, model, criterion, test=True)
    auroc = calc_avg_AUROC(model, testloader, CLASSES, CUDA)

    print( 'AUROC: {}'.format(auroc) )

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

        # convert labels into one hot vectors
        targets_onehot = one_hot(targets, CLASSES)

        if CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_onehot = targets_onehot.cuda()

        inputs = Variable(inputs)
        targets_onehot = Variable(targets_onehot)

        #compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets_onehot)

        # record loss
        losses.update(loss.data[0], inputs.size(0))

        if not test:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(batchloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg)

def adjust_learning_rate(learning_rate, optimizer, epoch):
    if epoch % EPOCHS_DROP == 0:
        learning_rate *= LR_DROP
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    return learning_rate

def save_checkpoint(state, is_best):
    filepath = os.path.join(CHECKPOINT, "last.pt")
    torch.save(state, filepath)
    if is_best:
        filepath_best = os.path.join(CHECKPOINT, "best.pt")
        shutil.copyfile(filepath, filepath_best)

if __name__ == '__main__':
    main()
