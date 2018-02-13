import time
import os
import shutil
import random

import torch
from torch.autograd import Variable
from progress.bar import Bar

from .misc import AverageMeter

__all__ = ['train', 'save_checkpoint']

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def one_hot(targets, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in classes:
            targets_onehot[i][classes.index(t)] = 1
    return targets_onehot

def train(batchloader, model, criterion, all_classes, classes, optimizer = None, penalty = None, test = False, use_cuda = False):
    
    # switch to train or evaluate mode
    if test:
        model.eval()
    else:
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if test:
        bar = Bar('Testing', max=len(batchloader))
    else:
        bar = Bar('Training', max=len(batchloader))

    for batch_idx, (inputs, targets) in enumerate(batchloader):

        # measure data loading time
        data_time.update(time.time() - end)

        # convert labels into one hot vectors
        targets_onehot = one_hot(targets, classes)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_onehot = targets_onehot.cuda()

        inputs = Variable(inputs)
        targets_onehot = Variable(targets_onehot)

        # compute output
        outputs = model(inputs)

        # calculate loss
        loss = 0
        for i, cls in enumerate(classes):
            loss = loss + criterion(outputs[:, all_classes.index(cls)], targets_onehot[:, i])
        if penalty is not None:
        	loss = loss + penalty(model)

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
    return losses.avg

def save_checkpoint(state, path, is_best = False):
    filepath = os.path.join(path, "last.pt")
    torch.save(state, filepath)
    if is_best:
        filepath_best = os.path.join(path, "best.pt")
        shutil.copyfile(filepath, filepath_best)