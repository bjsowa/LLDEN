from __future__ import print_function

import os
import time

import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from progress.bar import Bar

from alexnet import AlexNet
from utils import *

# PATHS
CHECKPOINT = "./checkpoints"
DATA = "./data"

# BATCH
BATCH_SIZE = 8
NUM_WORKERS = 4

# SGD
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# MISC
EPOCHS = 300
CUDA = False

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    dataloader = datasets.CIFAR100
    num_classes = 100

    trainset = dataloader(root=DATA, train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    testset = dataloader(root=DATA, train=False, download=False, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("==> creating model")
    model = AlexNet(num_classes=100)

    if CUDA:
        model = model.cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                    lr=LEARNING_RATE, 
                    momentum=MOMENTUM, 
                    weight_decay=WEIGHT_DECAY
                )

    for epoch in range(EPOCHS):

        print('\nEpoch: [%d | %d]' % (epoch + 1, EPOCHS))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer )
        test_loss, test_acc = train(testloader, model, criterion, test = True )


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

if __name__ == '__main__':
    main()