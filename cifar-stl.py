from __future__ import print_function

import os
import time
import shutil

import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from progress.bar import Bar

from models import AlexNet
from utils import *

# PATHS
CHECKPOINT = "./checkpoints/cifar-stl"
DATA = "./data"

# BATCH
BATCH_SIZE = 256
NUM_WORKERS = 4

# SGD
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Step Decay
LR_DROP = 0.5
EPOCHS_DROP = 30

# MISC
EPOCHS = 300
CUDA = True

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    dataloader = datasets.CIFAR100
    num_classes = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_train)
    trainlabels = list(i[1] for i in trainset) 
    trainsampler = ClassSampler(trainlabels, range(10))
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler = trainsampler, num_workers=NUM_WORKERS)

    testset = dataloader(root=DATA, train=False, download=False, transform=transform_test)
    testlabels = list(i[1] for i in testset)
    testsampler = ClassSampler(testlabels, range(10))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, sampler = testsampler, num_workers=NUM_WORKERS)

    print("==> Creating model")
    model = AlexNet(num_classes=num_classes)

    if CUDA:
        model = model.cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    criterionL1 = nn.L1Loss
    optimizer = optim.SGD(model.parameters(), 
                    lr=LEARNING_RATE, 
                    momentum=MOMENTUM, 
                    weight_decay=WEIGHT_DECAY
                )

    for epoch in range(EPOCHS):

        adjust_learning_rate(optimizer, epoch + 1)

        print('\nEpoch: [%d | %d]' % (epoch + 1, EPOCHS))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer )
        test_loss, test_acc = train(testloader, model, criterion, test = True )

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'optimizer': optimizer.state_dict()
            }, is_best)

    filepath_best = os.path.join(CHECKPOINT, "best.pt")
    checkpoint = torch.load(filepath_best)
    model.load_state_dict(checkpoint['state_dict'])

    #test_loss, test_acc = train(testloader, model, criterion, test = True )

    # TODO
    # Calculate AUROC for each class (One vs All)


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

        # print( targets) 

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

def adjust_learning_rate(optimizer, epoch):
    global LEARNING_RATE

    if epoch % EPOCHS_DROP == 0:
        LEARNING_RATE *= LR_DROP
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

def save_checkpoint(state, is_best):
    filepath = os.path.join(CHECKPOINT, "last.pt")
    torch.save(state, filepath)
    if is_best:
        filepath_best = os.path.join(CHECKPOINT, "best.pt")
        shutil.copyfile(filepath, filepath_best)


if __name__ == '__main__':
    main()
