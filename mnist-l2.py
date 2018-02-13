from __future__ import print_function

import os
import time
import shutil
import random
import copy

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
CHECKPOINT    = "./checkpoints/mnist-l2"

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
EPOCHS        = 100
CUDA          = True

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)

#CLASSES = [5,3,4,9,8,7,0,1,2,6]
ALL_CLASSES = range(10)

def main():

    if not os.path.isdir(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    trainloader, validloader, testloader = load_MNIST(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    print("==> Creating model")
    model = FeedForward(num_classes=len(ALL_CLASSES))

    if CUDA:
        model = model.cuda()
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    print('    Total params: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.BCELoss()
    penalty = None

    CLASSES = []
    AUROCs = []

    for t, cls in enumerate(ALL_CLASSES):

        optimizer = optim.SGD(model.parameters(), 
                lr=LEARNING_RATE, 
                momentum=MOMENTUM, 
                weight_decay=WEIGHT_DECAY
            )

        print('\nTask: [%d | %d]\n' % (t + 1, len(ALL_CLASSES)))

        CLASSES.append(cls)

        print("==> Learning")

        best_loss = 1e10
        learning_rate = LEARNING_RATE

        for epoch in range(EPOCHS):

            # decay learning rate
            if (epoch + 1) % EPOCHS_DROP == 0:
                learning_rate *= LR_DROP
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            print('Epoch: [%d | %d]' % (epoch + 1, EPOCHS))

            train_loss = train(trainloader, model, criterion, ALL_CLASSES, [cls], optimizer = optimizer, penalty = penalty, use_cuda = CUDA)
            test_loss = train(validloader, model, criterion, ALL_CLASSES, [cls], test = True, penalty = penalty, use_cuda = CUDA)

            # save model
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': test_loss,
                'optimizer': optimizer.state_dict()
                }, CHECKPOINT, is_best)

        print("==> Calculating AUROC")

        filepath_best = os.path.join(CHECKPOINT, "best.pt")
        checkpoint = torch.load(filepath_best)
        model.load_state_dict(checkpoint['state_dict'])

        auroc = calc_avg_AUROC(model, testloader, ALL_CLASSES, CLASSES, CUDA)

        print( 'AUROC: {}'.format(auroc) )

        AUROCs.append(auroc)

        # create a copy of current model and make the weights fixed
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.requires_grad = False

        # create l2 norm penalty for the next task
        penalty = l2_penalty(model_copy)

    print( '\nAverage Per-task Performance over number of tasks' )
    for i, p in enumerate(AUROCs):
        print("%d: %f" % (i+1,p))


class l2_penalty(object):
    def __init__(self, model, coeff = 1e-2):
        self.old_model = model
        self.coeff = coeff

    def __call__(self, new_model):
        penalty = 0
        for ((name1, param1), (name2, param2)) in zip(self.old_model.named_parameters(), new_model.named_parameters()):
            if name1 != name2 or param1.shape != param2.shape:
                raise Exception("model parameters do not match!")

            # get only weight parameters
            if 'bias' not in name1:
                diff = param1 - param2
                penalty = penalty + diff.norm(2)

        return self.coeff * penalty

if __name__ == '__main__':
    main()