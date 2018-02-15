from __future__ import print_function

import os
import random
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

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

# Step Decay
LR_DROP       = 0.5
EPOCHS_DROP   = 20

# MISC
EPOCHS = 50
CUDA = True

# L1 REGULARIZATION
L1_COEFF = 1e-5

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)

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

    print('    Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000) )

    criterion = nn.BCELoss()

    CLASSES = []
    AUROCs = []

    # threshold = 1e-6
    # suma = 0
    # for p in model.parameters():
    #     p = p.data.cpu().numpy()
    #     suma += (p < threshold).sum()
    # print(suma)

    for t, cls in enumerate(ALL_CLASSES):

        print('\nTask: [%d | %d]\n' % (t + 1, len(ALL_CLASSES)))

        CLASSES.append(cls)

        if t == 0:
            print("==> Learning")

            optimizer = optim.SGD(model.parameters(), 
                    lr=LEARNING_RATE, 
                    momentum=MOMENTUM, 
                    weight_decay=WEIGHT_DECAY
                )

            penalty = l1_penalty(coeff = L1_COEFF)
            best_loss = 1e10
            learning_rate = LEARNING_RATE

            # for name, param in model.named_parameters():
            #     if 'bias' not in name:
            #         param.register_hook(my_hook)

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
                is_best = test_loss > best_loss
                best_loss = min(test_loss, best_loss)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': test_loss,
                    'optimizer': optimizer.state_dict()
                    }, CHECKPOINT, is_best)

        else:
            print("==> Selective Retraining")

            # copy model 
            old_model = copy.deepcopy(model)

            # freeze all layers except the last one (last 2 parameters)
            params = list(model.parameters())
            for param in params[:-2]:
                param.requires_grad = False

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE, 
                momentum=MOMENTUM, 
                weight_decay=WEIGHT_DECAY
            )

            penalty = l1_penalty(coeff = L1_COEFF)
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
                is_best = test_loss > best_loss
                best_loss = min(test_loss, best_loss)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': test_loss,
                    'optimizer': optimizer.state_dict()
                    }, CHECKPOINT, is_best)


                # threshold = 1e-6
                # params = list(model.parameters())
                # p = params[-2].data.cpu().numpy()
                # suma = (p < threshold).sum()
                # print(suma)



        print("==> Calculating AUROC")

        filepath_best = os.path.join(CHECKPOINT, "best.pt")
        checkpoint = torch.load(filepath_best)
        model.load_state_dict(checkpoint['state_dict'])

        auroc = calc_avg_AUROC(model, testloader, ALL_CLASSES, CLASSES, CUDA)

        print( 'AUROC: {}'.format(auroc) )

        AUROCs.append(auroc)

    print( '\nAverage Per-task Performance over number of tasks' )
    for i, p in enumerate(AUROCs):
        print("%d: %f" % (i+1,p))

def my_hook(grad):
    print("")
    print(grad)

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

def bfs(model, outputs):
    model = copy.deepcopy(model)
    
    # TODO: set all to ones? or outputs?
    # prev_active = np.ones(outputs.size())
    prev_active = outputs
    
    layers = reversed(list(model.classifier.parameters()))
    
    for layer in layers:

        x_size, y_size = layer.size()
        for x in range(x_size):

            # we skip the weight if connected neuron wasn't selected
            if (prev_active[x] < 1e-2):
                continue

            for y in range(y_size):
                weight = layer[x][y]
                # check if weight is active
                if (weight > 1e-2):
                    # mark affected connected neuron as active
                    active[y] = 1

            # TODO: do something with active neurons
        
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
