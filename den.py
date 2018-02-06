import torch
import torch.nn as nn
import torch.optim as optim

CUDA = True

def main():
    #TODO: get datasets, tau (threshold), sigma (threshold)...
    pass

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
    layers = [p for p in model.parameters()]
    reg = torch.norm(layers[-1], 1)
    return loss(outputs, targets) + coeff * reg

def loss4(model, outputs, targets):
    pass

def bfs(model):
    pass

def incremental_learning(datasets, tau, sigma):

    model = setup_model()

    for t, dataset in enumerate(datasets):
        if t == 0:
            criterion = lambda outputs, targets: 
                            loss1(model, output, targets)
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
     layers = [p for p in model.parameters()]
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
