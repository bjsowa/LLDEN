

def main():
    #TODO: get datasets, tau (threshold), sigma (threshold)...
    pass

def incremental_learning(dataset, tau, sigma):

    for t, dataset in enumerate(datasets):
        if t == 0:
            #TODO: train weights W_0 using eq (2)
        else:
            
            loss, weights = selective_training(D, W)


def selective_training(dataset, prev_params):
    """Finds neurons that are relevant to the new
       task and retrains the network parameters
       associated with them.
    """
    #TODO: implementation
    pass

def dynamic_network_expansion(dataset, tau):
    """Expands the network capacity in a top-down
       manner, while eliminating any unnecessary
       neurons using group-sparsity regularization.
    """
    #TODO: implementation
    pass

def network_split_duplication(weight, sigma):
    """Calculates the drift for each unit to
       identify units that have drifted too much
       from their original values during training
       and duplicates them.
    """
    #TODO: implementation
    pass
