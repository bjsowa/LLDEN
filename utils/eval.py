from __future__ import print_function, absolute_import

import torch
from torch.autograd import Variable
import numpy as np

__all__ = ['one_hot', 'accuracy', 'calc_avg_AUROC', 'AUROC']

def one_hot(targets, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in classes:
            targets_onehot[i][classes.index(t)] = 1
    return targets_onehot

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calc_avg_AUROC(model, batchloader, classes, use_cuda, num_classes = 10):
    """Calculates average of the AUROC for selected classes in the dataset
    """
    sum_targets = torch.cuda.LongTensor() if use_cuda else torch.LongTensor()
    sum_outputs = torch.cuda.FloatTensor() if use_cuda else torch.FloatTensor()

    for batch_idx, (inputs, targets) in enumerate(batchloader):

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        outputs = model(inputs).data

        sum_targets = torch.cat((sum_targets, targets), 0)
        sum_outputs = torch.cat((sum_outputs, outputs), 0)

    sum_area = 0
    for i, cls in enumerate(classes):
        scores = sum_outputs[:, i]
        sum_area += AUROC(scores.cpu().numpy(), (sum_targets == cls).cpu().numpy())
    
    return (sum_area / len(classes))

def AUROC(scores, targets):
    """Calculates the Area Under the Curve.
    Args:
        scores: Probabilities that target should be possitively classified.
        targets: 0 for negative, and 1 for positive examples.
    """
    # case when number of elements added are 0
    if scores.shape[0] == 0:
        return 0.5
    
    # sorting the arrays
    scores, sortind = torch.sort(torch.from_numpy(
        scores), dim=0, descending=True)
    scores = scores.numpy()
    sortind = sortind.numpy()

    # creating the roc curve
    tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
    fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

    for i in range(1, scores.size + 1):
        if targets[sortind[i - 1]] == 1:
            tpr[i] = tpr[i - 1] + 1
            fpr[i] = fpr[i - 1]
        else:
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1] + 1

    tpr /= (targets.sum() * 1.0)
    fpr /= ((targets - 1.0).sum() * -1.0)

    # calculating area under curve using trapezoidal rule
    n = tpr.shape[0]
    h = fpr[1:n] - fpr[0:n - 1]
    sum_h = np.zeros(fpr.shape)
    sum_h[0:n - 1] = h
    sum_h[1:n] += h
    area = (sum_h * tpr).sum() / 2.0

    return area