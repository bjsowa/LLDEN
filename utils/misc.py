import errno
import os

import torch
from torch.utils.data.sampler import Sampler

from PIL import ImageFilter

__all__ = ['mkdir_p', 'AverageMeter', 'ClassSampler', 'GaussianNoise']


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClassSampler(Sampler):

    def __init__(self, labels, classes, start_from = 0, amount = None):
        self.indices = []
        start = [start_from] * len(classes)
        left = [amount] * len(classes)

        for i, label in enumerate(labels):
            if label in classes:
                idx = classes.index(label)

                if start[idx] == 0:
                    if left[idx] is None:
                        self.indices.append(i)
                    elif left[idx] > 0:
                        self.indices.append(i)
                        left[idx] -= 1
                else: 
                    start[idx] -= 1

    def __iter__(self):
        #return (i for i in range(self.prefix))
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class GaussianNoise(object):

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, img):
        noise = img.clone()
        noise = noise.normal_(self.mean, self.stddev)
        new_img = img + noise
        new_img = torch.clamp(new_img, 0, 1)
        return new_img