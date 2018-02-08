import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):

    def __init__(self, num_classes=10):
        super(FeedForward, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(28*28, 312),
            nn.ReLU(inplace = True),
            nn.Linear(312, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return x
