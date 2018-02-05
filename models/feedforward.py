import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, num_classes=10):
        super(FeedForward, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(28*28, 312),
            nn.ReLU(inplace = True),
            nn.Linear(312, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, num_classes)
        )

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return x

    def probabilities(self, x):
        logits = self(x)
        return self.softmax(logits)
