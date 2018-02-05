import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*128, 384),
            nn.Linear(384, 192),
            nn.Linear(192,num_classes)
        )

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def probabilities(self, x):
        logits = self(x)
        return self.softmax(logits)
