import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset

from .misc import ClassSampler, GaussianNoise

__all__ = ['load_MNIST']

DATA = './data'

def load_MNIST(classes = range(10), batch_size = 256, num_workers = 4):

    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.5)
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_all)
    testset = dataloader(root=DATA, train=False, download=False, transform=transform_all)

    allset = ConcatDataset([trainset, testset])
    labels = list(i[1] for i in allset)

    trainsampler = ClassSampler(labels, classes, amount=1000)
    trainloader = DataLoader(allset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = ClassSampler(labels, classes, amount=200, start_from=1000)
    validloader = DataLoader(allset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = ClassSampler(labels, classes, amount=5000, start_from=1200)
    testloader = DataLoader(allset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)