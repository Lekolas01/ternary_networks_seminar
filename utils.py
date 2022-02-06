import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from torch.utils.data.dataloader import DataLoader

def get_mnist_dataloader(train, n_samples=None, **dl_args):
    transform = transforms.Compose([transforms.Resize((32, 32)),
        transforms.ToTensor()])

    dataset = datasets.MNIST(root='data/', 
        train=True, 
        transform=transform,
        download=True)

    if (n_samples is not None and 1 <= n_samples < len(dataset)):
        dataset = data_utils.Subset(dataset, torch.arange(n_samples))

    data_loader = DataLoader(dataset, **dl_args)

    return data_loader
    
