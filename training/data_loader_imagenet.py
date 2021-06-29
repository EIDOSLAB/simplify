import os

import numpy as np
import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets
from torchvision import transforms


class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn
    
    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)


def get_data_loaders(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers, pin_memory,
                     random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.ImageNet DataSet.
    :param data_dir: Location of the DataSet. Downloading not supported.
    :param train_batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    data_dir = os.path.join(data_dir)
    
    parent_dataset = datasets.ImageNet(
        root=data_dir, split="train"
    )
    
    if valid_size > 0:
        
        dataset_length = len(parent_dataset)
        valid_length = int(np.floor(valid_size * dataset_length))
        train_length = dataset_length - valid_length
        train_dataset, valid_dataset = random_split(parent_dataset,
                                                    [train_length, valid_length],
                                                    generator=torch.Generator().manual_seed(random_seed))
        
        train_dataset = MapDataset(train_dataset, transform_train)
        valid_dataset = MapDataset(valid_dataset, transform_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=test_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory
        )
    
    else:
        parent_dataset = MapDataset(parent_dataset, transform_train)
        train_loader = torch.utils.data.DataLoader(
            parent_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory
        )
    
    test_dataset = datasets.ImageNet(
        root=data_dir, split="val", transform=transform_test
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory
    )
    
    return (train_loader, valid_loader, test_loader) if valid_size > 0 else (train_loader, test_loader)
