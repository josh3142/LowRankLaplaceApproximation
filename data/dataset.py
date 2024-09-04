import torch
from torch import nn, Tensor
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import (Compose, Normalize, ToTensor, Resize,
    RandomHorizontalFlip, RandomGrayscale, RandomApply, RandomResizedCrop)
from torchvision.transforms import ElasticTransform, GaussianBlur
from torch.utils.data import Dataset
import numpy as np

from typing import Callable, Optional, Tuple, Union

from data.redwine import get_redwine, get_redwine_trafo
from data.protein import get_protein, get_protein_trafo
from data.california import get_california, get_california_trafo
from data.enb import get_enb, get_enb_trafo
from data.navalprop import get_navalpro, get_navalpro_trafo


def get_dataset(
        name: str, 
        path: str, 
        train: bool, 
    ) -> Dataset:
    
    if name.lower()=="cifar10":
        def transform(train: bool) -> Callable:
            mu  = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
            trafo = [ToTensor(), Normalize(mu, std)]
            if train:
                trafo += [
                    RandomHorizontalFlip(p = 0.5),
                    RandomGrayscale(p = 0.3),
                    RandomApply(
                        nn.ModuleList([
                            RandomResizedCrop(32, scale=(0.2, 1.0), antialias=True)]), 
                        p = 0.5)
                ]
            return Compose(trafo)
        
        data = CIFAR10(path, train=train, transform=transform(train), 
            download=True)
        
    elif name.lower()=="cifar10_corrupt":
        def transform(train: bool) -> Callable:
            mu  = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
            trafo = [
                ElasticTransform(alpha=50.),
                ToTensor(), 
                Normalize(mu, std)]
            
            return Compose(trafo)
        
        data = CIFAR10(path, train=train, transform=transform(train), 
            download=True)

        
    elif name.lower()=="mnist":
        def transform() -> Callable:
            return Compose([ToTensor(), Normalize((0.5, ), (0.5,))])
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)

    elif name.lower()=="mnist_corrupt":
        def transform() -> Callable:
            return Compose([ElasticTransform(alpha=150.),
                            #GaussianBlur(kernel_size=(7, 7), sigma=(2., 5.)),
                            ToTensor(), 
                            Normalize((0.5, ), (0.5,))]
                        )
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)


    elif name.lower()=="mnist_small":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(14, antialias=True), 
                 Normalize((0.5, ), (0.5,))]
            )
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="mnist_small2class":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(7, antialias=True), 
                 Normalize((0.5, ), (0.5,))]
            )
        
        # extract zeroth (0) and first (1) class
        data = MNIST(path, train=train, transform=None, 
            download=True)
        idcs = torch.logical_or(data.targets == 0, data.targets == 1)
        X, Y = data.data.numpy()[idcs, ...], data.targets[idcs] 
        data = DatasetGenerator(X, Y, transform=transform())

    elif name.lower()=="redwine":
        if train:
            X, Y, _, _ = get_redwine()
        else:
            _, _, X, Y = get_redwine()
        data = DatasetGenerator(X, Y, transform=get_redwine_trafo(train))

    elif name.lower()=="protein":
        if train:
            X, Y, _, _ = get_protein()
        else:
            _, _, X, Y = get_protein()
        data = DatasetGenerator(X, Y, transform=get_protein_trafo(train))

    elif name.lower()=="california":
        if train:
            X, Y, _, _ = get_california()
        else:
            _, _, X, Y = get_california()
        data = DatasetGenerator(X, Y, transform=get_california_trafo(train))

    elif name.lower()=="enb":
        if train:
            X, Y, _, _ = get_enb()
        else:
            _, _, X, Y = get_enb()
        data = DatasetGenerator(X, Y, transform=get_enb_trafo(train))

    elif name.lower()=="navalpro":
        if train:
            X, Y, _, _ = get_navalpro(path)
        else:
            _, _, X, Y = get_navalpro(path)
        data = DatasetGenerator(X, Y, transform=get_navalpro_trafo(train))

    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
        
    return data


class DatasetGenerator(Dataset):
    """ Create a dataset """
    def __init__(
            self, 
            X: Union[Tensor, np.ndarray], 
            Y: Union[Tensor, np.ndarray], 
            transform: Optional[Callable]=None, 
        ):
        self.X         = X
        if len(Y.shape) > 1:
            self.Y         = Y
        else:
            self.Y = Y[...,None] # add target dimension
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.X[idx], self.Y[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.Y)