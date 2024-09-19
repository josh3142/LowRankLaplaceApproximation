import os

import torch
from torch import nn, Tensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import (Compose, Normalize, ToTensor, Resize,
    RandomHorizontalFlip, RandomGrayscale, RandomApply, RandomResizedCrop)
from torchvision.transforms import ElasticTransform
from torch.utils.data import Dataset
import numpy as np

from typing import Callable, Optional, Tuple, Union, Literal

from data.redwine import get_redwine, get_redwine_trafo
from data.protein import get_protein, get_protein_trafo
from data.california import get_california, get_california_trafo
from data.enb import get_enb, get_enb_trafo
from data.navalprop import get_navalpro, get_navalpro_trafo
from data.imagenet import get_ImageNet


def get_dataset(
        name: str, 
        path: str, 
        train: bool, 
        dtype: torch.dtype="float64"
    ) -> Dataset:
    
    if name.lower()=="cifar10":
        def transform(train: bool) -> Callable:
            mu  = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
            trafo = [ToTensor(), Normalize(mu, std), SetType(dtype)]
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
                Normalize(mu, std),
                SetType(dtype)
            ]
            
            return Compose(trafo)
        
        data = CIFAR10(path, train=train, transform=transform(train), 
            download=True)

        
    elif name.lower()=="mnist":
        def transform() -> Callable:
            return Compose([
                ToTensor(), 
                Normalize((0.5, ), (0.5,)), 
                SetType(dtype)]
            )
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)

    elif name.lower()=="mnist_c-brightness":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-canny_edges":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-dotted_line":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-fog":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-glass_blur":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-impulse_noise":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-motion_blur":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-rotate":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-scale":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-shear":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-shot_noise":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-spatter":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-stripe":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-translate":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_c-zigzag":
        data = get_mnist_corrupted(path, name, train, dtype)

    elif name.lower()=="mnist_small":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(14, antialias=True), 
                 Normalize((0.5, ), (0.5,)),
                 SetType(dtype)]
            )
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="mnist_small2class":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(7, antialias=True), 
                 Normalize((0.5, ), (0.5,)),
                 SetType(dtype)]
            )
        
        # extract zeroth (0) and first (1) class
        data = MNIST(path, train=train, transform=None, 
            download=True)
        idcs = torch.logical_or(data.targets == 0, data.targets == 1)
        X, Y = data.data.numpy()[idcs, ...], data.targets[idcs] 
        data = DatasetGenerator(X, Y, transform=transform())

    elif name.lower()=="fashionmnist":
        def transform() -> Callable:
            return Compose([
                ToTensor(), 
                Normalize((0.5, ), (0.5,)), 
                SetType(dtype)]
            )
        
        data = FashionMNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="imagenet10":
        # preprocessing according to 
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        # The loaded data is resized to 256**3 and cropped to 224**3
        def transform(train: bool=True) -> Callable:
            trafo = [ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            if train:
                trafo += [
                    RandomHorizontalFlip(p=0.5)
                ]

            return Compose(trafo)
        if train:
            X, Y, _, _ =  get_ImageNet(path, n_class=10)
        else:
            _, _, X, Y =  get_ImageNet(path, n_class=10)
        data = DatasetGenerator(X, Y, transform=transform(), is_classification=True)

    elif name.lower()=="redwine":
        def transform(train: bool, dtype: Literal["float32", "float64"]) -> Callable:
            return Compose(get_redwine_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_redwine()
        else:
            _, _, X, Y = get_redwine()
        data = DatasetGenerator(X, Y, transform=transform(train, dtype))

    elif name.lower()=="protein":
        def transform(train: bool, dtype: Literal["float32", "float64"]) -> Callable:
                    return Compose(get_protein_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_protein()
        else:
            _, _, X, Y = get_protein()
        data = DatasetGenerator(X, Y, transform=transform(train, dtype))

    elif name.lower()=="california":
        def transform(train: bool, dtype: Literal["float32", "float64"]) -> Callable:
            return Compose(get_california_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_california()
        else:
            _, _, X, Y = get_california()
        data = DatasetGenerator(X, Y, transform=transform(train, dtype))

    elif name.lower()=="enb":
        def transform(train: bool, dtype: Literal["float32", "float64"]) -> Callable:
            return Compose(get_enb_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_enb()
        else:
            _, _, X, Y = get_enb()
        data = DatasetGenerator(X, Y, transform=transform(train, dtype))

    elif name.lower()=="navalpro":
        def transform(train: bool, dtype: Literal["float32", "float64"]) -> Callable:
            return Compose(get_navalpro_trafo(train) + [SetType(dtype)])

        if train:
            X, Y, _, _ = get_navalpro(path)
        else:
            _, _, X, Y = get_navalpro(path)
        data = DatasetGenerator(X, Y, transform=transform(train, dtype))

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
            is_classification: bool=False 
        ):
        self.X         = X
        if len(Y.shape) > 1 or is_classification:
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
    

class SetType:
    def __init__(self, dtype: Literal["float32", "float64"]="float64"):
        self.dtype = dtype

    def __call__(self, tensor: Tensor):
        if self.dtype=="float32":
            return tensor.float()
        elif self.dtype=="float64":
            return tensor.double()
        else:
            raise NotImplementedError()


def get_mnist_corrupted(path: str, name: str, train: bool, dtype: str) -> Dataset:
    def transform() -> Callable:
        return Compose([
            ToTensor(), 
            Normalize((0.5, ), (0.5,)), 
            SetType(dtype)]
        )    
    
    name, corruption = name.split("-")
    file_path = os.path.join(path, name, corruption)
    if train:
        X = np.load(os.path.join(file_path, "train_images.npy"))
        Y = np.load(os.path.join(file_path, "train_labels.npy"))
    else:
        X = np.load(os.path.join(file_path, "test_images.npy"))
        Y = np.load(os.path.join(file_path, "test_labels.npy"))
    return DatasetGenerator(X, Y, transform=transform()) 