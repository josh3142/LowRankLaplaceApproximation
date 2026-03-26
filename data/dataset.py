import os

import torch
from torch import nn, Tensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.models.resnet import ResNet18_Weights
from torchvision.transforms import (Compose, Normalize, ToTensor, Resize,
    RandomHorizontalFlip, RandomGrayscale, RandomApply, RandomResizedCrop)
from torchvision.transforms import ElasticTransform
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

from typing import Callable, Optional, Tuple, Union, Literal

from data.redwine import get_redwine, get_redwine_trafo
from data.protein import get_protein, get_protein_trafo
from data.california import get_california, get_california_trafo
from data.enb import get_enb, get_enb_trafo
from data.navalprop import get_navalpro, get_navalpro_trafo
from data.imagenet import get_ImageNet_from_npz_file, ImageNetKaggle

def get_dataset(
        name: str, 
        path: str, 
        train: bool, 
        dtype: Literal["float32", "float64"]="float64"
    ) -> Dataset:
    
    if name.lower()=="cifar10":
        def transform() -> Callable: # type: ignore
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
        
        data = CIFAR10(path, train=train, transform=transform(), 
            download=True)
        
    elif name.startswith("cifar10_c"):
        if train:
            data = get_dataset(name="cifar10", path=path, train=True, dtype=dtype)
        else:
            data = get_cifar10_corrupted(path, name, train, dtype)
        
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
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

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
        def transform() -> Callable:
            trafo = [ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            if train:
                trafo += [
                    RandomHorizontalFlip(p=0.5)
                ]

            return Compose(trafo)
        if train:
            X, Y, _, _ =  get_ImageNet_from_npz_file(path, n_class=10)
        else:
            _, _, X, Y =  get_ImageNet_from_npz_file(path, n_class=10)
        data = DatasetGenerator(X, Y, transform=transform(), 
                is_classification=True, dtype=dtype)

    elif name.lower() == "imagenet1000":
        split = 'train' if train else 'val'
        def transform() -> Callable:
            trafo = [
               ResNet18_Weights.DEFAULT.transforms(),
               SetType(dtype),
            ]
            if train:
                trafo += [
                    RandomHorizontalFlip(p=0.5)
                ]

            return Compose(trafo)
        data = ImageNetKaggle(
            root=path,
            split=split,
            transform=transform(),
        )

    elif name.lower()=="redwine":
        def transform() -> Callable:
            return Compose(get_redwine_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_redwine()
        else:
            _, _, X, Y = get_redwine()
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

    elif name.lower()=="protein":
        def transform() -> Callable:
                    return Compose(get_protein_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_protein()
        else:
            _, _, X, Y = get_protein()
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

    elif name.lower()=="california":
        def transform() -> Callable:
            return Compose(get_california_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_california()
        else:
            _, _, X, Y = get_california()
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

    elif name.lower()=="enb":
        def transform() -> Callable:
            return Compose(get_enb_trafo(train) + [SetType(dtype)])
        
        if train:
            X, Y, _, _ = get_enb()
        else:
            _, _, X, Y = get_enb()
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

    elif name.lower()=="navalpro":
        def transform() -> Tuple[Callable, Callable]:
            return (
                Compose(get_navalpro_trafo(train)[0] + [SetType(dtype)]),
                Compose(get_navalpro_trafo(train)[1] + [SetType(dtype)])
            )

        if train:
            X, Y, _, _ = get_navalpro(path)
        else:
            _, _, X, Y = get_navalpro(path)
        data = DatasetGenerator(X, Y, transform=transform(), dtype=dtype)

    elif name.lower()=="synthsinus":
        data_loaded = np.load(path)
        if train:
            X, Y = data_loaded["X_tr"], data_loaded["Y_tr"]
        else:
            X, Y = data_loaded["X_te"], data_loaded["Y_te"]
        data = DatasetGenerator(X, Y, transform=None, is_classification=False) 

    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
        
    return data


class DatasetGenerator(Dataset):
    """ Create a dataset """
    def __init__(
            self, 
            X: Union[Tensor, np.ndarray], 
            Y: Union[Tensor, np.ndarray], 
            transform: Optional[
                Union[Callable, Tuple[Callable, Callable]]
            ]=None,
            is_classification: bool=False,
            dtype: Literal["float32", "float64"]="float64",
            change_X_dtype: bool=True,
        ):
        """
        Args:
            X: Independent variable
            Y: Dependent variable
            transform: Torch data transformation
            is_classification: Is it a classification or regression task?
            dtype: Data type of `X`. Data type of `Y` is changed only for 
                regression tasks (classification task should be of integer type) 
        """
        self.X = X
        if len(Y.shape) > 1 or is_classification:
            self.Y         = Y
        else:
            self.Y = Y[...,None] # add target dimension
        if not is_classification:
            self.Y = self.change_dtype(self.Y, dtype)
        self.transform = transform
        self.dtype = dtype

    @staticmethod
    def change_dtype(
            data: Union[Tensor, np.ndarray], 
            dtype: Literal["float32", "float64"]
        ) -> Union[Tensor, np.ndarray]:
        if isinstance(data, np.ndarray):
            datatype = np.float64 if dtype == "float64" else np.float32
            return data.astype(datatype)
        elif isinstance(data, Tensor):
            datatype = torch.float64 if dtype == "float64" else torch.float32
            return data.to(datatype)
        else:
            raise TypeError("Input should be either NumPy array or Pytorch Tensor.")

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.X[idx], self.Y[idx]

        if self.transform is not None:
            if type(self.transform) is tuple:
                x = self.transform[0](x)
                y = self.transform[1](y)
            else:
                x = self.transform(x)
        x = self.change_dtype(x, self.dtype)
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
            SetType(dtype),
        ])

    name, corruption = name.split("-")
    file_path = os.path.join(path, name, corruption)
    if train:
        X = np.load(os.path.join(file_path, "train_images.npy"))
        Y = np.load(
            os.path.join(file_path, "train_labels.npy")
            ).astype(np.int64)
    else:
        X = np.load(os.path.join(file_path, "test_images.npy"))
        Y = np.load(
            os.path.join(file_path, "test_labels.npy")
            ).astype(np.int64)
    return DatasetGenerator(
        X,
        Y,
        transform=transform(),
        dtype=dtype,
        is_classification=True,
    )

def get_cifar10_corrupted(path: str, name: str, train: bool, dtype: str,
                          split_seed: int = 0) -> Dataset:
    assert not train, "CIFAR10 corrupted only available as test data"
    def transform() -> Callable: # type: ignore
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

    name, corruption = name.split("-")
    file_path = os.path.join(path, name)
    X = np.load(
        os.path.join(file_path, f"{corruption}.npy")
    )
    Y = np.load(
        os.path.join(file_path, f"labels.npy")
    ).astype(np.int64)
    return DatasetGenerator(
        X,
        Y,
        transform=transform(),
        dtype=dtype,
        is_classification=True,
        change_X_dtype=False,
    )