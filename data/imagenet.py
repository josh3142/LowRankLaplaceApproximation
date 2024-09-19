import os

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms import InterpolationMode

from PIL import Image
import numpy as np

import json 

from typing import Tuple, Callable, Literal, Optional, List, Dict

import pdb

class ImageNetKaggle(Dataset):
    """
    ImageNet Dataset. Code taken from 
    """
    def __init__(self, root: str,
                 split: Literal['train', 'val'],
                 transform: Optional[Callable]):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


class ImageNet100Kaggle(Dataset):
    """
    https://www.kaggle.com/datasets/ambityga/imagenet100
    """
    def __init__(self, root: str,
                 split: Literal['train', 'val'],
                 transform: Optional[Callable]):
        self.transform = transform
        self.syn_to_class = self.get_syn_to_class(root, "imagenet_class_index.json")
        self.val_to_syn = self.get_val_to_syn(root, "ILSVRC2012_val_labels.json")
        self.samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        self.subset_class = self.get_subset(".", "imagenet10_selection.json")

        self.samples, self.targets = self.get_samples_targets(split, subset = True)

    def get_samples_targets(self, split: str, subset: bool = True) -> Tuple[List]:
        samples, targets = [], []
        for entry in os.listdir(self.samples_dir):
            syn_id = entry if split == "train" else self.val_to_syn[entry]

            if subset:
                if syn_id not in self.subset_class:
                    continue

            if split == "train":
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(self.samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    samples.append(sample_path)
                    targets.append(target)
            elif split == "val":
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(self.samples_dir, entry)
                samples.append(sample_path)
                targets.append(target)
        
        return samples, targets
    

    def get_syn_to_class(self, root: str, file_name: str) -> Dict:
        syn_to_class = {}
        with open(os.path.join(root, file_name), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)
        return syn_to_class


    def get_val_to_syn(self, root: str, file_name: str) -> Dict:
        with open(os.path.join(root, file_name), "rb") as f:
            val_to_syn = json.load(f)
        return val_to_syn


    def get_subset(self, root: str, file_name: str) -> List[str]:
        with open(os.path.join(root, file_name), "rb") as f:
            class_selected = json.load(f)
        return list(class_selected.keys())


    def __len__(self) -> int:
            return len(self.samples)


    def __getitem__(self, idx: int) -> Tuple:
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            x1 = np.copy(np.asarray(x))
            return x1, self.targets[idx]
    

def load_ImageNet(path: str, n_class: int, train: bool = True,
    transform: Optional[Callable] = None) -> Dataset:
    if train:
        imagenet_split = 'train'
    else:
        imagenet_split = 'val'

    if n_class == 100:
        return ImageNet100Kaggle(split  = imagenet_split,
                    root      = path,
                    transform = transform)
    elif n_class == 1000:
        return ImageNetKaggle(split  = imagenet_split,
                    root      = path,
                    transform = transform)


def generate_ImageNet(path: str, n_class: int, train: bool, 
    transform: Optional[Callable]
    ) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    
    data = load_ImageNet(path, n_class, train = train, transform = transform)

    X, Y = [], []
    for idx in range(len(data)):
        x, y = data.__getitem__(idx)
        X.append(x)
        Y.append(y)
    
    X = np.stack(X, axis = 0)
    Y = np.array(relabel(Y))
    assert len(Y) == X.shape[0] == len(data)

    return X, Y


def relabel(Y: List) -> List:
    """
    Rename the labels such that they are in the range from [0, n_class - 1]
    """
    y_unique   = sorted(set(Y))
    old_to_new = {}
    for y_new, y_old in enumerate(y_unique):
        old_to_new[y_old] = y_new
    Y_new = [old_to_new[y_old] for y_old in Y]

    assert len(Y_new) == len(Y)
    return Y_new        


def store_dataset(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, 
    Y_te: np.ndarray, file_name: str) -> None:
    """
    Saves the dataset
    """
    np.savez(file_name, X_tr = X_tr, Y_tr = Y_tr, X_te = X_te, Y_te = Y_te)


def load_dataset(file_name: str) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    file = np.load(file_name)
    X_tr = file["X_tr"]
    Y_tr = torch.tensor(file["Y_tr"])
    X_te = file["X_te"]
    Y_te = torch.tensor(file["Y_te"])
    return X_tr, Y_tr, X_te, Y_te


def get_ImageNet(path: str, n_class: int) -> Tuple[Tensor, np.ndarray, Tensor, np.ndarray]:
    
    file_name = f"imagenet{n_class}_data.npz"
    file_name = os.path.join(path, file_name)
    X_tr, Y_tr, X_te, Y_te = load_dataset(file_name)
    return X_tr, Y_tr, X_te, Y_te


def get_ImageNet_trafo(train: bool=True) -> Callable:
    trafo = [ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if train:
        trafo += [
            RandomHorizontalFlip(p = 0.5)
        ]

    return Compose(trafo)

if __name__ == "__main__":
    n_class   = 1000
    file_name = f"imagenet{n_class}_data.npz"
    path = "../../../SharedData/AI/datasets/ImageNet/"

    trafo = Compose(
        [Resize(256, interpolation= InterpolationMode.BICUBIC),
        CenterCrop(224)])
    X_tr, Y_tr = generate_ImageNet(path, n_class, train = True, transform = trafo)
    X_te, Y_te = generate_ImageNet(path, n_class, train = False, transform = trafo)
    
    assert X_tr.shape[0] == Y_tr.shape[0]
    assert X_te.shape[0] == Y_te.shape[0]
    assert X_tr.shape[1:] == X_te.shape[1:]
    store_dataset(X_tr, Y_tr, X_te, Y_te, file_name)
    X1_tr, Y1_tr, X1_te, Y1_te = load_dataset(file_name)
    assert np.array_equal(X_tr, X1_tr)
    assert np.array_equal(Y_tr, Y1_tr)
    assert np.array_equal(X_te, X1_te)
    assert np.array_equal(Y_te, Y1_te) 
