from torch import nn

from pred_model.mlp import MLP
from pred_model.resnet9 import ResNet9
from pred_model.cnn import cnn_small


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs) -> nn.Module:
    if name.startswith("mlp"):
        model = MLP(**kwargs)
    elif name == "resnet9":
        model = ResNet9(**kwargs)
    elif name == "cnn_small":
        model = cnn_small(**kwargs)
    else:
        raise NotImplementedError(name)
    
    return model