# Note that these code snippets rely on the property that the order of
# parameters() is not supposed to change for different calls. This is be 
# expected since the parameters in the dictionary are named according to the 
# generator model.name_parameters() which has the correct order

# https://discuss.pytorch.org/t/does-model-parameters-return-the-parameters-in-topologically-sorted-order/81735/2
# https://github.com/pytorch/pytorch/blob/17d743ff0452b9cf12c7fcab7314079b83012bd4/torch/nn/modules/module.py#L257

import torch
from torch import Tensor, nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call

import random
import numpy as np

from typing import Generator, List, Dict, Callable


def param_to_vec(params: Generator) -> Tensor:
    """" 
    Maps the parameters of a model to a vector. 
    Similar to https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    """
    vec = []
    for p in params:
        vec.append(p.view(-1))
    return torch.cat(vec)


def param_to_dict(named_params: Generator) -> Tensor:
    """" 
    Maps the parameters of a model to a ordered dictionary. 
    Similar to https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    """
    dict = OrderedDict()
    for name, p in named_params:
        dict[name] = p
    return dict

def vec_to_list(vec: Tensor, params: Generator) -> List:
    """
    Maps a vector to list of parameters of the shape of the model parameters
    `params`
    Similar to https://pytorch.org/docs/stable/generated/torch.nn.utils.vector_to_parameters.html
    """
    pointer = 0
    ps = []
    for p_dummy in params:
        n_p = p_dummy.numel()
        ps.append(vec[pointer: pointer + n_p].view_as(p_dummy))
        pointer += n_p 
    return ps


def vec_to_dict(vec: Tensor, params_named: Generator) -> Dict:
    """
    Maps a vector to dictionary of {name: parameters} of the state dictionary 
    of the model from which `params_named` is given.
    Similar to Similar to https://pytorch.org/docs/stable/generated/torch.nn.utils.vector_to_parameters.html
    """
    pointer = 0
    ps = OrderedDict()
    for name, p_dummy in params_named:
        n_p = p_dummy.numel()
        ps[name] = vec[pointer: pointer + n_p].view_as(p_dummy)
        pointer += n_p 
    assert pointer == len(vec), f"The vector has {len(vec)} components " + \
        f"but the generator `params_named` has {pointer}"

    return ps



def get_grads(model: nn.Module, X: Tensor) -> Tensor:
    """ 
    Computes the gradients of the input wrt to the model `model` and stacks
    them into a tensor.
    """
    dataset = TensorDataset(X)

    grads = []
    for x in DataLoader(dataset, batch_size=1, shuffle=False): 
        y_hat = model(x[0])

        model.zero_grad()
        y_hat.backward()
        grad = get_grad(model.parameters())
        grads.append(param_to_vec(grad))

    return torch.stack(grads, dim=0)


class MakeUnitVec(nn.Module):
    """ 
    Normalization class. Can be used to normalize a layer to have unit norm. 
    """
    def forward(self, v: Tensor) -> Tensor:
        return v / torch.sqrt(torch.sum(v**2))


def get_grad(params: Generator) -> Tensor:
    grad = [param.grad.detach().clone() for param in params]
    return grad



def get_projection_rank1_matrix(v: Tensor, eigenvalues: List, eigenvectors: List
                             ) -> Tensor:
    """
    Calculates a inner product of a sum of rank 1 matrices defined by 
    `eigenvectors` times `v`. Each rank 1 matrix is scaled by the scalar
    `eigenvalues`.
    Args:
        v (Tensor): Projection vector
        eigenvalues (List): List of eigenvalues 
        eigenvectors (List): List of eigenvectors     
    """
    assert len(eigenvalues) == len(eigenvectors)
    return sum([eigenvalues[k] * torch.einsum("i, i", v, eigenvectors[k])**2 
                for k in range(len(eigenvalues))])


def get_softmax_model_fun(
        param_vec: Tensor, 
        param_gen: Generator, 
        model: nn.Module, 
        X: Tensor, 
        fun: Callable=torch.log
    ) -> Tensor:
    """
    Concatenates the model function m with the softmax function s and a 
    different function f s.t. f(s(m(x))).
    Args:
        param_vec: Parameters with respect the model is differentiated
        param_gen: Generator of the parameters that has the correct order of 
            the flattened `param_vec` 
    """
    return fun(nn.functional.softmax(functional_call(
        model, vec_to_dict(param_vec, param_gen), X
    ), dim=-1))


def get_model_fun(
        param_vec: Tensor, 
        param_gen: Generator, 
        model: nn.Module, 
        X: Tensor, 
        fun: Callable=lambda x: x
    ) -> Tensor:
    """
    Concatenates the model function m with a function f s.t. f(m(x)).
    Args:
        param_vec: Parameters with respect the model is differentiated
        param_gen: Generator of the parameters that has the correct order of 
            the flattened `param_vec` 
    """
    return fun(functional_call(model, vec_to_dict(param_vec, param_gen), X))


def get_max_class_softmax_model_fun(
        param_vec: Tensor, 
        param_gen: Generator, 
        model: nn.Module, 
        X: Tensor, 
        Y: Tensor,  
        fun: Callable=torch.log
    ):
    """
    Concatenates the model function m with the softmax function s and a 
    different function f s.t. f(s(m(x))).
    Args:
        param_vec: Parameters with respect the model is differentiated
        param_gen: Generator of the parameters that has the correct order of 
            the flattened `param_vec` 
    """
    return fun(nn.functional.softmax(functional_call(
        model, vec_to_dict(param_vec, param_gen), X
    ), dim=-1))[torch.arange(len(X), device=Y.device), Y]


def make_deterministic(seed) -> None:
    random.seed(seed)   	    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)