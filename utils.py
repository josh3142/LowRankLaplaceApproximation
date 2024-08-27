# Note that these code snippets rely on the property that the order of
# parameters() is not supposed to change for different calls. This is be 
# expected since the parameters in the dictionary are named according to the 
# generator model.name_parameters() which has the correct order

# https://discuss.pytorch.org/t/does-model-parameters-return-the-parameters-in-topologically-sorted-order/81735/2
# https://github.com/pytorch/pytorch/blob/17d743ff0452b9cf12c7fcab7314079b83012bd4/torch/nn/modules/module.py#L257

from typing import Generator, List, Dict, Callable, Optional, Iterable,\
    Literal, Union

import torch
from torch import Tensor, nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call

import random
from tqdm import tqdm
import numpy as np



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



def pred_cov_eigenvalues(pred_cov: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.svdvals(pred_cov)
    return L


def iterator_wise_quadratic_form(quadratic_form: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                              create_iterator: Callable[[], Iterable[torch.Tensor]],
                              number_of_batches: Optional[int]=None) -> torch.Tensor:

    """ Iterative computation of the matrix that arises from the
    blocks`(quadrative_form(X_i,X_j))_ij` where the X_i,X_j are obtained by
    running through the iterator obtained by the call `create_iterator()`.

    Args: 
        quadratic_form: Should return a torch.tensor matrix
        create_iterator: When called with no arguments should return an iterator
            yielding torch.tensor
        number_of_batches (optional): When specified determines the number of
            tensors taken from `create_iterator()`. When None, the iterator is
            called up to `StopIteration`.

    Returns:
        A torch.tensor matrix
    """
    rows = []
    for i, X_i in tqdm(enumerate(create_iterator())):
        columns = []
        if number_of_batches is not None:
            if i >= number_of_batches:
                break
        for j, X_j in enumerate(create_iterator()):
            if number_of_batches is not None:
                if j >= number_of_batches:
                    break
            columns.append(quadratic_form(X_i, X_j))
        rows.append(torch.concat(columns, dim=1))
    return torch.concat(rows, dim=0)


def iterator_wise_matmul(create_a_iterator: Callable[[], Iterable],
                            b: torch.Tensor,
                            number_of_batches: Optional[int]=None,
                            transpose_a: bool = False,
                            iteration_dim: Literal[0,1]=0) -> torch.Tensor:
    """Computes `torch.matmul(a, b)` with the a yielded by `create_a_iterator()`
    and concatenates the resulting tensors along the first dimension (if
    `iteration_dim=0`) or adds them (if `iteration_dim=1`), i.e., `iteratin_dim`
    denotes the dimension along which `create_a_iterator` samples a. 
    *Note*: `a` is expected to be 2-dimensional, `b` can either be 1- or
    *2-dimensional.
    If `transpose_a` is set to True, `torch.matmul(a.T,b) is computed instead in
    each iteration.

    Args:
        a_it (_type_): A function that returns an iterator with no argument.
        b (torch.Tensor): 1- or 2-dimensional torch.Tensor
        transpose_a (bool): Compute a.T @ b instead.

    Returns:
        torch.Tensor: The concatenation of `a @ b` (or `a.T @ b` if
        `transpose_a` is True) along the first dimension for a drawn from
        `create_a_iterator()`.
    """
    assert b.ndim in [1,2]
    if iteration_dim==0:
        a_times_b_collection = []
        for i, a_i in enumerate(create_a_iterator()):
            if number_of_batches is not None:
                if i >= number_of_batches:
                    break
            assert a_i.ndim == 2
            if not transpose_a:
                a_times_b_collection.append(torch.matmul(a_i,b))
            else:
                a_times_b_collection.append(torch.matmul(a_i.T,b))
        return torch.concat(a_times_b_collection, dim=0)
    elif iteration_dim==1:
        a_times_b = 0.0
        batch_start_index = 0
        for i, a_i in enumerate(create_a_iterator()):
            if number_of_batches is not None:
                if i >= number_of_batches:
                    break
            assert a_i.ndim == 2
            if transpose_a:
                a_i = a_i.T
            batch_end_index = batch_start_index + a_i.size(-1)
            a_times_b += torch.matmul(a_i,b[batch_start_index:batch_end_index])
            batch_start_index = batch_end_index
        return a_times_b
        
    else:
        raise ValueError



def flatten_batch_and_target_dimension(J_X: Union[torch.Tensor,
                                                  Callable[[], Iterable]]
                                                  ) -> Union[torch.Tensor,
                                                             Callable[[],Iterable]]:
    if type(J_X) is torch.Tensor:
        assert J_X.dim() in [2,3], "can only be called for 2D or 3D Tensors."
        return J_X.view(-1, J_X.size(-1))
    else:
        def create_flattened_iterator() -> Iterable:
            for j in J_X():
                assert type(j) is torch.Tensor # to avoid infinite recursion
                yield flatten_batch_and_target_dimension(J_X=j)
        return create_flattened_iterator

