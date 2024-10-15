""" Wraps the submodel routine of `laplace`
into a suitable framework.
"""
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.utils
from tqdm import tqdm
import laplace
import laplace.utils
import laplace.utils.subnetmask

from projector.projector1d import (
    number_of_parameters_with_grad,
    where_parameters_with_grad,
)


class subset_indices():
    def __init__(self, model: nn.Module,
                 likelihood: Literal['regression','classification'],
                 train_loader: DataLoader,
                 method: Literal['diagonal', 'magnitude','swag','custom']='diagonal',
                 **kwargs) -> None:
        self.method = method
        self.number_of_parameters = number_of_parameters_with_grad(model)
        if method == 'diagonal':
            diag_laplace_model = laplace.Laplace(model=model,
                                            likelihood=likelihood,
                                            subset_of_weights='all',
                                            hessian_structure='diag')
            self.subnet_mask = laplace.utils.subnetmask.LargestVarianceDiagLaplaceSubnetMask(model=model,
                                                                                        n_params_subnet=self.number_of_parameters,
                                                                                        diag_laplace_model=diag_laplace_model)
        elif method == 'magnitude':
            self.subnet_mask = laplace.utils.subnetmask.LargestMagnitudeSubnetMask(model=model,
                                                                                   n_params_subnet=self.number_of_parameters)
        elif method == 'swag':
            self.subnet_mask = laplace.utils.LargestVarianceSWAGSubnetMask(model=model,
                                                                    n_params_subnet=self.number_of_parameters,
                                                                    likelihood=likelihood,
                                                                    **kwargs)
        elif method == 'custom':
            self.metric = kwargs['metric']
        else:
            raise NotImplementedError
            
        if method != 'custom':
            self.metric = self.subnet_mask.compute_param_scores(train_loader=train_loader)

        # only consider those parameters with requires_grad = True
        self.metric = self.metric[where_parameters_with_grad(model)]

    
    def __call__(self, s: int, sort: bool = False):
        idcs_select = torch.topk(self.metric, s).indices
        if sort: 
            idcs_select = idcs_select.sort()[0]
        return idcs_select


    def P(self, s, sort: bool=False) -> torch.Tensor:
        idx = self.__call__(s, sort=sort)
        return subP(number_of_parameters=self.number_of_parameters, idx=idx).to(self.metric.device)

def subP(number_of_parameters: int, idx: torch.Tensor) -> torch.Tensor:
    P = torch.zeros(number_of_parameters, len(idx))
    P[idx, torch.arange(len(idx))] = 1.0
    return P



    
