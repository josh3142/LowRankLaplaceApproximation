import math

import pytest
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from linearized_model.approximation_metrics import NLL, collect_NLL

@pytest.fixture
def random_predictions(request) -> Tuple:
    seed = 0
    n_data, n_target = 20, 13
    regularizer = 1.0
    generator = torch.Generator().manual_seed(seed)
    half_variances = torch.randn((n_data, n_target, n_target), generator=generator)
    variances = torch.einsum('bij, bjk->bik', half_variances, half_variances.transpose(1,2))
    variances += regularizer * torch.eye(variances.size(1))[None,...]
    if is_classification := request.param:
        targets = torch.randint(low=0, high=n_target, size=(n_data,), generator=generator)
    else:
        targets = torch.randn((n_data, n_target), generator=generator)
    predictions = torch.randn((n_data, n_target), generator=generator)

    return predictions, targets, variances, is_classification


@pytest.mark.parametrize('random_predictions',  [True, False], indirect=True)
def test_NLL(random_predictions: tuple):
    def single_x_nll(prediction: torch.Tensor, cov: torch.Tensor,
                    target: torch.Tensor, is_classification: bool) -> torch.Tensor:
        if not is_classification:
           res = target-prediction
           quad_term =  -0.5 * res.T @ torch.linalg.inv(cov) @ res
           normalization = -0.5 * torch.logdet(2 * math.pi * cov)
           return -1.0 * (quad_term + normalization)
        else:
            logprobs = nn.LogSoftmax(dim=-1)(prediction/torch.sqrt(1+math.pi / 8 * torch.diagonal(cov)))
            return -1.0 * (logprobs[target])

    predictions, targets, variances, is_classification = random_predictions
    batch_size = predictions.size(0)
    theoretical_nll = 0
    for b in range(batch_size):
        theoretical_nll += single_x_nll(prediction=predictions[b],
                                        cov=variances[b],
                                        target=targets[b],
                                        is_classification=is_classification)
    computed_nll = NLL(predictions=predictions, variances=variances, targets=targets,
                     is_classification=is_classification, sum=True)
    assert torch.isclose(computed_nll, theoretical_nll).item()
    

@pytest.mark.parametrize('random_predictions',  [True, False], indirect=True)
def test_collect_NLL(random_predictions: tuple):
    predictions, targets, variances, is_classification = random_predictions
    number_of_batches = 3
    batch_size = predictions.size(0) // number_of_batches
    # taking predictions as dummy inputs suffices for this test
    dataset = TensorDataset(predictions, targets) 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    predictive_counter = 0
    def predictive(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal predictive_counter
        pred, var = predictions[predictive_counter:predictive_counter+X.size(0)], \
        variances[predictive_counter:predictive_counter+X.size(0)]
        predictive_counter += X.size(0)
        return pred, var

    theoretical_value = torch.mean(NLL(predictions=predictions, variances=variances,
                           targets=targets, is_classification=is_classification,
                           sum=False))
    
    computed_value = collect_NLL(predictive=predictive, dataloader=dataloader,
                                is_classification=is_classification, reduction='mean')
    
    assert torch.isclose(theoretical_value, computed_value)
    

    
