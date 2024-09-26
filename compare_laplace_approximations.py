""" Compute the predictive covariance of different methods, use them to infer
a projection operator and compare the results with `update_performance_metrics`.
"""

import os
from typing import Optional, Union

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import laplace
from laplace import FullLaplace, KronLaplace
from tqdm import tqdm

from projector.projector1d import create_jacobian_data_iterator
from projector.fisher import get_V_iterator
from data.dataset import get_dataset
from pred_model.model import get_model
from linearized_model.low_rank_laplace import FullInvPsi, HalfInvPsi,\
    KronInvPsi, compute_Sigma, compute_optimal_P, compute_Sigma_P
from linearized_model.submodel import submodel_indices
from linearized_model.approximation_metrics import update_performance_metrics

# default parameters
default_batch_size = 100
default_number_of_batches = 10
default_s_step = 1
default_prior_precision = 1.0
layers_to_ignore = [nn.BatchNorm2d]

@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    # store all results in this dictionary
    results = {} 
    # store configuration dict
    results['cfg'] = cfg

    # load non-optional arguments
    dataset_name = cfg.data.name
    dtype_str = cfg.dtype
    dtype = getattr(torch, dtype_str)
    device = torch.device(cfg.device_torch)
    torch.set_default_dtype(dtype)
    print(f'Considering {dataset_name}')
    model_name = cfg.pred_model.name
    get_model_kwargs = dict(cfg.pred_model.param)
    ckpt_name = cfg.data.model.ckpt
    if is_classification := cfg.data.is_classification:
        likelihood = 'classification'
    else:
        likelihood = 'regression'

    # load optional arguments
    prior_precision = getattr(cfg, 'prior_precision', default_prior_precision)
    stored_hessians = getattr(cfg, 'stored_hessians', [])
    laplace_methods = getattr(cfg, 'laplace_methods', [])
    if len(stored_hessians) > 0:
        laplace_methods += ['file']
    subset_methods = getattr(cfg, 'subset_methods', [])
    standard_batch_size = getattr(cfg, 'standard_batch_size', default_batch_size)
    fit_batch_size = getattr(cfg, 'fit_batch_size', standard_batch_size)
    projector_batch_size = getattr(cfg, 'projector_batch_size', standard_batch_size)
    v_batch_size = getattr(cfg, 'v_batch_size',standard_batch_size)
    projector_number_of_batches = getattr(cfg, 'projector_number_of_batches',
                                      default_number_of_batches)
    v_n_batches = getattr(cfg, 'v_n_batches', None)
    chunk_size = getattr(cfg, 'chunk_size', None)
    postfix = getattr(cfg, 'postfix', '')
    seed_list = getattr(cfg, 'seed_list', [None,])
    results['seed_list'] = seed_list
    reference_method = getattr(cfg, 'reference_method',None)
        
    compute_reference_method = getattr(cfg, 'compute_reference_method', True)
    s_max = getattr(cfg, 's_max', None)
    s_step = getattr(cfg, 's_step', default_s_step)

    # Fix reference method 
    if reference_method is None:
        print('No reference method specified')
        if len(stored_hessians) > 0:
            print('Choosing first stored hessian')
            reference_method = 'file_' + stored_hessians[0]
        else:
            assert len(laplace_methods) > 0
            print('Choosing first laplace method')
            reference_method = laplace_methods[0]
    print(f'Taking {reference_method} as reference method')
    results['reference_method'] = reference_method

    # setting up kwargs for loading of model and data
    get_model_kwargs['name'] = model_name
    get_model_kwargs |= dict(cfg.data.param)
    results['get_model_kwargs'] = get_model_kwargs
    get_dataset_kwargs = dict(name=dataset_name,
                    path=cfg.data.path,
                    dtype=dtype_str)
    results['get_dataset_kwargs'] = get_dataset_kwargs

    # setting up paths
    results_path = os.path.join('results', dataset_name, model_name)
    results_name = f'comparison_laplace_approximations{postfix}.pt'
    results_filename = os.path.join(results_path, results_name)
    print(f'Using folder {results_path}')

    def ckpt_file(seed: Union[None, int], ckpt_name = ckpt_name) -> str:
        if seed is not None:
            return os.path.join(results_path, f'seed{seed}', 'ckpt', ckpt_name)
        else:
            return os.path.join(results_path,'ckpt', ckpt_name)

    def projector_file(seed: Union[None, int],
                       projector_name: str) -> str:
        # take first letters of filename as projector type
        if projector_name[0] == 'H':
            projector_type = 'H'
        else:
            assert projector_name[0] == 'I'
            if projector_name.startswith('Ihalf'):
                projector_type = 'Ihalf'
            else:
                projector_type ='I'
        if seed is not None:
            return os.path.join(results_path, f'seed{seed}', 'projector',
                                projector_type, projector_name)
        else:
            return os.path.join(results_path,'projector',
                                projector_type, projector_name)

    # load data
    train_data = get_dataset(**get_dataset_kwargs,
                    train=True)
    test_data = get_dataset(**get_dataset_kwargs,
                    train=False)
    # used for fitting laplacian
    fit_dataloader = DataLoader(dataset=train_data,
                                batch_size=fit_batch_size,
                                shuffle=True) 
    
    # load network
    model = get_model(**get_model_kwargs)
    model.eval()
    model.to(device)
    # switch off layers to ignore
    for module in model.modules():
        if type(module) in layers_to_ignore:
            for par in module.parameters():
                par.requires_grad = False


    #  The following objects create upon call an iterator over the jacobian
    create_train_proj_jac_it = lambda: create_jacobian_data_iterator(dataset=train_data,
                                                    model=model,
                                                    batch_size=projector_batch_size,
                                                    number_of_batches=projector_number_of_batches,
                                                    device=device,
                                                    dtype=dtype,
                                                    chunk_size=chunk_size)
    create_test_proj_jac_it = lambda: create_jacobian_data_iterator(dataset=test_data,
                                                    model=model,
                                                    batch_size=projector_batch_size,
                                                    number_of_batches=projector_number_of_batches,
                                                    device=device,
                                                    dtype=dtype,
                                                    chunk_size=chunk_size)

    # Compute s_max and s_List
    if s_max is None:
        number_of_parameters = sum([p.numel() for p in model.parameters()])
        test_out = model(next(iter(fit_dataloader))[0].to(device))
        if len(test_out.shape) == 1:
            n_out = 1
        else:
            n_out = test_out.size(-1)
        n_data = min(len(train_data), projector_number_of_batches * projector_batch_size)
        s_max = min(n_data * n_out, number_of_parameters)
    s_list = np.arange(1,s_max,step=s_step)
    results['s_list'] = s_list

    # Collect for each seed results and store them in `results[seed]`
    for seed in seed_list:
        results[seed] = {}
        print(f'Using seed {seed}\n............')

        # load checkpoint for seed
        ckpt_file_name = ckpt_file(seed=seed, ckpt_name=ckpt_name)
        results[seed]['ckpt_file_name'] = ckpt_file_name
        print(f'Loading model from {ckpt_file_name}')
        with open(ckpt_file_name, 'rb') as f:
            state_dict = torch.load(f, map_location=device)

        model.load_state_dict(state_dict=state_dict)

        # collecting posterior covariances for low rank methods
        print('>>>> Collecting posterior covariances')
        results[seed]['low_rank'] = {}
        if reference_method == 'Ihalf_it':

            V_it_dataloader = DataLoader(dataset=train_data,
                                        batch_size=v_batch_size,
                                        shuffle=True) 
            create_V_it = lambda: get_V_iterator(model=model, dl=V_it_dataloader,
                                                 is_classification=is_classification,
                                                 n_batches=v_n_batches,
                                                 chunk_size=chunk_size)
            # don't store InvPsi in results as it contains callables
            # which cannot be pickled
            results[seed]['low_rank'][reference_method] =  {
                'InvPsi': None}
            IPsi_ref = HalfInvPsi(V=create_V_it, prior_precision=prior_precision)
        else:
            assert reference_method.startswith('file') or reference_method in laplace_methods

        for method_name in laplace_methods:
            if method_name == 'file':
                for hessian_name in stored_hessians:
                    # load Hessian
                    hessian_file_name = projector_file(seed=seed,
                                                       projector_name=hessian_name)
                    print(f'Loading Hessian from file {hessian_file_name}')

                    with open(hessian_file_name, 'rb') as f:
                        H_file = torch.load(f, map_location=device)

                    if hessian_file_name.startswith('Ihalf'):
                        V = H_file['H'].to(dtype)
                        results[seed]['low_rank']['file_'+hessian_file_name] = {'InvPsi': HalfInvPsi(V=V, prior_precision=prior_precision)}
                    else:
                        H = H_file['H'].to(dtype)
                        # check if really square matrix
                        assert H.size(0) == H.size(1)
                        # construct inverse posterior variance
                        inv_Psi = H + prior_precision * torch.eye(H.size(0)).to(device)
                        results[seed]['low_rank']['file_'+hessian_name] = {'InvPsi': FullInvPsi(inv_Psi=inv_Psi)}
            else: 
                print(f'Computing {method_name}')
                if method_name in ['ggn']:
                    la = laplace.Laplace(model=model, hessian_structure='full', likelihood=likelihood,
                                        subset_of_weights='all', prior_precision=prior_precision)
                    la.fit(fit_dataloader)
                    assert type(la) is FullLaplace
                    results[seed]['low_rank'][method_name] = {
                        'InvPsi': FullInvPsi(inv_Psi=la.posterior_precision.to(dtype))}
                elif method_name in ['kron']:
                    la = laplace.Laplace(model=model, hessian_structure='kron', likelihood=likelihood,
                                        subset_of_weights='all', prior_precision=prior_precision)
                    la.fit(fit_dataloader)
                    assert type(la) is KronLaplace
                    results[seed]['low_rank'][method_name] = {'InvPsi': KronInvPsi(inv_Psi=la)}
                else:
                    raise NotImplementedError

        # Collect indices for subset methods
        print('>>>>> Collecting subset methods')
        results[seed]['subset'] = {}
        for method in subset_methods:
            print(f'Computing {method}')
            if method == 'swag':
                subset_kwargs = dict(cfg.data.swag_kwargs)
            else:
                subset_kwargs = {}
            results[seed]['subset'][method] = {'Indices': submodel_indices(model=model,
                                                                    likelihood=likelihood,
                                                                    train_loader=fit_dataloader,
                                                                    method=method,
                                                                    **subset_kwargs)}


        # collect reference and baseline metrics
        results[seed]['baseline'] = {}
        # do only store reference InvPsi if it isn't 
        # build using a V iterator (cf. above)
        if reference_method != 'Ihalf_it':
            IPsi_ref = results[seed]['low_rank'][reference_method]['InvPsi']
            results[seed]['baseline']['InvPsi']  = IPsi_ref
        else:
            results[seed]['baseline']['InvPsi']  = None

        # obtain best optimal approximation using test data
        # and the reference method
        if compute_reference_method:
            print('>>>>> Computing baseline results for reference method')
            Sigma_ref = compute_Sigma(IPsi=IPsi_ref, J_X=create_test_proj_jac_it)
            U, Lamb, _ = torch.linalg.svd(Sigma_ref)
            P = compute_optimal_P(IPsi=IPsi_ref,
                                J_X=create_test_proj_jac_it,
                                U=U)
            results[seed]['baseline']['metrics'] = {}
            create_Sigma_P_s_it = compute_Sigma_P(P=P, IPsi=IPsi_ref,
                                            J_X=create_test_proj_jac_it,
                                            s_iterable=s_list)
            assert callable(create_Sigma_P_s_it)
            for Sigma_P_s in create_Sigma_P_s_it():
                update_performance_metrics(
                        metrics_dict=results[seed]['baseline']['metrics'],
                        Sigma_approx=Sigma_P_s, Sigma=Sigma_ref)
        else:
            Sigma_ref = None
            results[seed]['baseline']['metrics'] = None
        results[seed]['baseline']['Sigma'] = Sigma_ref
            
        # collect metrics for low_rank methods
        print('>>>>>> Evaluating low rank methods\n............')
        for method in results[seed]['low_rank'].keys():
            if not compute_reference_method and method == reference_method:
                continue
            print(f'> Computing results for {method}')
            results[seed]['low_rank'][method]['metrics'] = {}
            if method != 'Ihalf_it':
                IPsi = results[seed]['low_rank'][method]['InvPsi']
            elif reference_method == 'Ihalf_it':
                IPsi = IPsi_ref
            else:
                raise NotImplementedError('Ihalf_it is only available as reference method')
            print('Performing SVD on predictive covariance on train data')
            U, Lamb = IPsi.Sigma_svd(create_train_proj_jac_it)
            print('Computing P for train data')
            P = compute_optimal_P(IPsi=IPsi, J_X=create_train_proj_jac_it, U=U)
            print('Computing performance of P on test data')
            create_Sigma_P_s_it = compute_Sigma_P(P=P, IPsi=IPsi_ref,
                                            J_X=create_test_proj_jac_it,
                                            s_iterable=s_list)
            assert callable(create_Sigma_P_s_it) 
            for s, Sigma_P_s in tqdm(zip(s_list, create_Sigma_P_s_it()),
                                     desc='Running through s'):
                update_performance_metrics(
                    metrics_dict=results[seed]['low_rank'][method]['metrics'],
                    Sigma_approx=Sigma_P_s, Sigma=Sigma_ref)
        
        # collect metrics for subet methods
        print('>>>>>> Evaluating subset methods\n............')
        for method in results[seed]['subset'].keys():
            print(f'> Computing results for {method}')
            Ind = results[seed]['subset'][method]['Indices']
            print('Obtaining P')
            P = Ind.P(s_max).to(device)
            print('Computing performance of P on test data')
            results[seed]['subset'][method]['metrics'] = {}
            create_Sigma_P_s_it = compute_Sigma_P(P=P, IPsi=IPsi_ref,
                                            J_X=create_test_proj_jac_it,
                                            s_iterable=s_list)
            assert callable(create_Sigma_P_s_it)
            for Sigma_P_s in create_Sigma_P_s_it():
                update_performance_metrics(
                    metrics_dict=results[seed]['subset'][method]['metrics'],
                    Sigma_approx=Sigma_P_s, Sigma=Sigma_ref)
        
        # save results after each seed computation
        print(f'Done with seed {seed}! Saving results under {results_filename}')
        with open(results_filename, 'wb') as f:
            torch.save(results, f)
        
if __name__=="__main__":
    run_main()