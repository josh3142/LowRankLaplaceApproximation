#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_max=753

# data
data=redwine
data_folder_name=redwine

# model parameters
model=mlp_small
n_hidden=128
n_layer=2

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag,lowrankoptimal-ggnit"
psi_ref=ggnit
psi_approx=kron

# Plotting options
plot_evaluation_methods="[Psubset-swag_Psiggnit,Plowrank-diag_Psiggnit,Psubset-diag_Psiggnit,Plowrank-kron_Psiggnit,Psubset-magnitude_Psiggnit,Plowrankoptimal-ggnit_Psiggnit,PNone_Psiggnit]"

# compute projectors for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_projector.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute epistemic covariance for full model
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.s.max=$s_max \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    seed=$seed

# compute epistemic covariance for all submodels
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute covariance based metrics for full model
CUDA_VISIBLE_DEVICES=$cuda python compute_covariance_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    seed=$seed

# compute covariance based metrics for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_covariance_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute predictive distribution based metrics for full model
CUDA_VISIBLE_DEVICES=$cuda python compute_predictive_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    seed=$seed

# compute predictive distribution based metrics for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_predictive_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed


# transform dictionaries into a dataframes
python make_dict_to_df.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    plot.evaluation.methods=$plot_evaluation_methods \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
