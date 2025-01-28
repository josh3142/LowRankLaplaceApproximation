#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_min=1
s_max=60 
s_n=58

# data
data=synthsinus
folder_name=synthsinus100_std01_test
data_std=0.1

# model parameters
model=mlp_small
n_hidden=1024
n_layer=1


# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag,lowrankoptimal-ggnit" 
psi_ref=ggnit
psi_approx=kron
jacobian_seed=0

# Plotting options
plot_evaluation_methods="[Psubset-swag_Psiggnit,Plowrank-diag_Psiggnit,Psubset-diag_Psiggnit,Plowrank-kron_Psiggnit,Psubset-magnitude_Psiggnit,Plowrankoptimal-ggnit_Psiggnit]"

# compute epistemic covariance and nll for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    projector.s.max=$s_max \
    projector.s.min=$s_min\
    projector.s.n=$s_n\
    projector.data_std=$data_std\
    projector.jacobian_seed=$jacobian_seed \
    seed=$seed

# compute epistemic covariance and nll for all projectors p
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.s.max=$s_max \
    projector.s.min=$s_min\
    projector.s.n=$s_n\
    projector.data_std=$data_std\
    projector.jacobian_seed=$jacobian_seed \
    seed=$seed

# compute metric for all projector p
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.s.max=$s_max \
    projector.s.min=$s_min\
    projector.s.n=$s_n\
    seed=$seed

# compute metric for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    device_torch=$device \
    projector=Sigma\
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    projector.s.max=$s_max \
    projector.s.min=$s_min\
    projector.s.n=$s_n\
    seed=$seed

# transform the python dictionaries into a dataframe
python make_dict_to_df.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    data=$data data.folder_name=$folder_name\
    pred_model=$model \
    plot.evaluation.methods=$plot_evaluation_methods \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \





