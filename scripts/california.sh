#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_max=null

# data
data=california
data_folder_name=california

# projector
projector=I

# model parameters
model=mlp_small
n_hidden=128
n_layer=2

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag,lowrankoptimal-ggnit" 
psi_ref=loadfile
posterior_hessian_file=I16512.pt
psi_approx=kron
projector_batch_size=100
n_batches=50

# Plotting options
plot_evaluation_methods="[Psubset-swag_Psiloadfile,Plowrank-diag_Psiloadfile,Psubset-diag_Psiloadfile,Plowrank-kron_Psiloadfile,Psubset-magnitude_Psiloadfile,Plowrankoptimal-ggnit_Psiloadfile,PNone_Psiloadfile]"


# compute the Hessian (required for load file)
CUDA_VISIBLE_DEVICES=$cuda python get_hessian.py -m \
    seed=$seed \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    projector=$projector \
    projector.batch_size=$projector_batch_size projector.n_batches=null\

# compute projectors for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_projector.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.batch_size=$projector_batch_size \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute epistemic covariance for full model
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=null \
    projector.batch_size=$projector_batch_size \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    seed=$seed

# compute epistemic covariance for all submodels
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.batch_size=$projector_batch_size \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute covariance based metrics for full model
CUDA_VISIBLE_DEVICES=$cuda python compute_covariance_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
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
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute predictive distribution based metrics for full model
CUDA_VISIBLE_DEVICES=$cuda python compute_predictive_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
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
    projector.posterior_hessian.load.name=$posterior_hessian_file \
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
