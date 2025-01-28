#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
n_batches=50
s_max=null

# data
data=california

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
projector_batch_size=10
jacobian_seed=0

# Plotting options
plot_evaluation_methods="[Psubset-swag_Psiloadfile,Plowrank-diag_Psiloadfile,Psubset-diag_Psiloadfile,Plowrank-kron_Psiloadfile,Psubset-magnitude_Psiloadfile,Plowrankoptimal-ggnit_Psiloadfile]"

# compute the Hessian or Fisher information matrix. (required for load file)
CUDA_VISIBLE_DEVICES=$cuda python get_hessian.py -m \
    seed=$seed \
    data=$data \
    pred_model=$model \
    projector=$projector \
    projector.batch_size=$projector_batch_size projector.n_batches=null\

# obtain and plot the spectrum
CUDA_VISIBLE_DEVICES=$cuda python plot_spectrum_H.py -m \
    seed=$seed \
    data=$data \
    pred_model=$model \
    projector=$projector

# compute epistemic covariance and nll for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    projector.jacobian_seed=$jacobian_seed \
    seed=$seed

# compute epistemic covariance and nll for all projectors p
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.jacobian_seed=$jacobian_seed \
    seed=$seed

# compute metric for all projector p
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    seed=$seed

# compute metric for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \
    projector=Sigma\
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.n_batches=$n_batches \
    projector.s.max=$s_max \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    seed=$seed

# transform the python dictionaries into a dataframe
python make_dict_to_df.py -m \
    data=$data \
    pred_model=$model \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    data=$data \
    pred_model=$model \
    plot.evaluation.methods=$plot_evaluation_methods \
    pred_model.param.n_hidden=$n_hidden pred_model.param.n_layer=$n_layer \






