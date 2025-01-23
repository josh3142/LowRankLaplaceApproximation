#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_number=10

# data
data=mnist

# model parameters
model=cnn_small

# projector
projector=I

# swag
swag_lr=1.0e-2
swag_n_snapshots=40

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag,lowrankoptimal-loadfile" 
psi_ref=loadfile
psi_approx=kron
posterior_hessian_file=I60000.pt
projector_batch_size=50
v_batch_size=20


# Plotting options
plot_evaluation_methods="[Psubset-swag_Psiloadfile,Plowrank-diag_Psiloadfile,Psubset-diag_Psiloadfile,Plowrank-kron_Psiloadfile,Psubset-magnitude_Psiloadfile,Plowrankoptimal-loadfile_Psiloadfile]"


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

compute epistemic covariance and nll for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# compute epistemic covariance and nll for all projectors p
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# compute metric for all projector p
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# compute metric for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma \
    projector.sigma.method.p=null \
    projector.sigma.method.psi="$psi_ref,$psi_approx" \
    projector.posterior_hessian.load.name=$posterior_hessian_file \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# transform the python dictionaries into a dataframe
python make_dict_to_df.py -m \
    data=$data \
    pred_model=$model \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    plot.evaluation.methods=$plot_evaluation_methods \
    data=$data \
    pred_model=$model \
