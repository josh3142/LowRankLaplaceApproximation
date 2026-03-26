#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_number=10
s_max=100

# data
data=imagenet10
data_folder_name=imagenet10

# model parameters
model=resnet18

# swag
swag_lr=1.0e-2
swag_n_snapshots=40

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag" 
psi_ref=ggnit
projector_batch_size=5
n_batches=100
v_batch_size=5


# compute projectors for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_projector.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    projector=Sigma \
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref  \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    projector.n_batches=$n_batches \
    seed=$seed

# compute epistemic covariance for all submodels
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    projector=Sigma\
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    projector.n_batches=$n_batches \
    seed=$seed

# compute covariance based metrics for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_covariance_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    projector=Sigma\
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    projector.n_batches=$n_batches \
    seed=$seed

# compute predictive distribution based metrics for all submodels
CUDA_VISIBLE_DEVICES=$cuda python compute_predictive_metrics.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \
    projector=Sigma\
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.v.batch_size=$v_batch_size \
    projector.n_batches=$n_batches \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# transform dictionaries into a dataframes
python make_dict_to_df.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    data=$data \
    data.folder_name=$data_folder_name \
    pred_model=$model \


