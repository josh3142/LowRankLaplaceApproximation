#!/bin/bash

# general
cuda=0
seed="1,2,3,4,5"
s_number=10
s_max=1000


# data
data=cifar10

# model parameters
model=resnet9

# swag
swag_lr=1.0e-2
swag_n_snapshots=40

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag" 
psi_ref=ggnit
projector_batch_size=10
v_batch_size=10

# compute epistemic covariance and nll for all projectors p
CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma\
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# compute metric for the full Laplace approximation
CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
    data=$data \
    pred_model=$model \
    projector=Sigma\
    projector.s.max=$s_max \
    projector.sigma.method.p=$p \
    projector.sigma.method.psi=$psi_ref \
    projector.v.batch_size=$v_batch_size \
    projector.batch_size=$projector_batch_size \
    seed=$seed

# transform the python dictionaries into a dataframe
python make_dict_to_df.py -m \
    data=$data \
    pred_model=$model \

# create the plots based on the dataframes
python plot_evaluation.py -m \
    data=$data \
    pred_model=$model \


