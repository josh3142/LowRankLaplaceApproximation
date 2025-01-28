#!/bin/bash
# general
cuda=0
seed="1,2,3,4,5" 
s_number=3
s_min=100
s_max=900


# corruption
use_corrupt=True
corruptions="brightness canny_edges dotted_line 
    fog glass_blur impulse_noise motion_blur rotate 
    scale shear shot_noise spatter stripe translate zigzag"

# data
data=mnist
folder_name="mnist_c"

# model parameters
model=cnn_small

# swag
swag_lr=1.0e-2
swag_n_snapshots=40

# Sigma options
p="lowrank-kron,lowrank-diag,subset-diag,subset-magnitude,subset-swag,lowrankoptimal-loadfile"
psi_ref=loadfile
psi_approx=kron
projector_batch_size=10
v_batch_size=10
hessian_name=I60000.pt
jacobian_seed=0


# Note: If the Fisher information matrix for MNIST has been already computed
# it can be copied and does not have to be computed again.
# --------------------------------------------------------
# compute the Hessian or Fisher information matrix. (required for load file)
# CUDA_VISIBLE_DEVICES=$cuda python get_hessian.py -m \
#     seed=$seed \
#     data=$data data.folder_name=$folder_name \
#     pred_model=$model \
#     projector=$projector \
#     projector.batch_size=$projector_batch_size projector.n_batches=null\

# # obtain and plot the spectrum
# CUDA_VISIBLE_DEVICES=$cuda python plot_spectrum_H.py -m \
#     seed=$seed \
#     data=$data data.folder_name=$folder_name \
#     pred_model=$model \
#     projector=$projector
# --------------------------------------------------------


for c in $corruptions
do
    echo "[Considering corruption $c]"
    CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
        data=$data data.folder_name=$folder_name \
        pred_model=$model \
        projector=Sigma\
        projector.s.n=$s_number \
        projector.s.min=$s_min \
        projector.s.max=$s_max \
        projector.sigma.method.p=null \
        projector.sigma.method.psi="$psi_ref,$psi_approx" \
        projector.batch_size=$projector_batch_size \
        projector.v.batch_size=$v_batch_size \
        projector.name_postfix="_c-$c" \
        data.swag_kwargs.swag_lr=$swag_lr \
        data.swag_kwargs.swag_n_snapshots=$swag_n_snapshots \
        data.use_corrupt=$use_corrupt \
        data.name_corrupt=mnist_c-$c \
        projector.jacobian_seed=$jacobian_seed \
        seed=$seed \
        projector.posterior_hessian.load.name=$hessian_name \

    CUDA_VISIBLE_DEVICES=$cuda python get_epistemic_covariance.py -m \
        data=$data data.folder_name=$folder_name \
        pred_model=$model \
        projector=Sigma\
        projector.s.n=$s_number \
        projector.sigma.method.p=$p \
        projector.sigma.method.psi=$psi_ref \
        projector.batch_size=$projector_batch_size \
        projector.v.batch_size=$v_batch_size \
        projector.name_postfix="_c-$c" \
        data.swag_kwargs.swag_lr=$swag_lr \
        data.swag_kwargs.swag_n_snapshots=$swag_n_snapshots \
        data.use_corrupt=$use_corrupt \
        data.name_corrupt=mnist_c-$c \
        projector.posterior_hessian.load.name=$hessian_name \
        seed=$seed \
        projector.jacobian_seed=$jacobian_seed \

    CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
        data=$data data.folder_name=$folder_name \
        pred_model=$model \
        projector=Sigma\
        projector.s.n=$s_number \
        projector.sigma.method.p=$p \
        projector.sigma.method.psi=$psi_ref \
        projector.batch_size=$projector_batch_size \
        projector.v.batch_size=$v_batch_size \
        projector.name_postfix="_c-$c" \
        data.swag_kwargs.swag_lr=$swag_lr \
        data.swag_kwargs.swag_n_snapshots=$swag_n_snapshots \
        data.use_corrupt=$use_corrupt \
        data.name_corrupt=mnist_c-$c \
        projector.posterior_hessian.load.name=$hessian_name \
        seed=$seed

    CUDA_VISIBLE_DEVICES=$cuda python compute_metrics.py -m \
        data=$data data.folder_name=$folder_name \
        pred_model=$model \
        projector=Sigma\
        projector.s.n=$s_number \
        projector.sigma.method.p=null \
        projector.sigma.method.psi=$psi_approx \
        projector.batch_size=$projector_batch_size \
        projector.v.batch_size=$v_batch_size \
        projector.posterior_hessian.load.name=$hessian_name \
        projector.name_postfix="_c-$c" \
        data.swag_kwargs.swag_lr=$swag_lr \
        data.swag_kwargs.swag_n_snapshots=$swag_n_snapshots \
        data.use_corrupt=$use_corrupt \
        data.name_corrupt=mnist_c-$c \
        seed=$seed
done

# plot_corrupted_mnist.py doesn't run with hydra. 
# All configurations are hard coded. Please change them in the script itself
python plot_corrupted_mnist.py
