#!/bin/bash
#SBATCH -J llama
#SBATCH -p gpu-a100
#SBATCH -t 48:00:00
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks-per-node=3
#SBATCH -o logs/large_train_adamw.o%j  # Output file
#SBATCH -e logs/large_train_adamw.e%j   # Error file

source ~/.bashrc
conda deactivate
cd /scratch/10152/baiyusu/litgpt
conda activate llamaenv

# Define MASTER_ADDR and MASTER_PORT
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname | head -n1)
MASTER_PORT=12355

export NCCL_IB_DISABLE=0
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=3

# NCCL settings
export NCCL_DEBUG=INFO

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}

echo "Checking GPUs on each node..."
srun -l bash -c 'hostname; nvidia-smi -L'

torchrun --nproc_per_node=2 train.py --config configs/config.py
