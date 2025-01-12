#!/bin/bash
#SBATCH -J llama
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/large_train_adamw.o%j  # Output file
#SBATCH -e logs/large_train_adamw.e%j   # Error file

source ~/.bashrc
conda deactivate
cd /scratch/10152/baiyusu/DiscreteFlow
conda activate flowenv

# Define MASTER_ADDR and MASTER_PORT
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname | head -n1)
MASTER_PORT=12355

export NCCL_IB_DISABLE=0
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=1

# NCCL settings
export NCCL_DEBUG=INFO

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}

# Not using distributed training for debug
torchrun --nproc_per_node=1 train.py --config configs/config.py
