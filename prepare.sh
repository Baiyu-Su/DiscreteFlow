#!/bin/bash
#SBATCH -J NAMD
#SBATCH -p gg
#SBATCH -t 24:00:00
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/flow_prepare.o%j  # Output file
#SBATCH -e logs/flow_prepare.e%j   # Error file

source ~/.bashrc
conda deactivate
cd /scratch/10152/baiyusu/DiscreteFlow
conda activate flowenv

GPUS_PER_NODE=1

# Create temp directory for torch inductor cache
mkdir -p /tmp/torchinductor_cache_${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Not using distributed training for debug
python3 train.py --config configs/med_config.py --per_device_train_batch_size=4 --gradient_checkpointing=True
