#!/bin/bash
#SBATCH -J download
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH -o logs/download.o%j  # Output file
#SBATCH -e logs/download.e%j   # Error file

module load tacc-apptainer

SCRATCH=/scratch/10152/baiyusu
CONTAINER=$SCRATCH/pytorch_24.12-py3.sif
export PYTHONUSERBASE=/scratch/10152/baiyusu/packages
export PATH="$PYTHONUSERBASE/bin:$PATH"

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}
export WANDB_API_KEY=a95d91ca9178e628b73ef33438d5a81306701c3b

srun --mpi=pmi2 \
     apptainer exec --nv \
        --home "$SCRATCH" \
        --bind "$SCRATCH:$SCRATCH" \
        --env PYTHONUSERBASE="$PYTHONUSERBASE" \
        --env XDG_CACHE_HOME="$SCRATCH/.cache" \
        --env PIP_NO_CACHE_DIR=1 \
        "$CONTAINER" \
        torchrun \
          --standalone \
          --nproc_per_node=1 \
          train.py --config configs/small_config.py