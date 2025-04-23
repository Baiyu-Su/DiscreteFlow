#!/bin/bash
#SBATCH -J flow
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/flow.o%j
#SBATCH -e logs/flow.e%j


### Rendezvous info for torch.distributed.run
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=12380

####— Container & Apptainer options —####
CONTAINER=/scratch/10152/baiyusu/pytorch_25.03-py3.sif
APPTAINER_OPTS="--nv \
  --home /scratch/10152/baiyusu \
  --bind /scratch/10152/baiyusu:/scratch/10152/baiyusu \
  --env HOME=/scratch/10152/baiyusu \
  --env PYTHONUSERBASE=/scratch/10152/baiyusu/packages \
  --env XDG_CACHE_HOME=/scratch/10152/baiyusu/.cache \
  --env PIP_NO_CACHE_DIR=1"

export NCCL_IB_DISABLE=1
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=1

# NCCL settings
export NCCL_DEBUG=INFO

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}

####— Load module and go to code directory —####
module load tacc-apptainer
cd /scratch/10152/baiyusu/DiscreteFlow

####— Launch distributed training across all nodes —####
srun apptainer exec $APPTAINER_OPTS $CONTAINER \
  python -u -m torch.distributed.run \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/med_config.py
  "
