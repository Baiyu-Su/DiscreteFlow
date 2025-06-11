#!/bin/bash
#SBATCH -J olmo
#SBATCH -t 50:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8        # Request 8 GPUs on the node
#SBATCH --ntasks-per-node=8                  # 1 task per GPU
#SBATCH --cpus-per-task=12                   # 96 / 8 = 12 CPUs per task
#SBATCH -o logs/flow.o%j
#SBATCH -e logs/flow.e%j

source ~/.bashrc
conda activate olmoenv
cd /mnt/weka/home/lzchen/bscode/DiscreteFlow

export WANDB_API_KEY=a95d91ca9178e628b73ef33438d5a81306701c3b

/mnt/weka/home/lzchen/miniconda3/envs/olmoenv/bin/python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 1 \
    train.py --config configs/med_config.py
