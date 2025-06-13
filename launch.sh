#!/bin/bash

#SBATCH --job-name=gumbel-flow                                                       
#SBATCH --partition=dgx                                      
#SBATCH --nodes=1                                            
#SBATCH --ntasks-per-node=1                                  
#SBATCH --gres=gpu:1                                         
#SBATCH --mem=50G                                            
#SBATCH --cpus-per-task=8                                    
#SBATCH --time=100:00:00  

#SBATCH --output=/u/chizhang/Projects/DiscreteFlow/logs/%x-%j.out
#SBATCH --error=/u/chizhang/Projects/DiscreteFlow/logs/%x-%j.err

bash fineweb_run.sh