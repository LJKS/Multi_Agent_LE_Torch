#!/bin/bash
#SBATCH --time=0-07:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc
spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2
#args: $1 --> run_key (name), $2 num_agents (int), $3 extend (bool), $4 experiment (function name)
echo "--experiment $4 --num_senders $2 --num_receivers $2 --finetuning_epochs 50 --run_key $1 --extend $3"
srun python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment $4 --num_senders $2 --num_receivers $2 --finetuning_epochs 50 --run_key $1 --extend $3
