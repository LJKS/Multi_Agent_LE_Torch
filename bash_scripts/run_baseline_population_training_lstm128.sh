#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc
spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2
python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment baseline_population_training_lstm128 --finetuning_epochs 1000 --batch_size 4096


