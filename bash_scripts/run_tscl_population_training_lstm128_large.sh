#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu

spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2
python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment tscl_population_training_lstm128 --batch_size 4096 --num_receivers 4 --num_senders 4