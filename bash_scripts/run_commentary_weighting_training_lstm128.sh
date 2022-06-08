#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2
python experiments/Multi_Agent_LE_Torch/scripts.py --experiment commentary_weighting_training_lstm128 --batch_size 1024