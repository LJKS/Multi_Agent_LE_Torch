#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2
python experiments/Multi_Agent_LE_Torch/scripts.py --experiment baseline_population_training_lstm128 --finetuning_epochs 1000 --batch_size 4096 --num_senders 8 --num_receivers 8