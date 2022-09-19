#!/bin/bash
#SBATCH --time=5-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc
spack load miniconda3@4.10.3
conda activate Multi_Agent_LE_Torch2

for runi in {1..3}
do
  for epsilon_val in 0.05 0.1 0.15 0.2
  do
    echo "tscl_epsilon${epsilon_val}_run${runi}"
    srun python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment tscl_population_training --num_senders 5 --num_receivers 5 --finetuning_epochs 150 --save_every 50 --run_key "tscl_epsilon${epsilon_val}_run${runi}_medium" --tscl_sampling epsilon_greedy --tscl_epsilon $epsilon_val

  done
done