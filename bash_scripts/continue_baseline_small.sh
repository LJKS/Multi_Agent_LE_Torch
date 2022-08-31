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
#run 50 epochs per step
for i in {1..$2}
do
  srun python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment baseline_population_training --num_senders 2 --num_receivers 2 --finetuning_epochs 50 --save_every 1 --run_key $1 --extend True
  #just make sure system got time to save all the models and stuff...
  sleep 10
done