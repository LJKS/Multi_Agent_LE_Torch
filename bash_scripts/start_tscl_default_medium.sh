#!/bin/bash
#SBATCH --time=3-00:00
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
srun python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment tscl_population_training --num_senders 5 --num_receivers 5 --finetuning_epochs 50 --save_every 1 --run_key $1
#just make sure system got time to save all the models and stuff...
srun sleep 10

for (( i=1; i<=$2; i++ ))
do
  srun python $HOME/experiments/Multi_Agent_LE_Torch/scripts.py --experiment tscl_population_training --num_senders 5 --num_receivers 5 --finetuning_epochs 50 --save_every 1 --run_key $1 --extend True
  #just make sure system got time to save all the models and stuff...
  srun sleep 10
done