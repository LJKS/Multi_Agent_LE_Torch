#!/bin/bash
for lr in 0.001 0.0001 0.00001 0.000001 0.0000001
do
   for batch_size in 64 128 256 512
do
   for exp in baseline_population_training_lstm128 baseline_population_training_lstm64
do
   sbatch search_hyperparameters_baseline.sh $exp $batch_size $lr
done
done
done