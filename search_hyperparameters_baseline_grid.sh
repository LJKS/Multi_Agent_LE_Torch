####This is obsolete, needs fixing for current iteration (changed scripts.py to include args for agent architectures)
#!/bin/bash
for lr in 0.001 0.0001 0.00001 0.000001 0.0000001
do
   for batch_size in 64 128 256 512
do
   for exp in baseline_population_training baseline_population_training
do
   sbatch search_hyperparameters_baseline.sh $exp $batch_size $lr
done
done
done