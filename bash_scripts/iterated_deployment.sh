#!/bin/bash
# args: $1 --> run_key (name), $2 num_agents (int), $3 how many iterations beyond the first should be deployed
JOBID=$(sbatch runscript.sh $1 $2 False --parsable)

for (( i=1; i<=$3; i++ ))
do
  JOBID=$(sbatch --parsable --dependency=afterok:${JOBID##* }} runscript.sh $1 $2 True)
  echo $JOBID
done