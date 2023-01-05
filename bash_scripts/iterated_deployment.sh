#!/bin/bash
while getopts e:n:a:i:s: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;
        n) run_key=${OPTARG};;
        a) num_agents=${OPTARG};;
        i) iterations=${OPTARG};;
        s) extend=${OPTARG};;

    esac
done

#check if start from scratch
if [ extend = "true" ];
then
  JOBID=$(sbatch --parsable --nice runscript.sh $run_key $num_agents False $experiment)
fi;
if [ $extend = "false" ];
then
  JOBID=$(sbatch --parsable --nice runscript.sh $run_key $num_agents True $experiment)
fi;

sleep 1
echo $JOBID

for (( i=1; i<=$iterations; i++ ))
do
  JOBID=$(sbatch --parsable --nice --dependency=afterok:$JOBID runscript.sh $run_key $num_agents True $experiment)
  echo $JOBID
  sleep 1
done