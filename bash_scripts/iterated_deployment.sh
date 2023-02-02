#!/bin/bash

###RECEIVING ARGS########################################################################
#e, n, a, i are some exemplary parameters forwarded to the target script
#s controls start of a new (iterated) job vs. continuing an existing once
#s is needed for scripts to decide whether to load old data, or create new
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
###END BLOCK##############################################################################

###RUN FIRST ITER WITH FLAG DEPICTING IT IS A NEW START FOR USAGE IN RESPECTIVE SCRIPT###
#check if start from scratch or continue some run that has been deployed earlier
if [ "$extend" = "true" ];
then
  JOBID=$(sbatch --parsable --nice runscript.sh $run_key $num_agents True $experiment)
fi;
if [ "$extend" = "false" ];
then
  JOBID=$(sbatch --parsable --nice runscript.sh $run_key $num_agents False $experiment)
fi;
echo $JOBID
###END BLOCK##############################################################################

###DEPLOY ITERATED RUN SCRIPT WITH NICE FLAG AND CHAINED VIA DEPENDENCY/afterok###########
for (( i=1; i<=$iterations; i++ ))
do
  JOBID=$(sbatch --parsable --nice --dependency=afterok:$JOBID runscript.sh $run_key $num_agents True $experiment)
  echo $JOBID
done
###END BLOCK##############################################################################
