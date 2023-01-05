#!/bin/bash

# 2x2 baseline runs, 200 epochs each
bash iterated_deployment.sh -e tscl_population_training -n salty_lightning -a 2 -i 4 -s true
bash iterated_deployment.sh -e tscl_population_training -n dazzling_hawk -a 2 -i 4 -s true
# 5x5 baseline runs, 1000 epochs each
bash iterated_deployment.sh -e tscl_population_training -n basic_behemoth -a 5 -i 20 -s true
bash iterated_deployment.sh -e tscl_population_training -n amazing_guardian -a 5 -i 20 -s true
# 10x10 baseline runs, 2500 epochs each
bash iterated_deployment.sh -e tscl_population_training -n hard_rose -a 10 -i 100 -s true
bash iterated_deployment.sh -e tscl_population_training -n impure_cayman -a 10 -i 100 -s true
# 15x15 baseline runs, 5000 epochs each
bash iterated_deployment.sh -e tscl_population_training -n agile_neptune -a 15 -i 200 -s true
bash iterated_deployment.sh -e tscl_population_training -n reckless_ghost -a 15 -i 200 -s true

# 2x2 baseline runs, 200 epochs each
bash iterated_deployment.sh -e baseline_population_training -n dizzy_author -a 2 -i 4 -s true
bash iterated_deployment.sh -e baseline_population_training -n calm_castle -a 2 -i 4 -s true
# 5x5 baseline runs, 1000 epochs each
bash iterated_deployment.sh -e baseline_population_training -n quiet_lancer -a 5 -i 20 -s true
bash iterated_deployment.sh -e baseline_population_training -n father_carpenter -a 5 -i 20 -s true
# 10x10 baseline runs, 2500 epochs each
bash iterated_deployment.sh -e baseline_population_training -n arctic_demon -a 10 -i 100 -s true
bash iterated_deployment.sh -e baseline_population_training -n twin_foxtail -a 10 -i 100 -s true
# 15x15 baseline runs, 5000 epochs each
bash iterated_deployment.sh -e baseline_population_training -n rich_flamingo -a 15 -i 200 -s true
bash iterated_deployment.sh -e baseline_population_training -n virtual_hunter -a 15 -i 200 -s true

