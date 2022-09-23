#!/bin/bash
# submission scripts are structured as  | script_name | iterations | run_name |
# notice: each iteration is 50 epochs (+ one additional starting iteration of also 50 epochs), run_names make it easy to track experiments by name
sbatch start_baseline_small.sh kermes_whale 9
sbatch start_baseline_small.sh pyrrhous_duck 9
sbatch start_baseline_small.sh nigrine_wolf 9
sbatch start_baseline_medium.sh titian_sparrow 49
sbatch start_baseline_medium.sh brunneous_shrimp 49
sbatch start_baseline_medium.sh rhodopsin_turtle 49
# these might be too short, but I do not want to create jobs that are longer than 4 days really.
sbatch start_baseline_large.sh meline_monkey 49
sbatch start_baseline_large.sh taupe_ostrich 49
sbatch start_baseline_large.sh miniaceous_kangaroo 49