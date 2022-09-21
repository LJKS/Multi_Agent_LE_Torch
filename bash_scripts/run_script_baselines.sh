#!/bin/bash
# submission scripts are structured as  | script_name | iterations | run_name |
# notice: each iteration is 50 epochs (+ one additional starting iteration of also 50 epochs), run_names make it easy to track experiments by name
sbatch start_baseline_small 9 kermes_whale
sbatch start_baseline_small 9 pyrrhous_duck
sbatch start_baseline_small 9 nigrine_wolf
sbatch start_baseline_medium 49 titian_sparrow
sbatch start_baseline_medium 49 brunneous_shrimp
sbatch start_baseline_medium 49 rhodopsin_turtle
# these might be too short, but I do not want to create jobs that are longer than 4 days really.
sbatch start_baseline_large 49 meline_monkey
sbatch start_baseline_large 49 taupe_ostrich
sbatch start_baseline_large 49 miniaceous_kangaroo