#!/bin/bash

# Experiment script to run try-3.py with varying noise_level_fraction_of_max and noise_seed

for noise_level in $(seq 0.1 0.1 1.0); do
    for noise_seed in {0..20}; do
        echo -e "-------------------\nRunning try-3.py with noise_level_fraction_of_max=$noise_level and noise_seed=$noise_seed\n-------------------"
        python main.py --noise_level_fraction_of_max $noise_level --noise_seed $noise_seed
    done
done

echo "All experiments completed."
# End of script