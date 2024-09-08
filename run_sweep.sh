#!/bin/bash

yaml_files=(
    "example/7B_sweeps/7B_lr_3e-5_r_128.yaml"
)
# Loop through each YAML file
for yaml_file in "${yaml_files[@]}"
do
    echo "Running torchrun with $yaml_file"
    /home/jonathan_lu/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 4 --master_port $RANDOM -m train $yaml_file
    
    echo "-----------------------------------"
done

echo "All commands have been executed"
