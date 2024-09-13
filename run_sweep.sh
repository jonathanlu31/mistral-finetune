#!/bin/bash

yaml_files=(
    "example/7B_lowercase_ultra.yaml"
)
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Loop through each YAML file
for yaml_file in "${yaml_files[@]}"
do
    echo "Running torchrun with $yaml_file"
    /home/jonathan/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 4 --master_port $RANDOM -m train $yaml_file
    
    echo "-----------------------------------"
done

echo "All commands have been executed"
