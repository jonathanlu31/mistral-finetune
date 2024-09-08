#! /usr/bin/bash

/home/jonathan_lu/miniconda3/envs/mistral/bin/python -m utils.merge_lora --initial_model_ckpt /mnt/nvme/home/shared_models/mistral_models/Mistral-7B-Instruct-v0.3 --lora_ckpt --dump_ckpt temp_m
/home/jonathan_lu/miniconda3/envs/mistral/bin/python -m utils.convert_mistral_weights_to_hf.py --input_dir temp_m --output_dir mistral_hf --is_v3 --safe_serialization --model_size 7B
rm -rf temp_m
