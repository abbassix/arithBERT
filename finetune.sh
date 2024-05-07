#!/bin/bash

# Run Python files with their respective arguments
# to fine-tune three models
python finetuning.py finetune_configs/baseline
python finetuning.py finetune_configs/compare_reframing
python finetuning.py finetune_configs/compare_masking
python finetuning.py finetune_configs/compare_models
