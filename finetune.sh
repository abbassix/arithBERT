#!/bin/bash

# Run Python files with their respective arguments
# to fine-tune three models
python finetuning.py compare_reframing
python finetuning.py compare_masking
python finetuning.py compare_models
