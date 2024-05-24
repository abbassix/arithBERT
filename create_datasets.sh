#!/bin/bash

# Run Python files with their respective arguments
# to create six datasets
python create_datasets.py dataset_configs/single_digit_train
python create_datasets.py dataset_configs/single_digit_test
python create_datasets.py dataset_configs/double_digit_test
python create_datasets.py dataset_configs/single_digit_original_train
python create_datasets.py dataset_configs/single_digit_original_test
python create_datasets.py dataset_configs/double_digit_original_test
python create_datasets.py dataset_configs/double-digit_new