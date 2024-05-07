#!/bin/bash

# Run Python files with their respective arguments
python create_datasets.py single_digit_train
python create_datasets.py single_digit_test
python create_datasets.py double_digit_test
python create_datasets.py single_digit_original_train
python create_datasets.py single_digit_original_test
python create_datasets.py double_digit_original_test
