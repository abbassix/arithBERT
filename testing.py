# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries
import sys
import yaml
import warnings
import logging
from functions import load_model, load_dataset, accuracy


logging.basicConfig(level=logging.WARNING)


def tokenize_label(example: dict) -> dict:
    """
    Tokenize the label columns
    """

    return tokenizer(example["unmasked"])


print("Successfully imported the necessary libraries.\n")

# receive the configuration file name from the first argument
config_file = sys.argv[1]

print("Loading the configuration file...\n")
# check for errors in the configuration file
if config_file[-5:] == ".yaml":
    config_file = config_file[:-5]
try:
    with open(f'{config_file}.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError as e:
    warnings.warn(
        f"Error: The configuration file '{config_file}.yaml' does not exist."
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

model_checkpoint = config['model_checkpoint']
test_dataset_name = config['single-digit_test_dataset_name']

model_name = model_checkpoint.split("/")[-1]

print("Loading the model...\n")
try:
    model, tokenizer = load_model(model_checkpoint)
except FileNotFoundError as e:
    warnings.warn(
        f"Error: The model checkpoint '{model_checkpoint}' does not exist."
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

test_dataset_path = f"../datasets/{test_dataset_name}"

print("Loading the test datasets...\n")
try:
    test_dataset = load_dataset(test_dataset_path)
except FileNotFoundError as e:
    warnings.warn(
        f"""
        Error: The test datasets '{test_dataset_path}' do not exist.
        """
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

print("Calculating the initial test accuracies...\n")
test_dataset_accuracy = accuracy(model, tokenizer, test_dataset)

print(f"test dataset accuracy: {test_dataset_accuracy}")
