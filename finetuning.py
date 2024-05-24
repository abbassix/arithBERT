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
from functions import load_model, load_dataset, fine_tune, accuracy


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
if config_file[-5:] == ".yaml":
    config_file = config_file[:-5]
# check for errors in the configuration file
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
train_dataset_name = config['train_dataset_name']
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lrs = config['lrs']
collator = config['collator']
single_digit_test_name = config['single-digit_test_dataset_name']
double_digit_test_name = config['double-digit_test_dataset_name']

model_name = model_checkpoint.split("/")[-1]
train_dataset_path = f"../datasets/{train_dataset_name}"

print("Loading the model and the train dataset...\n")
try:
    model, tokenizer = load_model(model_checkpoint)
except FileNotFoundError as e:
    warnings.warn(
        f"Error: The model checkpoint '{model_checkpoint}' does not exist."
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

try:
    train_dataset = load_dataset(train_dataset_path)
except FileNotFoundError as e:
    warnings.warn(
        f"Error: The train dataset '{train_dataset_path}' does not exist."
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

print("Tokenizing the train dataset...\n")
tokenized_dataset = train_dataset.map(
    tokenize_label,
    batched=True,
    remove_columns=["unmasked", "masked"])

single_digit_test_path = f"../datasets/{single_digit_test_name}"
double_digit_test_path = f"../datasets/{double_digit_test_name}"

print("Loading the test datasets...\n")
try:
    single_digit_test = load_dataset(single_digit_test_path)
    double_digit_test = load_dataset(double_digit_test_path)
except FileNotFoundError as e:
    warnings.warn(
        f"""
        Error: The test datasets '{single_digit_test_name}' or
        '{double_digit_test_name}' do not exist.
        """
        )
    logging.error(f"Encountered a problem: {str(e)}")
    sys.exit(1)

single_digit_test_acc = []
double_digit_test_acc = []

print("Calculating the initial test accuracies...\n")
single_digit_test_acc.append(accuracy(model, tokenizer, single_digit_test))
double_digit_test_acc.append(accuracy(model, tokenizer, double_digit_test))
print(f"single-digit test accuracy: {single_digit_test_acc}")
print(f"double-digit test accuracy: {double_digit_test_acc}")

print("Fine-tuning the model...\n")
for i in range(n_epochs):
    if i == 0:
        print(f"Fine-tuning first epoch of {n_epochs}.")
    elif i == 1:
        print(f"Fine-tuning second epoch of {n_epochs}.")
    elif i == 2:
        print(f"Fine-tuning third epoch of {n_epochs}.")
    else:
        print(f"Fine-tuning {i+1}th epoch of {n_epochs}.")
    # in case the number of epochs is more than the number of
    # learning rates provided use the last learning rate
    if i >= len(lrs):
        warnings.warn(
            "Warning: The number of epochs is more than the number"
            "of learning rates provided."
            )
        logging.warning(
            "Using the last learning rate for the rest of the epochs."
            )
        lr = lrs[-1]
    # otherwise, use the corresponding learning rate
    else:
        lr = lrs[i]
    model = fine_tune(
        model,
        tokenizer,
        tokenized_dataset,
        batch_size=batch_size,
        num_epochs=1,
        lr=lr,
        collator=collator,
        weight_decay=0.0)

    single_digit_test_acc.append(accuracy(model, tokenizer, single_digit_test))
    double_digit_test_acc.append(accuracy(model, tokenizer, double_digit_test))

    print(f"single-digit test accuracy: {single_digit_test_acc[-1]}")
    print(f"double-digit test accuracy: {double_digit_test_acc[-1]}")

print("Fine-tuning completed.\n")

# save the model and the tokenizer
print("Saving the model and the tokenizer...\n")
saving_path = f"../models/{model_name}_{train_dataset_name}_{collator}"
model.save_pretrained(saving_path)
tokenizer.save_pretrained(saving_path)

# save the accuracy results in a YAML file
print("Saving the accuracy results...\n")
results_file = f"../results/{model_name}_{collator}_{train_dataset_name}.yaml"
with open(results_file, 'w') as file:
    yaml.dump({
        "single-digit_accuracy": single_digit_test_acc,
        "double-digit_accuracy": double_digit_test_acc
    }, file)
