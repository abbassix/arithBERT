# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries
import yaml
from functions import load_model, load_dataset, fine_tune, accuracy
from transformers import AutoModelForMaskedLM, AutoTokenizer


print("Successfully imported the necessary libraries.\n")

# define the model, train dataset, number of epochs, and learning rates
print("Loading the configuration file...\n")
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_checkpoint = config['model_checkpoint']
train_dataset_name = config['train_dataset_name']
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lrs = config['lrs']
collator = config['collator']
single_digit_test_name = config['single-digit_test_dataset_name']
double_digit_test_name = config['double-digit_test_dataset_name']

model_name = model_checkpoint.split("/")[-1]

model_path = f"../models/{model_name}"
train_dataset_path = f"../datasets/{train_dataset_name}"

print("Loading the model and the train dataset...\n")
# check if the model exists in the models directory
# if it does not exist, download it
try:
    model, tokenizer = load_model(model_path)
except FileNotFoundError:
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_dataset = load_dataset(train_dataset_path)


def tokenize_label(example: dict) -> dict:
    """
    Tokenize the label columns
    """

    return tokenizer(example["unmasked"])


print("Tokenizing the train dataset...\n")
tokenized_dataset = train_dataset.map(
    tokenize_label,
    batched=True,
    remove_columns=["unmasked", "masked"])

single_digit_test_path = f"../datasets/{single_digit_test_name}"
double_digit_test_path = f"../datasets/{double_digit_test_name}"

print("Loading the test datasets...\n")
single_digit_test = load_dataset(single_digit_test_path)
double_digit_test = load_dataset(double_digit_test_path)

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

print(f"single-digit test accuracy: {single_digit_test_acc}")
print(f"double-digit test accuracy: {double_digit_test_acc}")

# save the model and the tokenizer
print("Saving the model and the tokenizer...\n")
model.save_pretrained(f"../models/{model_name}_{train_dataset_name}")
tokenizer.save_pretrained(f"../models/{model_name}_{train_dataset_name}")

# save the accuracy results in a YAML file
print("Saving the accuracy results...\n")
with open(f"../results/{model_name}_{train_dataset_name}.yaml", 'w') as file:
    yaml.dump({
        "single-digit_acc": single_digit_test_acc,
        "double-digit_acc": double_digit_test_acc
    }, file)
