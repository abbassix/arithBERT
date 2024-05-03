# Numerical Comprehension in LLMs

## Introduction
This project explores the capability of Large Language Models (LLMs), particularly BERT-based models, to understand and perform arithmetic tasks involving numerals. We fine-tuned models on single-digit addition and subtraction and extended our research to cover selective double-digit numerals.

## Datasets
The datasets were constructed using two templates of arithmetic equations involving the operators + (plus) and - (minus), ensuring non-negative results. The training dataset contains 3600 samples, with additional datasets for testing single-digit (1860 samples) and double-digit (7200 samples) numerals.

## Fine-Tuning Parameters
- **Epochs**: 10
- **Batch size**: 32
- **Learning rate**: Started at 2.0e-4, decreasing exponentially to 2.0e-5

## Experimental Setup
### Reframing Numerals
This preprocessing strategy aims to improve numeral understanding by focusing on positional values, breaking down numerals into individual digits (e.g., "23" becomes "2 3").

### Custom Masking Mechanisms
A modified masking approach targets spans of consecutive digits, enhancing the model's ability to understand and predict entire numerals within arithmetic contexts.

## Installation
To set up the project, follow these steps:
1. Clone the repository: `git clone http://github.com/abbassix/arithBERT`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
To run the project, first you need to create the training and test datasets. You will need to have a directory named `datasets` in the parent directoy of the project to store the datasets there. You will also need to have two more directories: `models` to load to store models and `results` to store the resulting accuracies.
To make sure if you have the required directories and if not, to create them, run the following lines.
```bash
chmod +x create_dirs.sh
./create_dirs.sh
```
Then you need to run the following lines to create the training and test datasets.
```bash
python create_datasets.py single_digit_train
python create_datasets.py single_digit_test
python create_datasets.py double_digit_test
```
After creating datasets, you need to have the distilBERTbase model in a directory named `models` again in the parent directory. The arguments to fine-tune the modela are stored in `config.yaml` file. Then run the following line.
```bash
python finetuning.py
```
