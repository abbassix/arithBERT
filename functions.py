# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries for load_model
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

# import the necessary library for load_dataset
from datasets import DatasetDict

# import the necessary libraries for fine_tune
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# import the necessary libraries for accuracy
# and CustomDataCollator
import torch

# import the necessary libraries for
# DataCollatorForLanguageModeling and accuracy
from itertools import groupby
from operator import itemgetter


def find_spans(lst):
    spans = []
    for k, g in groupby(enumerate(lst), key=itemgetter(1)):
        if k:
            glist = list(g)
            spans.append((glist[0][0], len(glist)))

    return spans


class CustomDataCollator(DataCollatorForLanguageModeling):

    mlm: bool = True
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary "
                "for masked language modeling. You should pass `mlm=False` to "
                "train on causal language modeling instead."
            )

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        NOTE: keep `special_tokens_mask` as an argument for avoiding error
        """

        # labels is batch_size x length of the sequence tensor
        # with the original token id
        # the length of the sequence includes the special tokens (2)
        labels = inputs.clone()

        batch_size = inputs.size(0)
        # seq_len = inputs.size(1)
        # in each seq, find the indices of the tokens that represent digits
        dig_ids = [1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]
        dig_idx = torch.zeros_like(labels)
        for dig_id in dig_ids:
            dig_idx += (labels == dig_id)
        dig_idx = dig_idx.bool()
        # in each seq, find the spans of Trues using `find_spans` function
        spans = []
        for i in range(batch_size):
            spans.append(find_spans(dig_idx[i].tolist()))
        masked_indices = torch.zeros_like(labels)
        # spans is a list of lists of tuples
        # in each tuple, the first element is the start index
        # and the second element is the length
        # in each child list, choose a random tuple
        for i in range(batch_size):
            if len(spans[i]) > 0:
                idx = torch.randint(0, len(spans[i]), (1,))
                start, length = spans[i][idx[0]]
                masked_indices[i, start:start + length] = 1
            else:
                print("No digit found in the sequence!")
        masked_indices = masked_indices.bool()

        # We only compute loss on masked tokens
        labels[~masked_indices] = -100

        # change the input's masked_indices to self.tokenizer.mask_token
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels


def load_model(
        model_name: str,
        verbose=True) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load the model and tokenizer from the local machine
    :param model_name: the name of the model
    :return: a tuple of the model and tokenizer
    """

    # load the model from the local machine
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # load the tokenizer from the local machine
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if verbose:
        num_parameters = model.num_parameters() / 1_000_000
        print(f"'>>> {model_name} is loaded.")
        print(f"The number of parameters: {round(num_parameters)}M'")

    return (model, tokenizer)


def load_dataset(dataset_name: str, verbose=True) -> DatasetDict:
    """
    Load the dataset from the local machine
    :param dataset_name: the name of the dataset
    :return: the dataset
    """

    # load the dataset from the local machine
    dataset = DatasetDict.load_from_disk(dataset_name)

    if verbose:
        # print the dataset is loaded
        print(f"'>>> {dataset_name} is loaded.'")

        # print number of rows in train, validation, and test
        train_size = dataset['train'].num_rows
        validation_size = dataset['validation'].num_rows
        test_size = dataset['test'].num_rows
        msg = "Number of rows in train, validation, and test:"
        print(f"'{msg} {train_size}, {validation_size}, {test_size}'")

        # print the first row of the train dataset
        print(f"First row of the train dataset: {dataset['train'][0]}")

    return dataset


def fine_tune(
        model: AutoModelForMaskedLM,
        tokenizer: AutoTokenizer,
        tokenized_dataset: DatasetDict,
        batch_size=32,
        num_epochs=1,
        lr=5e-5,
        collator="custom",
        weight_decay=0.0):
    """
    Fine-tune the model with the tokenized dataset
    :param model: the model
    :param tokenizer: the tokenizer
    :param dataset: the dataset
    :param masking_prob: the masking probability
    :param batch_size: the batch size
    :param num_epochs: the number of epochs
    :param lr: the learning rate
    :param weight_decay: the weight decay
    :return: the fine-tuned model
    """

    if collator == "custom":
        data_collator = CustomDataCollator(tokenizer=tokenizer)
    elif collator == "default":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    else:
        raise ValueError("Invalid collator!")

    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()

    return model


def accuracy(
        model: AutoModelForMaskedLM,
        tokenizer: AutoTokenizer,
        dataset: DatasetDict,
        test_set="test") -> float:
    """
    Calculate the accuracy of the model on the test dataset.
    :param model: the model
    :param tokenizer: the tokenizer
    :param dataset: the dataset
    :param test_set: the test set
    :param plus: whether to replace '+' with 'plus'
    :param equals: whether to replace '=' with 'equals'
    :return: the accuracy
    """

    # number of correct predictions
    n_correct = 0
    # number of uni-masked samples
    len_um = 0
    # number of multi-masked samples
    len_mm = 0
    # number of correct uni-masked predictions
    n_correct_um = 0
    # number of correct multi-masked predictions
    n_correct_mm = 0

    # loop through the test dataset
    for row in dataset[test_set]:
        # to be able to find `[MASK]` when splitting
        row['masked'] = row['masked'].replace(".", " .")
        row['unmasked'] = row['unmasked'].replace(".", " .")

        # get the index of the masked token(s)
        # note that we can have multiple masked tokens
        lst = row['masked'].split()
        first_idx = lst.index("[MASK]")
        last_idx = len(lst) - lst[::-1].index("[MASK]") - 1
        if first_idx == last_idx:
            label = row['unmasked'].split()[first_idx]
        else:
            label = row['unmasked'].split()[first_idx:last_idx + 1]

        # tokenize the text
        inputs = tokenizer(row['masked'], return_tensors="pt")
        # get the model outputs
        outputs = model(**inputs)

        # get the predicted token
        predictions = torch.argmax(outputs.logits, dim=-1)
        msk_id = tokenizer.mask_token_id
        # get the index of the masked token(s)
        if first_idx == last_idx:
            p_msk_idx = torch.where(inputs.input_ids == msk_id)[1]
        else:
            p_msk_idx = torch.where(inputs.input_ids == msk_id)[1]

        label_id = tokenizer.convert_tokens_to_ids(label)

        # check if the predicted token is correct
        if first_idx == last_idx:
            len_um += 1
            n_correct += (predictions[0, p_msk_idx] == label_id).item()
            n_correct_um += (predictions[0, p_msk_idx] == label_id).item()
        else:
            len_mm += 1
            # note that the both sides are tensors
            if torch.equal(predictions[0, p_msk_idx], torch.tensor(label_id)):
                print(f">> Correct! {row['masked']}: {row['unmasked']}")
                n_correct += 1
                n_correct_mm += 1
    accuracy = 100 * n_correct / len(dataset[test_set])

    if len_um == 0:
        acc_um = 0
    else:
        acc_um = 100 * n_correct_um / len_um
    if len_mm == 0:
        acc_mm = 0
    else:
        acc_mm = 100 * n_correct_mm / len_mm

    return (accuracy, len(dataset[test_set]), acc_um, len_um, acc_mm, len_mm)
