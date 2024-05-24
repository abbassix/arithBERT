# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries
import sys
import yaml
import random
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import DatasetDict


# add random seed for reproducibility
random.seed(42)


def reframe(numeral):
    """
    reframes the numeral by separating the digits with spaces
    :param numeral: the numeral to be reframed
    :return: the reframed numeral
    """

    # convert the numeral to a string
    numeral = str(numeral)

    # separate the digits with spaces
    reframed = " ".join(numeral)

    return reframed


def save_dataset(umsk, msk, name, n_train, n_test, val=0.1):
    """
    creates a Hugging Face dataset from a list of umsk and msk strings
    :param umsk: a list of umsk strings
    :param msk: a list of msk strings
    :param n: the number of samples to be generated
    :param name: the name of the dataset
    :return: None
    """

    # zip the msk and umsk lists together
    z = list(zip(umsk, msk))

    # randomly select n_samples from the zipped list
    if len(z) == n_train:
        train_zip = z
    elif len(z) > n_train:
        train_zip = random.sample(z, n_train)
    elif len(z) < n_train:
        random.shuffle(z)
        train_zip = z * (n_train // len(z)) + z[:n_train % len(z)]
    else:
        print("The number of training samples is not correct.")

    if len(z) == n_test:
        test_zip = z
    elif len(z) > n_test:
        test_zip = random.sample(z, n_test)
    elif len(z) < n_test:
        random.shuffle(z)
        test_zip = z * (n_test // len(z)) + z[:n_test % len(z)]
    else:
        print("The number of test samples is not correct.")

    # shuffle the zipped list
    random.shuffle(train_zip)
    random.shuffle(test_zip)

    # unzip the zipped list
    train_umsk, train_msk = zip(*train_zip)
    test_umsk, test_msk = zip(*test_zip)

    # convert the umsk and msk lists to a list
    train_umsk = list(train_umsk)
    train_msk = list(train_msk)
    test_umsk = list(test_umsk)
    test_msk = list(test_msk)

    # create a dictionary to store the msk and umsk lists
    train_ds_dic = {
        "unmasked": train_umsk,
        "masked": train_msk
    }
    test_ds_dic = {
        "unmasked": test_umsk,
        "masked": test_msk
    }

    # create a Hugging Face dataset from the dictionary
    train_ds = Dataset.from_dict(train_ds_dic)
    test_ds = Dataset.from_dict(test_ds_dic)

    # Split the dataset into train, validation, and test sets
    split_dataset = train_ds.train_test_split(test_size=val)

    # Reassign the splits to a more understandable structure
    split_dataset['validation'] = split_dataset['test']
    split_dataset['test'] = test_ds
    split_dataset['train'] = train_ds

    # Now you have train, test, and validation subsets
    print("Train Dataset:", split_dataset['train'])
    print("Validation Dataset:", split_dataset['validation'])
    print("Test Dataset:", split_dataset['test'])

    # dataset path is in datasets directory on the parent directory
    dataset_path = f"../datasets/{name}"
    # save the dataset to a file
    split_dataset.save_to_disk(dataset_path)


def merge_datasets(datasets: list, name: str):
    """
    merges a list of datasets and saves the merged dataset to a file
    :param dataset1: the first dataset
    :param dataset2: the second dataset
    :param name: the name of the merged dataset
    :return: None
    """
    for i, dataset in enumerate(datasets):
        if i == 0:
            conc_ds = DatasetDict.load_from_disk(f"../datasets/{dataset}")
        else:
            ds = DatasetDict.load_from_disk(f"../datasets/{dataset}")
            conc_ds = {split: concatenate_datasets([conc_ds[split], ds[split]])
                       for split in conc_ds.keys()}

    ds_fin = DatasetDict(conc_ds)

    # save the merged dataset to a file
    path = f"../datasets/{name}"
    ds_fin.save_to_disk(path)


def _upd_unmask_norm(lst, op, eq, op1, op2, res):
    """
    updates the umsk list with the given operation, operands, and result
    :param lst: the list of umsk strings
    :param op: the operation to be performed
    :param eq: the equality sign
    :param op1: the first operand
    :param op2: the second operand
    :param res: the result of the operation
    :return: the updated list of umsk strings
    """
    lst.append(f"{op1} {op} {op2} {eq} {res}.")
    lst.append(f"{op1} {op} {op2} {eq} {res}.")
    lst.append(f"{op1} {op} {op2} {eq} {res}.")

    return lst


def _upd_mask_norm(lst, op, eq, op1, op2, res):
    """
    updates the msk list with the given operation, operands, and result
    :param lst: the list of msk strings
    :param op: the operation to be performed
    :param eq: the equality sign
    :param op1: the first operand
    :param op2: the second operand
    :param res: the result of the operation
    :return: the updated list of msk strings
    """
    # make sure all operands and the result are strings
    op1 = str(op1)
    op2 = str(op2)
    res = str(res)

    len_op1 = len(op1.split())
    len_op2 = len(op2.split())
    len_res = len(res.split())
    msk_op1 = " ".join(["[MASK]"] * len_op1)
    msk_op2 = " ".join(["[MASK]"] * len_op2)
    msk_res = " ".join(["[MASK]"] * len_res)
    lst.append(f"{msk_op1} {op} {op2} {eq} {res}.")
    lst.append(f"{op1} {op} {msk_op2} {eq} {res}.")
    lst.append(f"{op1} {op} {op2} {eq} {msk_res}.")

    return lst


def _upd_unmask_rev(lst, op, eq, op1, op2, res):
    """
    updates the umsk list with the given operation, operands, and result
    :param lst: the list of umsk strings
    :param op: the operation to be performed
    :param eq: the equality sign
    :param op1: the first operand
    :param op2: the second operand
    :param res: the result of the operation
    :return: the updated list of umsk strings
    """
    lst.append(f"{res} {eq} {op1} {op} {op2}.")
    lst.append(f"{res} {eq} {op1} {op} {op2}.")
    lst.append(f"{res} {eq} {op1} {op} {op2}.")

    return lst


def _upd_mask_rev(lst, op, eq, op1, op2, res):
    """
    updates the msk list with the given operation, operands, and result
    :param lst: the list of msk strings
    :param op: the operation to be performed
    :param eq: the equality sign
    :param op1: the first operand
    :param op2: the second operand
    :param res: the result of the operation
    :return: the updated list of msk strings
    """
    # make sure all operands and the result are strings
    op1 = str(op1)
    op2 = str(op2)
    res = str(res)
    
    len_op1 = len(op1.split())
    len_op2 = len(op2.split())
    len_res = len(res.split())
    msk_op1 = " ".join(["[MASK]"] * len_op1)
    msk_op2 = " ".join(["[MASK]"] * len_op2)
    msk_res = " ".join(["[MASK]"] * len_res)
    lst.append(f"{msk_res} {eq} {op1} {op} {op2}.")
    lst.append(f"{res} {eq} {msk_op1} {op} {op2}.")
    lst.append(f"{res} {eq} {op1} {op} {msk_op2}.")

    return lst


def _upd(op, eq, op1, op2, res, umsk, msk, rev):
    """
    updates the umsk and msk lists with the given
    operation, operands, and result
    :param op: the operation to be performed
    :param eq: the equality sign
    :param op1: the first operand
    :param op2: the second operand
    :param res: the result of the operation
    :param umsk: the list of umsk strings
    :param msk: the list of msk strings
    :param rev: the direction of the operation
    :return: the updated umsk and msk lists
    """
    if rev == "both":
        umsk = _upd_unmask_norm(umsk, op, eq, op1, op2, res)
        msk = _upd_mask_norm(msk, op, eq, op1, op2, res)
        umsk = _upd_unmask_rev(umsk, op, eq, op1, op2, res)
        msk = _upd_mask_rev(msk, op, eq, op1, op2, res)
    elif rev == "no":
        umsk = _upd_unmask_norm(umsk, op, eq, op1, op2, res)
        msk = _upd_mask_norm(msk, op, eq, op1, op2, res)
    elif rev == "yes":
        umsk = _upd_unmask_rev(umsk, op, eq, op1, op2, res)
        msk = _upd_mask_rev(msk, op, eq, op1, op2, res)

    return umsk, msk


def ad_ds(floor, ceil, name, ops="both", rev="both",
          ref=True, list_pos=None, list_neg=None):
    """
    generates a two lists of addition problems and their solutions
    and calls the save_dataset function to save the dataset to a file
    :param ceil: the maximum number to be added
    :param name: the name of the dataset
    :param n_samples: the number of samples to be generated
    :param floor: the minimum number to be added
    :return: umsk, msk
    """
    umsk = []
    msk = []
    if list_pos is not None and list_neg is not None:
        range_pos = list_pos
    else:
        range_pos = range(floor, ceil)
        list_neg = []
    if ops == "sign":
        opeq = [("+", "=")]
    elif ops == "word":
        opeq = [("plus", "equals")]
    elif ops == "both":
        opeq = [("+", "="), ("plus", "equals")]

    # iterate over given range of numbers and generate addition problems
    for (op, eq) in opeq:
        for op1 in range_pos:
            for op2 in range_pos:
                res = op1 + op2
                if res in list_neg:
                    continue
                if ref:
                    _op1 = reframe(op1)
                    op2 = reframe(op2)
                    res = reframe(res)
                    umsk, msk = _upd(op, eq, _op1, op2, res, umsk, msk, rev)
                else:
                    umsk, msk = _upd(op, eq, op1, op2, res, umsk, msk, rev)

    # print the number of samples created
    print(f"We generated {len(umsk)} samples for {name}.")

    return umsk, msk


def sub_ds(floor, ceil, name, ops="both", rev="both",
           ref=True, list_pos=None, list_neg=None):
    """
    generates a two lists of addition problems and their solutions
    and calls the save_dataset function to save the dataset to a file
    :param ceil: the maximum number to be added
    :param name: the name of the dataset
    :param n_samples: the number of samples to be generated
    :param floor: the minimum number to be added
    :return: umsk, msk
    """
    umsk = []
    msk = []

    if list_pos is not None and list_neg is not None:
        range_pos = list_pos
    else:
        range_pos = range(floor, ceil)
        list_neg = []

    if ops == "sign":
        opeq = [("-", "=")]
    elif ops == "word":
        opeq = [("minus", "equals")]
    elif ops == "both":
        opeq = [("-", "="), ("minus", "equals")]

    # iterate over given range of numbers and generate subtraction problems
    for (op, eq) in opeq:
        for op1 in range_pos:
            for op2 in range_pos:
                res = op1 - op2
                if res in list_neg or res < 0:
                    continue
                if ref:
                    _op1 = reframe(op1)
                    op2 = reframe(op2)
                    res = reframe(res)
                    umsk, msk = _upd(op, eq, _op1, op2, res, umsk, msk, rev)
                else:
                    umsk, msk = _upd(op, eq, op1, op2, res, umsk, msk, rev)

    # print the number of samples created
    print(f"We generated {len(umsk)} samples for {name}.")

    return umsk, msk


def gen_ds(
        floor, ceil, name, n_train, n_test,
        op="both", ops="both", rev="both",
        ref=True, list_pos=None, list_neg=None):
    """
    generates a dataset of addition and subtraction problems
    """
    if op == "add":
        umsk, msk = ad_ds(floor, ceil, name, ops, rev,
                          ref, list_pos=list_pos, list_neg=list_neg)
    elif op == "sub":
        umsk, msk = sub_ds(floor, ceil, name, ops, rev,
                           ref, list_pos=list_pos, list_neg=list_neg)
    elif op == "both":
        umsk_ad, msk_ad = ad_ds(floor, ceil, name, ops, rev,
                                ref, list_pos=list_pos, list_neg=list_neg)
        umsk_su, msk_su = sub_ds(floor, ceil, name, ops, rev,
                                 ref, list_pos=list_pos, list_neg=list_neg)
        umsk = umsk_ad + umsk_su
        msk = msk_ad + msk_su

    # save the dataset to a file
    save_dataset(umsk, msk, name, n_train, n_test)


def main():
    """
    the main function
    :param config: the configuration dictionary
    :return: None
    """
    # YAML file name is given as an argument
    yaml_file = sys.argv[1]
    # load the configuration file
    if yaml_file[-5:] == ".yaml":
        yaml_file = yaml_file[:-5]
    with open(f"{yaml_file}.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # load the configuration parameters
    floor = config['floor']
    ceil = config['ceil']
    name = config['name']
    n_train = config['n_train']
    n_test = config['n_test']
    op = config['op']
    ops = config['ops']
    rev = config['rev']
    ref = config['ref']
    inst = config['instructions']

    if inst is None:
        gen_ds(floor, ceil, name, n_train, n_test, op,
               ops, rev, ref)
    elif inst[0] == 's':
        print(f'The given instructions are: {inst}')
        # split and map integers
        floor, ceil, train_ratio, test_ratio = map(int, inst[1:].split('-'))
        train_ratio = train_ratio / 100
        test_ratio = test_ratio / 100
        list_ = list(range(floor, ceil))
        random.shuffle(list_)
        train_list = list_[:int(train_ratio * len(list_))]
        train_list += [i for i in range(0, floor)]
        print(train_list)
        test_list = list_[-int(test_ratio * len(list_)):]
        name_train = f"{name}_train"
        name_test = f"{name}_test"
        # generate training dataset
        gen_ds(floor, ceil, name_train, n_train, n_train, op,
               ops, rev, ref, list_pos=train_list, list_neg=test_list)
        # generate test dataset
        gen_ds(floor, ceil, name_test, n_test, n_test, op,
               ops, rev, ref, list_pos=test_list, list_neg=train_list)
    elif inst[0] == 'm':
        print(f'The given instructions are: {inst}')
        # merge the datasets
        datasets = inst[1:].split('-')
        merge_datasets(datasets, name)
    else:
        print("The instructions are not correct.")


if __name__ == '__main__':
    main()
