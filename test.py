import random


def order_ds(n_samples, ceil):
    """
    Generate a dataset for the order task
    :param n_samples: number of samples to generate
    :param ceil: the maximum number to choose from
    :return: a list of tuples, each tuple contains a
    string of numbers and a string of masked numbers
    """
    dataset = []
    for _ in range(n_samples):
        n = random.randint(3, 7)
        numbers = list(range(1, ceil))
        start_idx = random.randint(1, ceil - n)
        numbers = numbers[start_idx:start_idx + n]
        mask_index = random.randint(1, n - 2)
        masked_numbers = numbers.copy()
        masked_numbers[mask_index] = '[MASK]'
        dataset.append((', '.join(map(str, numbers)), ', '.join(map(str, masked_numbers))))
    return dataset

ds = order_ds(20, 100)

for tup in ds:
    print(tup)
